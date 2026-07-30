#!/usr/bin/env python3
"""
construir_submission_final_ensemble.py
======================================

Submission final para el validation oficial, combinando lo mejor de cada parte:

  BASE: ensemble 5-fold del fine-tuning hard (predicciones sobre el oficial).
  WT   = (ensemble_hard >= 1) OR (soft_TC > THR_SOFT)   [via B, thr 0.5]
  TC   = del ensemble hard
  ET   = del ensemble hard (con prioridad en el ensamblado)
  ED   = ensemble_hard whole-tumour menos el core (edema del hard)
  CC   = logistica de intensidad thr 0.90 sobre el NET

Remapeo a etiquetas oficiales: ET=1, NET=2, CC=3, ED=4.
Salida: carpeta + zip plano listo para Synapse.

REQUISITOS previos:
  - /workspace/preds_ensemble_oficial : inferencia ensemble sobre los 91 (con --save_probabilities)
  - /workspace/preds_soft_oficial     : probabilidades del soft sobre los 91
  - /workspace/imagesVal              : 4 modalidades por caso (_0000.._0003)
"""

import numpy as np, os, glob, nibabel as nib
from scipy import ndimage

ENSEMBLE = "/workspace/preds_ensemble_oficial"   # .npz con probabilities (5-fold)
SOFT     = "/workspace/preds_soft_oficial"        # .npz del soft
IMAGES   = "/workspace/imagesVal"
OUT      = "/workspace/preds_final_ensemble"
ZIP      = "/workspace/submission_final_ensemble.zip"

TC_SOFT_CH = 1
THR_SOFT   = 0.5          # <-- threshold del soft (cambiado a 0.5)
COEF_T1C, COEF_T2W, INTERCEPT = -0.5844, 1.0382, -2.2613
THR_CC, MIN_CC = 0.90, 100
MIN_COMPONENT = 50

os.makedirs(OUT, exist_ok=True)


def zscore_brain(vol):
    m = vol > 0
    if m.sum() == 0: return np.zeros_like(vol, dtype=np.float32)
    mu, s = vol[m].mean(), vol[m].std()
    z = np.zeros_like(vol, dtype=np.float32); z[m] = (vol[m] - mu) / (s + 1e-6)
    return z


def filtrar(mask, minv):
    if mask.sum() == 0: return mask
    lab, n = ndimage.label(mask)
    if n == 0: return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (sizes > minv)[lab]


def regiones_desde_probs(prob, ref_shape):
    """
    Del ensemble hard (region-based), reconstruye WT, TC, ET binarios.
    prob: (3, Z, Y, X) con regiones [WT, TC, ET] segun regions_class_order.
    Se binariza a 0.5 y se alinea la orientacion.
    """
    p = prob
    # alinear cada canal
    chans = []
    for i in range(p.shape[0]):
        c = p[i]
        if c.shape != ref_shape:
            c = np.transpose(c, (2, 1, 0))
        chans.append(c)
    wt = chans[0] > 0.5
    tc = chans[1] > 0.5
    et = chans[2] > 0.5
    # jerarquia: TC dentro de WT, ET dentro de TC
    tc = tc & wt
    et = et & tc
    return wt, tc, et


def main():
    ens_files = sorted(glob.glob(os.path.join(ENSEMBLE, "*.npz")))
    print(f"Casos ensemble: {len(ens_files)}")
    n = 0
    for ef in ens_files:
        cid = os.path.basename(ef).replace(".npz", "")
        ref_p = os.path.join(IMAGES, f"{cid}_0000.nii.gz")
        if not os.path.exists(ref_p):
            print(f"  [!] falta imagen ref para {cid}"); continue
        ref = nib.load(ref_p)
        ref_shape = ref.shape

        # --- ensemble hard: regiones ---
        prob = np.load(ef)["probabilities"]
        wt_hard, tc_hard, et_hard = regiones_desde_probs(prob, ref_shape)

        # --- soft TC ---
        soft_wt_add = np.zeros(ref_shape, dtype=bool)
        sp = os.path.join(SOFT, cid + ".npz")
        if os.path.exists(sp):
            tc_soft = np.load(sp)["probabilities"][TC_SOFT_CH]
            if tc_soft.shape != ref_shape:
                tc_soft = np.transpose(tc_soft, (2, 1, 0))
            soft_wt_add = tc_soft > THR_SOFT

        # --- WT combinado (via B) ---
        wt = wt_hard | soft_wt_add
        wt = filtrar(wt, MIN_COMPONENT)
        # TC y ET del ensemble hard, dentro del WT
        tc = tc_hard & wt
        et = et_hard & tc
        edema = wt & ~tc          # edema = WT - core (del hard)

        # --- CC logistica sobre el NET ---
        net = tc & ~et
        cc = np.zeros(ref_shape, dtype=bool)
        t1c_p = os.path.join(IMAGES, f"{cid}_0001.nii.gz")
        t2w_p = os.path.join(IMAGES, f"{cid}_0002.nii.gz")
        if os.path.exists(t1c_p) and os.path.exists(t2w_p):
            t1c_z = zscore_brain(np.asanyarray(nib.load(t1c_p).dataobj).astype(np.float32))
            t2w_z = zscore_brain(np.asanyarray(nib.load(t2w_p).dataobj).astype(np.float32))
            logit = COEF_T1C * t1c_z + COEF_T2W * t2w_z + INTERCEPT
            prob_cc = 1.0 / (1.0 + np.exp(-logit))
            cc = filtrar(net & (prob_cc > THR_CC), MIN_CC)

        # --- ensamblar etiquetas: ET=1, NET=2, CC=3, ED=4 ---
        out = np.zeros(ref_shape, dtype=np.uint8)
        out[edema] = 4
        out[net] = 2
        out[cc] = 3
        out[et] = 1

        nib.save(nib.Nifti1Image(out, ref.affine, ref.header),
                 os.path.join(OUT, cid + ".nii.gz"))
        n += 1
        if n % 20 == 0: print(f"  {n}...")

    print(f"\nConstruidos: {n}")
    # verificacion
    fs = sorted(glob.glob(os.path.join(OUT, "*.nii.gz")))
    vals = set()
    for f in fs: vals |= set(np.unique(np.asanyarray(nib.load(f).dataobj)).tolist())
    con_cc = sum(1 for f in fs if 3 in np.unique(np.asanyarray(nib.load(f).dataobj)))
    print(f"valores: {sorted(vals)} (debe ser 0,1,2,3,4)")
    print(f"casos con CC: {con_cc}")
    wts = [(np.asanyarray(nib.load(f).dataobj) > 0).sum() for f in fs[:10]]
    print(f"WT medio (10 casos): {np.mean(wts):.0f} voxeles")

    # --- zip plano ---
    import subprocess
    if os.path.exists(ZIP): os.remove(ZIP)
    subprocess.run(f"cd {OUT} && zip -j -q {ZIP} *.nii.gz", shell=True)
    print(f"\nZIP listo: {ZIP}")
    print(f"  {len(fs)} casos empaquetados")


if __name__ == "__main__":
    main()
