#!/usr/bin/env python3
"""
construir_submission_optima.py
==============================

Submission optima que combina lo mejor de cada modelo, segun lo medido:

  - WT  = (TC_soft > thr) OR (WT_hard)   -> mejor cobertura + borde (0.909)
  - TC  = TC_soft > thr                   -> mejor borde del core (0.915 vs 0.866)
  - ET  = del hard                        -> region pequena, el hard va bien
  - CC  = logistica thr 0.90 sobre el NET -> identico a 9773553
  - ED  = del hard (WT_hard - TC_hard)    -> edema real donde el hard lo ve
  - Filtro de componentes conexas         -> limpia motas (mejora HD95)

Reconstruccion de etiquetas anidadas (ET dentro de TC dentro de WT):
  Se construye jerarquicamente para garantizar consistencia.

MODO: primero validar en INTERNO (con GT) para confirmar mejora.
      Cambiar MODO='oficial' para generar la submission real.

Etiquetas oficiales: ET=1, NET=2, CC=3, ED=4.
"""

import numpy as np, os, glob, json, nibabel as nib
from scipy import ndimage

# ====== CONFIGURACION ======
MODO = "oficial"   # "interno" (con GT, para validar) o "oficial" (para submission)

TC_SOFT_CH = 1
THR_SOFT   = 0.3
COEF_T1C, COEF_T2W, INTERCEPT = -0.5844, 1.0382, -2.2613
THR_CC = 0.90
MIN_CC_VOXELS = 100
MIN_COMPONENT = 50     # filtro de motas por region

if MODO == "interno":
    PROB   = "/workspace/preds_soft_interno"
    HARD   = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
    IMAGES = "/workspace/nnUNet_data/nnUNet_raw/Dataset503_BraTSPED/imagesTr"
    GT_DIR = "/workspace/nnUNet_data/nnUNet_raw/Dataset503_BraTSPED/labelsTr"
    OUT    = "/workspace/preds_optima_interno"
    ids    = json.load(open("/workspace/nnUNet_data/nnUNet_preprocessed/Dataset503_BraTSPED/splits_final.json"))[0]['val']
else:
    PROB   = "/workspace/preds_soft_oficial"
    HARD   = "/workspace/predsVal_sub"      # hard remapeado (0/1/2/4)
    IMAGES = "/workspace/imagesVal"
    GT_DIR = None
    OUT    = "/workspace/preds_optima_oficial"
    ids    = [os.path.basename(f).replace(".npz","") for f in glob.glob(os.path.join("/workspace/preds_soft_oficial","*.npz"))]

os.makedirs(OUT, exist_ok=True)


def zscore_brain(vol):
    m = vol > 0
    if m.sum() == 0: return np.zeros_like(vol, dtype=np.float32)
    mu, s = vol[m].mean(), vol[m].std()
    z = np.zeros_like(vol, dtype=np.float32)
    z[m] = (vol[m] - mu) / (s + 1e-6)
    return z


def filtrar_componentes(mask, min_vox):
    if mask.sum() == 0: return mask
    lab, n = ndimage.label(mask)
    if n == 0: return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = sizes > min_vox
    return keep[lab]


def cargar_hard_regiones(hard, ref_shape):
    """Devuelve (wt_hard, tc_hard, et_hard) desde la prediccion hard."""
    if hard.shape != ref_shape and hard.T.shape == ref_shape:
        hard = hard.T
    vals = np.unique(hard)
    if hard.max() >= 3 and 4 not in vals:
        # crudo region-based (1,2,3): 1=WT-solo? depende. Asumimos remap necesario.
        # En la carpeta validation cruda: valores 1,2,3 = regiones exportadas.
        # WT = >=1, TC = (1,2,3) todo, ET segun. Simplificamos:
        wt = hard >= 1
        tc = np.isin(hard, [1, 2, 3])
        et = (hard == 3)   # en region-based crudo el ET suele ser el valor mayor
    else:
        # remapeado oficial (1,2,4): ET=1, NET=2, ED=4
        wt = hard >= 1
        tc = np.isin(hard, [1, 2, 3])   # ET+NET+CC
        et = (hard == 1)
    return wt, tc, et


def main():
    print(f"MODO: {MODO} | casos: {len(ids)}")
    n = 0
    for cid in ids:
        pf = os.path.join(PROB, cid + '.npz')
        hf = os.path.join(HARD, cid + '.nii.gz')
        if not (os.path.exists(pf) and os.path.exists(hf)):
            continue
        himg = nib.load(hf)
        hard = np.asanyarray(himg.dataobj).astype(np.uint8)
        ref_shape = hard.shape

        # --- soft ---
        prob = np.load(pf)['probabilities']
        tc_soft = prob[TC_SOFT_CH]
        if tc_soft.shape != ref_shape:
            tc_soft = np.transpose(tc_soft, (2, 1, 0))
        tc_soft_mask = tc_soft > THR_SOFT

        # --- hard regiones ---
        wt_hard, tc_hard, et_hard = cargar_hard_regiones(hard, ref_shape)

        # --- construir regiones finales ---
        wt = tc_soft_mask | wt_hard          # WT combinado
        tc = tc_soft_mask                     # TC del soft (mejor borde)
        # el TC debe estar dentro del WT (lo esta por construccion)
        et = et_hard & tc                     # ET del hard, dentro del TC
        # edema = WT - TC
        edema = wt & ~tc

        # filtrar motas por region
        wt = filtrar_componentes(wt, MIN_COMPONENT)
        tc = tc & wt
        et = filtrar_componentes(et, MIN_COMPONENT) & tc
        edema = wt & ~tc

        # --- CC sobre el NET (dentro de TC, no ET) ---
        net = tc & ~et
        t1c_p = os.path.join(IMAGES, f"{cid}_0001.nii.gz")
        t2w_p = os.path.join(IMAGES, f"{cid}_0002.nii.gz")
        cc = np.zeros(ref_shape, dtype=bool)
        if os.path.exists(t1c_p) and os.path.exists(t2w_p):
            t1c_z = zscore_brain(np.asanyarray(nib.load(t1c_p).dataobj).astype(np.float32))
            t2w_z = zscore_brain(np.asanyarray(nib.load(t2w_p).dataobj).astype(np.float32))
            logit = COEF_T1C * t1c_z + COEF_T2W * t2w_z + INTERCEPT
            prob_cc = 1.0 / (1.0 + np.exp(-logit))
            cc_cand = net & (prob_cc > THR_CC)
            cc = filtrar_componentes(cc_cand, MIN_CC_VOXELS)

        # --- ensamblar etiquetas (prioridad: ET > CC > NET > ED) ---
        out = np.zeros(ref_shape, dtype=np.uint8)
        out[edema] = 4
        out[net] = 2
        out[cc] = 3
        out[et] = 1

        nib.save(nib.Nifti1Image(out, himg.affine, himg.header),
                 os.path.join(OUT, cid + '.nii.gz'))
        n += 1
        if n % 20 == 0: print(f"  {n}...")

    print(f"\nConstruidos: {n} en {OUT}")

    # --- si es interno, evaluar contra GT ---
    if MODO == "interno" and GT_DIR:
        print("\n=== EVALUACION vs GT interno ===")
        evaluar(OUT, GT_DIR, ids)


def dice(a, b):
    sa, sb = a.sum(), b.sum()
    if sa == 0 and sb == 0: return np.nan
    if sa == 0 or sb == 0: return 0.0
    return 2 * np.logical_and(a, b).sum() / (sa + sb)


def surf(m): return m & ~ndimage.binary_erosion(m)


def nsd(pred, gt, tol=1):
    sa, sb = pred.sum(), gt.sum()
    if sa == 0 and sb == 0: return np.nan
    if sa == 0 or sb == 0: return 0.0
    sp, sg = surf(pred), surf(gt)
    if sp.sum() == 0 or sg.sum() == 0: return 0.0
    dg = ndimage.distance_transform_edt(~sg); dp = ndimage.distance_transform_edt(~sp)
    return ((dg[sp] <= tol).sum() + (dp[sg] <= tol).sum()) / (sp.sum() + sg.sum())


def evaluar(out_dir, gt_dir, ids):
    regs = {'WT (>=1)': lambda x: x>=1, 'TC (1,2,3)': lambda x: np.isin(x,[1,2,3]), 'ET (1)': lambda x: x==1}
    for rn, rf in regs.items():
        dd, nn = [], []
        for cid in ids:
            of = os.path.join(out_dir, cid+'.nii.gz'); gf = os.path.join(gt_dir, cid+'.nii.gz')
            if not (os.path.exists(of) and os.path.exists(gf)): continue
            o = np.asanyarray(nib.load(of).dataobj); g = np.asanyarray(nib.load(gf).dataobj)
            dd.append(dice(rf(o), rf(g))); nn.append(nsd(rf(o), rf(g)))
        print(f"  {rn:12} dice={np.nanmean(dd):.4f}  nsd={np.nanmean(nn):.4f}")
    print("\nComparar con 9773553: WT dice~0.89, TC~0.87, y el NSD del combinado ~0.788")


if __name__ == "__main__":
    main()
