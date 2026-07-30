#!/usr/bin/env python3
"""
construir_combinada_final.py
============================

Submission del WT COMBINADO, construida con control total (sin depender de
carpetas ambiguas).

Pipeline:
  1. Base: predsVal_sub (hard remapeado limpio, valores 0/1/2/4, SIN CC).
  2. WT combinado: se añade el TC_soft (canal 1 de probabilidades) como edema (4)
     donde extiende mas alla del WT hard. Mejora el borde (NSD) del WT.
  3. CC thr=0.90: se aplica la logistica de intensidad (identica a la 9773553)
     sobre el NET (label 2), separando el componente quistico (label 3).
  4. Zip plano.

Coeficientes CC (9773553): T1c=-0.5844, T2w=1.0382, intercept=-2.2613,
thr=0.90, MIN_CC_VOXELS=100.
"""

import numpy as np, os, glob, nibabel as nib
from scipy import ndimage

BASE_DIR = "/workspace/predsVal_sub"          # hard remapeado sin CC (0/1/2/4)
SOFT_DIR = "/workspace/preds_soft_oficial"    # probabilidades soft
IMAGES   = "/workspace/imagesVal"             # modalidades (para z-score del CC)
OUT_DIR  = "/workspace/preds_combinada_final2"

TC_SOFT_CH = 1
THR_SOFT   = 0.5
ADD_MIN_VOX = 50      # filtro para el edema añadido por el combinado

# CC (identico a 9773553)
COEF_T1C, COEF_T2W, INTERCEPT = -0.5844, 1.0382, -2.2613
THR_CC = 0.90
MIN_CC_VOXELS = 100

os.makedirs(OUT_DIR, exist_ok=True)


def zscore_brain(vol):
    m = vol > 0
    if m.sum() == 0: return np.zeros_like(vol, dtype=np.float32)
    mu, s = vol[m].mean(), vol[m].std()
    z = np.zeros_like(vol, dtype=np.float32)
    z[m] = (vol[m] - mu) / (s + 1e-6)
    return z


def main():
    files = sorted(glob.glob(os.path.join(BASE_DIR, "*.nii.gz")))
    print(f"Casos base: {len(files)}")
    n = 0
    for bp in files:
        cid = os.path.basename(bp).replace(".nii.gz", "")
        img = nib.load(bp)
        seg = np.asanyarray(img.dataobj).astype(np.uint8)   # 0/1/2/4 (ET=1,NET=2,ED=4)

        # ---------- 1. WT COMBINADO ----------
        sp = os.path.join(SOFT_DIR, cid + ".npz")
        if os.path.exists(sp):
            tc_soft = np.load(sp)["probabilities"][TC_SOFT_CH]
            if tc_soft.shape != seg.shape:
                tc_soft = np.transpose(tc_soft, (2, 1, 0))
            add = (tc_soft > THR_SOFT) & (seg == 0)   # extension nueva del WT
            # filtrar motas
            if add.sum() > 0:
                lab, nl = ndimage.label(add)
                if nl > 0:
                    sizes = np.bincount(lab.ravel()); sizes[0] = 0
                    keep = sizes > ADD_MIN_VOX
                    add = keep[lab]
            seg[add] = 4   # el tejido añadido = edema (WT fuera del core)

        # ---------- 2. CC thr=0.90 sobre el NET (label 2) ----------
        t1c_p = os.path.join(IMAGES, f"{cid}_0001.nii.gz")
        t2w_p = os.path.join(IMAGES, f"{cid}_0002.nii.gz")
        if os.path.exists(t1c_p) and os.path.exists(t2w_p):
            t1c_z = zscore_brain(np.asanyarray(nib.load(t1c_p).dataobj).astype(np.float32))
            t2w_z = zscore_brain(np.asanyarray(nib.load(t2w_p).dataobj).astype(np.float32))
            net = (seg == 2)
            logit = COEF_T1C * t1c_z + COEF_T2W * t2w_z + INTERCEPT
            prob_cc = 1.0 / (1.0 + np.exp(-logit))
            cand = net & (prob_cc > THR_CC)
            lab, nl = ndimage.label(cand)
            if nl > 0:
                sizes = np.bincount(lab.ravel()); sizes[0] = 0
                keep = sizes > MIN_CC_VOXELS
                cc_final = keep[lab]
                seg[cc_final] = 3   # CC

        nib.save(nib.Nifti1Image(seg, img.affine, img.header),
                 os.path.join(OUT_DIR, cid + ".nii.gz"))
        n += 1
        if n % 20 == 0:
            print(f"  {n} casos...")

    print(f"\nConstruidos: {n}")
    # verificacion
    f0 = sorted(glob.glob(os.path.join(OUT_DIR, "*.nii.gz")))[0]
    print(f"Ejemplo: valores {np.unique(np.asanyarray(nib.load(f0).dataobj))}  (debe tener 0/1/2/3/4)")
    con_cc = sum(1 for f in glob.glob(os.path.join(OUT_DIR, "*.nii.gz"))
                 if 3 in np.unique(np.asanyarray(nib.load(f).dataobj)))
    print(f"Casos con CC: {con_cc}")


if __name__ == "__main__":
    main()
