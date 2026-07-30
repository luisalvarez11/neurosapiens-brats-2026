#!/usr/bin/env python3
"""
evaluar_AvsB_y_final.py
=======================

Sobre los 59 casos del val del fold 0 (out-of-fold para hard-fold0 y soft-fold0):

1. Compara construcciones del WT:
   A) WT = hard >= 1
   B) WT = (hard >= 1) OR (soft_TC > thr)
2. Evalua la construccion FINAL completa (con las decisiones tomadas) contra GT.

Todo con Dice y NSD, contra el GT real.
"""

import numpy as np, os, glob, json, nibabel as nib
from scipy import ndimage

HARD = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
SOFT = "/workspace/preds_soft_interno"
GT   = "/workspace/nnUNet_data/nnUNet_raw/Dataset503_BraTSPED/labelsTr"
IMAGES = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/imagesTr"
SPLITS = "/workspace/nnUNet_data/nnUNet_preprocessed/Dataset501_BraTSPED/splits_final.json"

TC_SOFT_CH = 1
THR_SOFT = 0.5
COEF_T1C, COEF_T2W, INTERCEPT = -0.5844, 1.0382, -2.2613
THR_CC, MIN_CC = 0.90, 100


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

def zscore_brain(vol):
    m = vol > 0
    if m.sum() == 0: return np.zeros_like(vol, dtype=np.float32)
    mu, s = vol[m].mean(), vol[m].std()
    z = np.zeros_like(vol, dtype=np.float32); z[m] = (vol[m] - mu) / (s + 1e-6)
    return z


def main():
    val_ids = json.load(open(SPLITS))[0]['val']
    ids = [c for c in val_ids
           if os.path.exists(f"{HARD}/{c}.nii.gz")
           and os.path.exists(f"{SOFT}/{c}.npz")
           and os.path.exists(f"{GT}/{c}.nii.gz")]
    print(f"Casos evaluables: {len(ids)}\n")

    # ---------- PRUEBA 1: A vs B para el WT ----------
    print("=== WT: A (solo hard) vs B (hard OR soft) ===")
    A_d, A_n, B_d, B_n = [], [], [], []
    for cid in ids:
        hard = np.asanyarray(nib.load(f"{HARD}/{cid}.nii.gz").dataobj)
        gt = np.asanyarray(nib.load(f"{GT}/{cid}.nii.gz").dataobj)
        if hard.shape != gt.shape and hard.T.shape == gt.shape: hard = hard.T
        soft_tc = np.load(f"{SOFT}/{cid}.npz")['probabilities'][TC_SOFT_CH]
        if soft_tc.shape != gt.shape: soft_tc = np.transpose(soft_tc, (2,1,0))
        wt_gt = gt >= 1
        wt_A = hard >= 1
        wt_B = (hard >= 1) | (soft_tc > THR_SOFT)
        A_d.append(dice(wt_A, wt_gt)); A_n.append(nsd(wt_A, wt_gt))
        B_d.append(dice(wt_B, wt_gt)); B_n.append(nsd(wt_B, wt_gt))
    print(f"  A (solo hard):      dice {np.nanmean(A_d):.4f}  nsd {np.nanmean(A_n):.4f}")
    print(f"  B (hard OR soft):   dice {np.nanmean(B_d):.4f}  nsd {np.nanmean(B_n):.4f}")
    print(f"  -> B {'mejora' if np.nanmean(B_n)>np.nanmean(A_n) else 'NO mejora'} el NSD del WT\n")

    # ---------- PRUEBA 2: construccion FINAL completa ----------
    # Decision: WT = A o B (elige la mejor de arriba); TC = hard; ET = hard; CC = logistica
    usar_B = np.nanmean(B_n) > np.nanmean(A_n)
    print(f"=== CONSTRUCCION FINAL (WT = {'B (hard OR soft)' if usar_B else 'A (solo hard)'}) ===")
    res = {r: ([], []) for r in ['WT', 'TC', 'ET']}
    for cid in ids:
        hard = np.asanyarray(nib.load(f"{HARD}/{cid}.nii.gz").dataobj).astype(np.uint8)
        gt = np.asanyarray(nib.load(f"{GT}/{cid}.nii.gz").dataobj)
        if hard.shape != gt.shape and hard.T.shape == gt.shape: hard = hard.T
        soft_tc = np.load(f"{SOFT}/{cid}.npz")['probabilities'][TC_SOFT_CH]
        if soft_tc.shape != gt.shape: soft_tc = np.transpose(soft_tc, (2,1,0))

        # regiones del hard crudo (valores 1,2,3 region-based)
        wt_hard = hard >= 1
        tc_hard = np.isin(hard, [1,2,3])
        et_hard = (hard == 3)

        wt = (wt_hard | (soft_tc > THR_SOFT)) if usar_B else wt_hard
        tc = tc_hard
        et = et_hard

        # evaluar contra GT
        res['WT'][0].append(dice(wt, gt>=1));            res['WT'][1].append(nsd(wt, gt>=1))
        res['TC'][0].append(dice(tc, np.isin(gt,[1,2,3]))); res['TC'][1].append(nsd(tc, np.isin(gt,[1,2,3])))
        res['ET'][0].append(dice(et, gt==1));            res['ET'][1].append(nsd(et, gt==1))
    for r in ['WT','TC','ET']:
        print(f"  {r}: dice {np.nanmean(res[r][0]):.4f}  nsd {np.nanmean(res[r][1]):.4f}")

    print("\nComparar con la referencia (fold 0 single, 9773553 oficial):")
    print("  WT nsd 0.646 | TC nsd 0.646 (oficial). Interno sera algo mas alto.")


if __name__ == "__main__":
    main()
