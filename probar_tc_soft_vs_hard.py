#!/usr/bin/env python3
"""
probar_tc_soft_vs_hard.py
=========================

Compara el TC del modelo soft vs el TC del modelo hard, ambos contra el
TC REAL del ground truth (labels 1,2,3), sobre el split interno.

Esto responde: ¿el TC del soft es mejor que el TC del hard COMO TC?
(la comparacion anterior era canal soft contra WT, que no es lo mismo)

TC del GT = labels {1,2,3} (ET + NET + CC, todo el core sin edema).
"""

import numpy as np, os, glob, json, nibabel as nib
from scipy import ndimage

PROB = "/workspace/preds_soft_interno"
HARD = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
GT   = "/workspace/nnUNet_data/nnUNet_raw/Dataset503_BraTSPED/labelsTr"
SPLITS = "/workspace/nnUNet_data/nnUNet_preprocessed/Dataset503_BraTSPED/splits_final.json"
TC_SOFT_CH = 1   # canal TC del soft


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
    dg = ndimage.distance_transform_edt(~sg)
    dp = ndimage.distance_transform_edt(~sp)
    return ((dg[sp] <= tol).sum() + (dp[sg] <= tol).sum()) / (sp.sum() + sg.sum())


def main():
    val_ids = json.load(open(SPLITS))[0]['val']
    print(f"Comparando TC soft vs TC hard contra el TC real del GT\n")

    print(f"{'umbral':>7}{'TC_soft_dice':>14}{'TC_soft_nsd':>13}{'TC_hard_dice':>14}{'TC_hard_nsd':>13}")
    for thr in [0.3, 0.4, 0.5]:
        s_d, s_n, h_d, h_n = [], [], [], []
        for cid in val_ids:
            pf = os.path.join(PROB, cid + '.npz')
            hf = os.path.join(HARD, cid + '.nii.gz')
            gf = os.path.join(GT, cid + '.nii.gz')
            if not (os.path.exists(pf) and os.path.exists(hf) and os.path.exists(gf)):
                continue
            gt = np.asanyarray(nib.load(gf).dataobj).astype(np.uint8)
            tc_gt = np.isin(gt, [1, 2, 3])   # TC real

            # TC soft
            tc_soft = np.load(pf)['probabilities'][TC_SOFT_CH]
            if tc_soft.shape != gt.shape:
                tc_soft = np.transpose(tc_soft, (2, 1, 0))
            tc_soft_mask = tc_soft > thr

            # TC hard: en la prediccion hard, el TC son las labels de core.
            # OJO: el hard esta remapeado? La carpeta validation tiene la salida
            # cruda region-based. TC hard = pred que corresponde a core.
            # Cargamos y asumimos que el core es todo menos el edema/fondo.
            hard = np.asanyarray(nib.load(hf).dataobj).astype(np.uint8)
            if hard.shape != gt.shape and hard.T.shape == gt.shape:
                hard = hard.T
            # el hard region-based exporta valores; TC = core. Probamos varias defs:
            # lo mas robusto: TC hard = todo lo que el hard predice como tumor solido.
            # Si el hard esta remapeado (1,2,4): TC = (hard==1)|(hard==2) [ET+NET]
            # Si crudo region (1,2,3): TC segun esquema.
            # Aqui usamos: TC hard = hard>=1 menos lo que sea edema. Como el hard
            # crudo no separa edema, TC hard ~ hard>=1 (aprox, sobrestima si hay edema)
            # Para comparacion justa contra TC_gt, usamos hard core:
            tc_hard_mask = np.isin(hard, [1, 2, 3]) if hard.max() >= 3 else (hard >= 1)

            s_d.append(dice(tc_soft_mask, tc_gt)); s_n.append(nsd(tc_soft_mask, tc_gt))
            h_d.append(dice(tc_hard_mask, tc_gt)); h_n.append(nsd(tc_hard_mask, tc_gt))
        print(f"{thr:>7.2f}{np.nanmean(s_d):>14.4f}{np.nanmean(s_n):>13.4f}"
              f"{np.nanmean(h_d):>14.4f}{np.nanmean(h_n):>13.4f}")

    print("\nCLAVE: si TC_soft supera a TC_hard en Dice Y NSD -> usar TC soft tiene sentido.")
    print("Si TC_hard es igual o mejor -> quedarse con el hard (ya daba TC 0.94 en Synapse).")
    print("\nNOTA: verificar como se define el TC del hard (ver comentarios en el codigo).")
    # diagnostico: valores del hard
    hf0 = os.path.join(HARD, val_ids[0] + '.nii.gz')
    if os.path.exists(hf0):
        h0 = np.unique(np.asanyarray(nib.load(hf0).dataobj))
        print(f"Valores en prediccion hard (primer caso): {h0}")


if __name__ == "__main__":
    main()
