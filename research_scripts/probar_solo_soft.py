#!/usr/bin/env python3
"""
probar_solo_soft.py
==================

Compara tres estrategias de WT sobre el split interno (con GT), separando
los casos CON edema de los SIN edema, para ver si "solo soft" funciona o
si falla donde hay edema.

Estrategias:
  A) solo_soft : canal 1 del soft > thr        (el TC soft como WT)
  B) solo_hard : prediccion hard >= 1
  C) combinado : (soft>thr) OR (hard>=1)

Barre umbrales del soft (0.3, 0.4, 0.5).
Reporta Dice y NSD del WT, SEPARADO por casos con/sin edema.
Si 'solo_soft' cae en los casos CON edema -> confirma que le falta el edema.
"""

import numpy as np, os, glob, json, nibabel as nib
from scipy import ndimage

PROB = "/workspace/preds_soft_interno"
HARD = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
GT   = "/workspace/nnUNet_data/nnUNet_raw/Dataset503_BraTSPED/labelsTr"
SPLITS = "/workspace/nnUNet_data/nnUNet_preprocessed/Dataset503_BraTSPED/splits_final.json"
TC_SOFT_CH = 1


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

    # cargar todo una vez
    cache = []
    for cid in val_ids:
        pf = os.path.join(PROB, cid + '.npz')
        hf = os.path.join(HARD, cid + '.nii.gz')
        gf = os.path.join(GT, cid + '.nii.gz')
        if not (os.path.exists(pf) and os.path.exists(hf) and os.path.exists(gf)):
            continue
        tc_soft = np.load(pf)['probabilities'][TC_SOFT_CH]
        gt = np.asanyarray(nib.load(gf).dataobj).astype(np.uint8)
        if tc_soft.shape != gt.shape:
            tc_soft = np.transpose(tc_soft, (2, 1, 0))
        hard = np.asanyarray(nib.load(hf).dataobj).astype(np.uint8)
        if hard.shape != gt.shape and hard.T.shape == gt.shape:
            hard = hard.T
        tiene_edema = (gt == 4).sum() > 0
        cache.append((cid, tc_soft, hard >= 1, gt >= 1, tiene_edema))

    con_ed = [c for c in cache if c[4]]
    sin_ed = [c for c in cache if not c[4]]
    print(f"Casos: {len(cache)} total | {len(con_ed)} con edema | {len(sin_ed)} sin edema\n")

    for thr in [0.3, 0.4, 0.5]:
        print(f"=== UMBRAL SOFT = {thr} ===")
        print(f"{'estrategia':12}{'grupo':12}{'WT_dice':>9}{'WT_nsd':>9}")
        for nombre, grupos in [("TODOS", cache), ("CON edema", con_ed), ("SIN edema", sin_ed)]:
            for estrat in ['solo_soft', 'solo_hard', 'combinado']:
                dd, nn = [], []
                for cid, ts, wh, wg, _ in grupos:
                    ss = ts > thr
                    if estrat == 'solo_soft':
                        pred = ss
                    elif estrat == 'solo_hard':
                        pred = wh
                    else:
                        pred = ss | wh
                    dd.append(dice(pred, wg))
                    nn.append(nsd(pred, wg))
                print(f"{estrat:12}{nombre:12}{np.nanmean(dd):>9.4f}{np.nanmean(nn):>9.4f}")
            print()
        print()

    print("CLAVE: mira 'solo_soft' en el grupo 'CON edema'.")
    print("  Si su Dice/NSD cae bastante vs 'combinado' -> le falta el edema.")
    print("  Si es igual -> solo_soft basta (poco probable).")


if __name__ == "__main__":
    main()
