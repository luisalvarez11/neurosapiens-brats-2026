#!/usr/bin/env python3
"""
evaluar_soft_interno.py
=======================

Evalua el modelo soft sobre el split de validacion interno (fold 0, con GT).
Calcula Dice y NSD de las tres regiones, con barrido de umbrales para el WT.

Usa el GT preprocesado (_seg.b2nd), que esta en la MISMA orientacion que las
probabilidades, evitando el lio de ejes (155,240,240) vs (240,240,155).

Identifica automaticamente que canal es el WT comparando tamanos.
"""

import numpy as np, os, glob, json
import blosc2
from scipy import ndimage

PROB   = "/workspace/preds_soft_interno"
PREP   = "/workspace/nnUNet_data/nnUNet_preprocessed/Dataset503_BraTSPED/nnUNetPlans_3d_fullres"
SPLITS = "/workspace/nnUNet_data/nnUNet_preprocessed/Dataset503_BraTSPED/splits_final.json"

# Umbrales a barrer para el WT
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70]


def dice(a, b):
    sa, sb = a.sum(), b.sum()
    if sa == 0 and sb == 0: return np.nan
    if sa == 0 or sb == 0: return 0.0
    return 2.0 * np.logical_and(a, b).sum() / (sa + sb)


def surf(m):
    return m & ~ndimage.binary_erosion(m)


def nsd(pred, gt, tol=1.0):
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
    print(f"Casos de validacion interna: {len(val_ids)}")

    # --- identificar el canal WT en el primer caso ---
    f0 = os.path.join(PROB, val_ids[0] + '.npz')
    p0 = np.load(f0)['probabilities']
    print(f"\nShape probabilities: {p0.shape}")
    # WT = region (1,2,3,4), la mas grande. Comparar con el seg preprocesado.
    seg0 = np.asarray(blosc2.open(os.path.join(PREP, val_ids[0] + '_seg.b2nd')))[0]
    wt_gt_size = (seg0 >= 1).sum()
    print(f"WT real (primer caso): {wt_gt_size} voxeles")
    # buscar el canal cuyo tamano a 0.5 mas se acerca al WT real
    best_ch, best_diff = 0, 1e18
    for i in range(p0.shape[0]):
        sz = (p0[i] > 0.5).sum()
        print(f"  canal {i}: >0.5 = {sz} voxeles")
        # el WT es grande pero NO saturado; descartar canales saturados (>3x el WT real)
        if sz < wt_gt_size * 3 and abs(sz - wt_gt_size) < best_diff:
            best_diff = abs(sz - wt_gt_size)
            best_ch = i
    WT_CH = best_ch
    print(f"\n-> Canal WT identificado: {WT_CH}")
    # TC y ET: los otros dos canales (asumimos orden region: WT, TC, ET)
    # pero como el orden puede variar, evaluamos WT (lo que importa para soft)

    # --- barrido de umbrales para WT ---
    print("\n=== BARRIDO WT (Dice y NSD) ===")
    print(f"{'umbral':>7}{'WT_dice':>10}{'WT_nsd':>10}")
    results = {}
    for thr in THRESHOLDS:
        dd, nn = [], []
        for cid in val_ids:
            fp = os.path.join(PROB, cid + '.npz')
            sf = os.path.join(PREP, cid + '_seg.b2nd')
            if not (os.path.exists(fp) and os.path.exists(sf)):
                continue
            p = np.load(fp)['probabilities'][WT_CH]
            seg = np.asarray(blosc2.open(sf))[0]
            pred = p > thr
            gtwt = seg >= 1
            dd.append(dice(pred, gtwt))
            nn.append(nsd(pred, gtwt))
        md, mn = np.nanmean(dd), np.nanmean(nn)
        results[thr] = (md, mn)
        print(f"{thr:>7.2f}{md:>10.4f}{mn:>10.4f}")

    # mejor umbral por NSD
    best_thr = max(results, key=lambda t: results[t][1])
    print(f"\nMEJOR umbral por NSD: {best_thr} -> Dice={results[best_thr][0]:.4f}, NSD={results[best_thr][1]:.4f}")
    print("\nComparar este NSD con el del modelo DURO (fine-tune sin soft) en el mismo split.")
    print("Modelo duro daba: NSD WT ~0.759 (interno)")


if __name__ == "__main__":
    main()
