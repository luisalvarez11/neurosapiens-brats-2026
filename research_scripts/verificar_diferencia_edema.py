#!/usr/bin/env python3
"""
verificar_diferencia_edema.py
=============================

Prueba la hipotesis: la diferencia (WT_soft - WT_hard) coincide con el edema
real (label 4) del ground truth?

Sobre el split interno (fold 0, con GT), para cada caso:
  - WT_soft  = canal TC del soft > 0.5  (o el canal WT)
  - WT_hard  = prediccion hard >= 1
  - diff     = WT_soft AND NOT WT_hard   (lo que el soft añade)
  - edema_gt = GT == 4

Mide cuanto de 'diff' es realmente edema, y cuanto edema captura.
Si diff coincide con edema_gt -> la hipotesis tiene sentido.
Si diff es ruido de borde -> no es edema.
"""

import numpy as np, os, glob, json, nibabel as nib

PROB = "/workspace/preds_soft_interno"
HARD = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
GT   = "/workspace/nnUNet_data/nnUNet_raw/Dataset503_BraTSPED/labelsTr"
SPLITS = "/workspace/nnUNet_data/nnUNet_preprocessed/Dataset503_BraTSPED/splits_final.json"

TC_SOFT_CH = 1
THR_SOFT = 0.5


def main():
    val_ids = json.load(open(SPLITS))[0]['val']
    print(f"Casos: {len(val_ids)}\n")

    total_diff = 0
    total_diff_es_edema = 0      # de la diff, cuanto es edema real
    total_edema_gt = 0
    total_edema_capturado = 0    # del edema real, cuanto captura la diff
    casos_con_edema = 0
    casos_con_diff = 0

    print(f"{'caso':22}{'diff_vox':>10}{'edema_gt':>10}{'diff∩ed':>10}{'%diff_ed':>9}")
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
        if hard.shape != gt.shape:
            # el hard puede estar en otra orientacion
            if hard.T.shape == gt.shape:
                hard = hard.T

        wt_soft = tc_soft > THR_SOFT
        wt_hard = hard >= 1
        diff = wt_soft & ~wt_hard        # lo que el soft añade
        edema_gt = (gt == 4)

        d = diff.sum()
        e = edema_gt.sum()
        inter = (diff & edema_gt).sum()

        total_diff += d
        total_diff_es_edema += inter
        total_edema_gt += e
        total_edema_capturado += inter
        if e > 0: casos_con_edema += 1
        if d > 0: casos_con_diff += 1

        pct = 100 * inter / d if d > 0 else 0
        if d > 100 or e > 100:  # solo mostrar casos con algo
            print(f"{cid:22}{d:>10}{e:>10}{inter:>10}{pct:>8.1f}%")

    print("\n" + "=" * 60)
    print(f"Casos con edema en GT:  {casos_con_edema}")
    print(f"Casos con diferencia:   {casos_con_diff}")
    print(f"\nTotal voxeles diff (soft añade):  {total_diff:,}")
    print(f"De la diff, cuanto es edema real: {total_diff_es_edema:,} "
          f"({100*total_diff_es_edema/max(total_diff,1):.1f}%)")
    print(f"\nTotal edema en GT:                {total_edema_gt:,}")
    print(f"Del edema real, cuanto captura:   {total_edema_capturado:,} "
          f"({100*total_edema_capturado/max(total_edema_gt,1):.1f}%)")
    print("=" * 60)
    print("\nINTERPRETACION:")
    print("  Si '% diff que es edema' es ALTO (>50%) -> la diff SI es edema. Hipotesis correcta.")
    print("  Si es BAJO (<20%) -> la diff es ruido de borde, NO edema.")
    print("  Si '% edema capturado' es alto Y '% diff es edema' alto -> muy buena señal.")


if __name__ == "__main__":
    main()
