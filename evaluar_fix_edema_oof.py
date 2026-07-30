#!/usr/bin/env python3
"""
evaluar_fix_edema_oof.py
==========================

Evalua sobre las predicciones OOF (5 folds de validation de nnUNet) el
impacto real del fix: ANTES (bug, tc = tc_hard & wt) vs DESPUES (fix,
tc = (tc_hard | soft_wt_add) & wt).

Usa exactamente la misma estructura y logica de carga/transposicion que
los scripts de analisis que ya days corristeis (TP/FP/FN globales,
agregados por voxel sobre todos los casos), para que las cifras sean
directamente comparables con el analisis original de 926,413 FP.
"""

import os, glob, numpy as np, nibabel as nib
from scipy import ndimage

# --- RUTAS LOCALES (reales, servidor) ---
DIR_HARD_BASE = '/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres'
DIR_SOFT = '/workspace/preds_soft_interno'
DIR_GT = '/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/labelsTr'
# Asumido por convencion nnUNet_raw (imagesTr hermana de labelsTr). Cambia si no aplica.
DIR_IMAGES = DIR_GT.replace('labelsTr', 'imagesTr')

TC_SOFT_CH = 1
THR_SOFT = 0.5
COEF_T1C, COEF_T2W, INTERCEPT = -0.5844, 1.0382, -2.2613
THR_CC, MIN_CC = 0.90, 100
MIN_COMPONENT = 50

REGIONES = ['ET', 'NET', 'CC', 'ED', 'TC', 'WT']


def filtrar(mask, minv):
    if mask.sum() == 0: return mask
    lab, n = ndimage.label(mask)
    if n == 0: return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (sizes > minv)[lab]


def zscore_brain(vol):
    m = vol > 0
    if m.sum() == 0: return np.zeros_like(vol, dtype=np.float32)
    mu, s = vol[m].mean(), vol[m].std()
    z = np.zeros_like(vol, dtype=np.float32); z[m] = (vol[m] - mu) / (s + 1e-6)
    return z


def alinear_prob(prob, ref_shape):
    # misma logica que vuestros scripts de analisis
    if prob[0].shape != ref_shape:
        if prob[0].shape == tuple(reversed(ref_shape)):
            prob = np.transpose(prob, (0, 3, 2, 1))
    return prob


def procesar_caso(cid, ef):
    gt_path = os.path.join(DIR_GT, f'{cid}.nii.gz')
    if not os.path.exists(gt_path):
        return None
    gt_data = np.asanyarray(nib.load(gt_path).dataobj)
    ref_shape = gt_data.shape

    prob_hard = np.load(ef)['probabilities']
    prob_hard = alinear_prob(prob_hard, ref_shape)

    wt_hard = prob_hard[0] > 0.5
    tc_hard = prob_hard[1] > 0.5
    et_hard = prob_hard[2] > 0.5 if prob_hard.shape[0] > 2 else np.zeros(ref_shape, dtype=bool)

    soft_wt_add = np.zeros(ref_shape, dtype=bool)
    sp = os.path.join(DIR_SOFT, f'{cid}.npz')
    if os.path.exists(sp):
        tc_soft = np.load(sp)['probabilities'][TC_SOFT_CH]
        if tc_soft.shape != ref_shape:
            tc_soft = np.transpose(tc_soft, (2, 1, 0))
        soft_wt_add = tc_soft > THR_SOFT

    wt = filtrar(wt_hard | soft_wt_add, MIN_COMPONENT)

    prob_cc = None
    t1c_p = os.path.join(DIR_IMAGES, f'{cid}_0001.nii.gz')
    t2w_p = os.path.join(DIR_IMAGES, f'{cid}_0002.nii.gz')
    if os.path.exists(t1c_p) and os.path.exists(t2w_p):
        t1c_z = zscore_brain(np.asanyarray(nib.load(t1c_p).dataobj).astype(np.float32))
        t2w_z = zscore_brain(np.asanyarray(nib.load(t2w_p).dataobj).astype(np.float32))
        if t1c_z.shape != ref_shape:
            t1c_z = np.transpose(t1c_z, (2, 1, 0))
        if t2w_z.shape != ref_shape:
            t2w_z = np.transpose(t2w_z, (2, 1, 0))
        logit = COEF_T1C * t1c_z + COEF_T2W * t2w_z + INTERCEPT
        prob_cc = 1.0 / (1.0 + np.exp(-logit))

    def rama(tc):
        et = et_hard & tc
        net = tc & ~et
        cc = np.zeros(ref_shape, dtype=bool)
        if prob_cc is not None:
            cc = filtrar(net & (prob_cc > THR_CC), MIN_CC)
        net = net & ~cc
        edema = wt & ~tc
        return {'ET': et, 'NET': net, 'CC': cc, 'ED': edema, 'TC': tc, 'WT': wt}

    antes = rama(tc_hard & wt)
    despues = rama((tc_hard | soft_wt_add) & wt)

    gt_masks = {
        'ET': gt_data == 1, 'NET': gt_data == 2, 'CC': gt_data == 3, 'ED': gt_data == 4,
        'TC': (gt_data == 1) | (gt_data == 2) | (gt_data == 3), 'WT': gt_data > 0,
    }
    return antes, despues, gt_masks, gt_data


def nuevo_acumulador():
    return {r: {'tp': 0, 'fp': 0, 'fn': 0} for r in REGIONES}


def main():
    acumulado = {'antes': nuevo_acumulador(), 'despues': nuevo_acumulador()}
    # distribucion de FP de edema en el GT, antes/despues (para comparar con el analisis original)
    fp_dist = {'antes': {0: 0, 1: 0, 2: 0, 3: 0}, 'despues': {0: 0, 1: 0, 2: 0, 3: 0}}
    total_fp_edema = {'antes': 0, 'despues': 0}
    casos_procesados = 0

    for f in range(5):
        val_dir = f'{DIR_HARD_BASE}/fold_{f}/validation'
        ens_files = sorted(glob.glob(os.path.join(val_dir, '*.npz')))
        for ef in ens_files:
            cid = os.path.basename(ef).replace('.npz', '')
            res = procesar_caso(cid, ef)
            if res is None:
                continue
            antes, despues, gt_masks, gt_data = res

            for rama_nombre, pred in (('antes', antes), ('despues', despues)):
                for r in REGIONES:
                    p, g = pred[r], gt_masks[r]
                    acumulado[rama_nombre][r]['tp'] += (p & g).sum()
                    acumulado[rama_nombre][r]['fp'] += (p & ~g).sum()
                    acumulado[rama_nombre][r]['fn'] += (~p & g).sum()

                fp_mask = pred['ED'] & (gt_data != 4)
                voxeles_reales = gt_data[fp_mask]
                total_fp_edema[rama_nombre] += len(voxeles_reales)
                for c in [0, 1, 2, 3]:
                    fp_dist[rama_nombre][c] += (voxeles_reales == c).sum()

            casos_procesados += 1
            if casos_procesados % 20 == 0:
                print(f'  procesados {casos_procesados}...')

    print(f'\n=== RESULTADOS ({casos_procesados} casos, OOF 5 folds) ===\n')

    print(f"{'Region':8s} {'Dice antes':>11s} {'Dice despues':>13s} {'Delta':>8s}   {'Prec antes':>10s} {'Prec despues':>12s}   {'Rec antes':>10s} {'Rec despues':>11s}")
    for r in REGIONES:
        a = acumulado['antes'][r]
        d = acumulado['despues'][r]
        dice_a = 2 * a['tp'] / (2 * a['tp'] + a['fp'] + a['fn']) if (a['tp'] + a['fp'] + a['fn']) > 0 else float('nan')
        dice_d = 2 * d['tp'] / (2 * d['tp'] + d['fp'] + d['fn']) if (d['tp'] + d['fp'] + d['fn']) > 0 else float('nan')
        prec_a = a['tp'] / (a['tp'] + a['fp']) if (a['tp'] + a['fp']) > 0 else float('nan')
        prec_d = d['tp'] / (d['tp'] + d['fp']) if (d['tp'] + d['fp']) > 0 else float('nan')
        rec_a = a['tp'] / (a['tp'] + a['fn']) if (a['tp'] + a['fn']) > 0 else float('nan')
        rec_d = d['tp'] / (d['tp'] + d['fn']) if (d['tp'] + d['fn']) > 0 else float('nan')
        print(f"{r:8s} {dice_a:11.4f} {dice_d:13.4f} {dice_d - dice_a:+8.4f}   {prec_a:10.4f} {prec_d:12.4f}   {rec_a:10.4f} {rec_d:11.4f}")

    print('\n(WT deberia salir con Delta ~0.0000: el fix solo reetiqueta dentro del WT.)')

    for rama_nombre in ('antes', 'despues'):
        tot = total_fp_edema[rama_nombre]
        print(f"\n=== Distribucion de FP de Edema en GT — {rama_nombre.upper()} ({tot:,} vóxeles) ===")
        if tot > 0:
            print(f"  Fondo Sano (0): {fp_dist[rama_nombre][0]:,} ({100*fp_dist[rama_nombre][0]/tot:.1f}%)")
            print(f"  NETC (2)      : {fp_dist[rama_nombre][2]:,} ({100*fp_dist[rama_nombre][2]/tot:.1f}%)")
            print(f"  ET (1)        : {fp_dist[rama_nombre][1]:,} ({100*fp_dist[rama_nombre][1]/tot:.1f}%)")
            print(f"  Quiste CC (3) : {fp_dist[rama_nombre][3]:,} ({100*fp_dist[rama_nombre][3]/tot:.1f}%)")


if __name__ == '__main__':
    main()
