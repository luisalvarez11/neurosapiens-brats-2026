#!/usr/bin/env python3
"""
evaluar_fix_edema.py
=====================

Compara, caso a caso y contra ground truth, el pipeline de ensamblado
ANTES del fix (bug: soft_wt_add solo entra en WT) vs DESPUES del fix
(soft_wt_add tambien cuenta como TC), para cuantificar cuanto se gana
en Dice de ED, NET, TC, ET y CC.

No toca el pipeline real de submission (construir_submission_final_ensemble.py).
Es solo para medir el impacto del fix sobre vuestro set local de 294 casos
con ground truth conocido, antes de aplicarlo al pipeline oficial.

*** EDITA ESTAS RUTAS ANTES DE CORRER ***
"""

import numpy as np, os, glob, csv, nibabel as nib
from scipy import ndimage

# ---------------------------------------------------------------------
# RUTAS: deben apuntar al set LOCAL de 294 casos con GT, no a los 91
# oficiales (que no tienen ground truth).
# ---------------------------------------------------------------------
ENSEMBLE = "/workspace/preds_ensemble_local"   # .npz probabilities (5-fold), 294 casos
SOFT     = "/workspace/preds_soft_local"       # .npz probabilities del soft, 294 casos
IMAGES   = "/workspace/imagesTr_local"         # 4 modalidades por caso (_0000.._0003)
GT       = "/workspace/labelsTr_local"         # ground truth, mismo esquema: 1=ET 2=NET 3=CC 4=ED
CSV_OUT  = "/workspace/comparacion_fix_edema.csv"

TC_SOFT_CH = 1
THR_SOFT   = 0.5
COEF_T1C, COEF_T2W, INTERCEPT = -0.5844, 1.0382, -2.2613
THR_CC, MIN_CC = 0.90, 100
MIN_COMPONENT = 50

REGIONES = ["ET", "NET", "CC", "ED", "TC", "WT"]


# ------------------------- utilidades (identicas al script original) -------------------------

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
    chans = []
    for i in range(prob.shape[0]):
        c = prob[i]
        if c.shape != ref_shape:
            c = np.transpose(c, (2, 1, 0))
        chans.append(c)
    wt = chans[0] > 0.5
    tc = chans[1] > 0.5
    et = chans[2] > 0.5
    tc = tc & wt
    et = et & tc
    return wt, tc, et


def dice(pred, gt):
    p, g = pred.sum(), gt.sum()
    if p == 0 and g == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return 2.0 * inter / (p + g)


# ------------------------- pipeline (bifurcado antes/despues) -------------------------

def construir_caso(cid):
    ref_p = os.path.join(IMAGES, f"{cid}_0000.nii.gz")
    if not os.path.exists(ref_p):
        return None
    ref = nib.load(ref_p)
    ref_shape = ref.shape

    ef = os.path.join(ENSEMBLE, cid + ".npz")
    if not os.path.exists(ef):
        return None
    prob = np.load(ef)["probabilities"]
    wt_hard, tc_hard, et_hard = regiones_desde_probs(prob, ref_shape)

    soft_wt_add = np.zeros(ref_shape, dtype=bool)
    sp = os.path.join(SOFT, cid + ".npz")
    if os.path.exists(sp):
        tc_soft = np.load(sp)["probabilities"][TC_SOFT_CH]
        if tc_soft.shape != ref_shape:
            tc_soft = np.transpose(tc_soft, (2, 1, 0))
        soft_wt_add = tc_soft > THR_SOFT

    wt = filtrar(wt_hard | soft_wt_add, MIN_COMPONENT)

    # intensidades para CC (se calculan una sola vez, se usan en ambas ramas)
    t1c_p = os.path.join(IMAGES, f"{cid}_0001.nii.gz")
    t2w_p = os.path.join(IMAGES, f"{cid}_0002.nii.gz")
    prob_cc = None
    if os.path.exists(t1c_p) and os.path.exists(t2w_p):
        t1c_z = zscore_brain(np.asanyarray(nib.load(t1c_p).dataobj).astype(np.float32))
        t2w_z = zscore_brain(np.asanyarray(nib.load(t2w_p).dataobj).astype(np.float32))
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
        return {"ET": et, "NET": net, "CC": cc, "ED": edema, "TC": tc, "WT": wt}

    # ANTES (bug actual): TC solo del hard
    antes = rama(tc_hard & wt)
    # DESPUES (fix): el aporte del soft tambien cuenta como TC
    despues = rama((tc_hard | soft_wt_add) & wt)

    return antes, despues


def cargar_gt(cid, ref_shape):
    gp = os.path.join(GT, cid + ".nii.gz")
    if not os.path.exists(gp):
        return None
    g = np.asanyarray(nib.load(gp).dataobj).astype(np.uint8)
    if g.shape != ref_shape:
        g = np.transpose(g, (2, 1, 0))
    return {
        "ET": g == 1, "NET": g == 2, "CC": g == 3, "ED": g == 4,
        "TC": (g == 1) | (g == 2) | (g == 3), "WT": g > 0,
    }


def main():
    ens_files = sorted(glob.glob(os.path.join(ENSEMBLE, "*.npz")))
    print(f"Casos disponibles en ensemble local: {len(ens_files)}")

    filas = []
    acumulado = {r: {"antes": [], "despues": []} for r in REGIONES}
    n_ok = 0

    for i, ef in enumerate(ens_files, 1):
        cid = os.path.basename(ef).replace(".npz", "")
        res = construir_caso(cid)
        if res is None:
            continue
        antes, despues = res
        gt = cargar_gt(cid, antes["WT"].shape)
        if gt is None:
            continue

        n_ok += 1
        for r in REGIONES:
            d_antes = dice(antes[r], gt[r])
            d_despues = dice(despues[r], gt[r])
            acumulado[r]["antes"].append(d_antes)
            acumulado[r]["despues"].append(d_despues)
            filas.append([cid, r, f"{d_antes:.4f}", f"{d_despues:.4f}", f"{d_despues - d_antes:+.4f}"])

        if n_ok % 20 == 0:
            print(f"  procesados {n_ok} casos...")

    print(f"\nCasos evaluados (con GT + preds disponibles): {n_ok}")

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "region", "dice_antes", "dice_despues", "delta"])
        w.writerows(filas)
    print(f"CSV por caso guardado en: {CSV_OUT}")

    print("\n=== RESUMEN (Dice medio sobre los casos evaluados) ===")
    print(f"{'Region':8s} {'Antes':>8s} {'Despues':>8s} {'Delta':>8s}")
    for r in REGIONES:
        a = np.mean(acumulado[r]["antes"]) if acumulado[r]["antes"] else float("nan")
        d = np.mean(acumulado[r]["despues"]) if acumulado[r]["despues"] else float("nan")
        print(f"{r:8s} {a:8.4f} {d:8.4f} {d - a:+8.4f}")

    print("\n(WT deberia salir con delta ~0.0000: el fix solo reetiqueta")
    print(" vóxeles dentro del WT, no cambia su contorno.)")


if __name__ == "__main__":
    main()
