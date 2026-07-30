#!/usr/bin/env python3
"""
barrer_umbrales_cc.py
=====================

Prueba varias combinaciones de (umbral de probabilidad, MIN_CC_VOXELS) para la
separacion NET/CC, y mide el DSC a nivel de caso de AMBAS columnas (CC y NETC)
sobre el split de validacion interno (donde SI hay ground truth).
"""

import os, glob
import numpy as np
import nibabel as nib
from scipy import ndimage

# ---- RUTAS EN ASIMOV --------------------------------------------------------
PREDS_VAL = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
IMAGES    = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/imagesTr"
GT_DIR    = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/labelsTr"

# ---- Clasificador calibrado -------------------------------------------------
COEF_T1C  = -0.5844
COEF_T2W  =  1.0382
INTERCEPT = -2.2613

# ---- Rejilla de busqueda ----------------------------------------------------
THRESHOLDS   = [0.5, 0.7, 0.8, 0.9, 0.95]
MIN_VOXELS   = [15, 30, 50, 100]

def zscore_brain(vol):
    mask = vol > 0
    if mask.sum() == 0: return np.zeros_like(vol, dtype=np.float32)
    mu, sigma = vol[mask].mean(), vol[mask].std()
    z = np.zeros_like(vol, dtype=np.float32)
    z[mask] = (vol[mask] - mu) / (sigma + 1e-6)
    return z

def dice(pred_mask, gt_mask):
    ps, gs = pred_mask.sum(), gt_mask.sum()
    if gs == 0 and ps == 0: return np.nan
    if gs == 0 or ps == 0: return 0.0
    inter = np.logical_and(pred_mask, gt_mask).sum()
    return 2.0 * inter / (ps + gs)

def main():
    files = sorted(glob.glob(os.path.join(PREDS_VAL, "*.nii.gz")))
    if not files:
        raise SystemExit(f" [!] Error: No se encontraron predicciones en {PREDS_VAL}")
    print(f" -> Analizando {len(files)} casos del split de validación interno (modo low-RAM)...")

    # Matrices para almacenar resultados [caso, thr_idx, mv_idx]
    n_cases = len(files)
    n_thr = len(THRESHOLDS)
    n_mv = len(MIN_VOXELS)
    
    net_scores = np.zeros((n_cases, n_thr, n_mv))
    cc_scores  = np.zeros((n_cases, n_thr, n_mv))
    
    net_ref = []
    cc_ref  = []

    valid_cases = 0
    for idx, pf in enumerate(files, 1):
        cid = os.path.basename(pf).replace(".nii.gz", "")
        gt_p = os.path.join(GT_DIR, cid + ".nii.gz")
        t1c_p = os.path.join(IMAGES, f"{cid}_0001.nii.gz")
        t2w_p = os.path.join(IMAGES, f"{cid}_0002.nii.gz")
        
        if not (os.path.exists(gt_p) and os.path.exists(t1c_p) and os.path.exists(t2w_p)):
            continue
            
        valid_cases += 1
        seg = np.asarray(nib.load(pf).dataobj).astype(np.uint8)
        gt  = np.asarray(nib.load(gt_p).dataobj).astype(np.uint8)
        t1c = np.asarray(nib.load(t1c_p).dataobj).astype(np.float32)
        t2w = np.asarray(nib.load(t2w_p).dataobj).astype(np.float32)
        
        t1c_z, t2w_z = zscore_brain(t1c), zscore_brain(t2w)

        # Referencia (sin separar CC: todo el core crudo seg==2 pasa a NET oficial, 0 CC)
        net_ref.append(dice(seg == 2, gt == 2))
        cc_ref.append(dice(np.zeros_like(seg, bool), gt == 3))

        ambiguous = (seg == 2)
        logit = COEF_T1C * t1c_z + COEF_T2W * t2w_z + INTERCEPT
        prob_cc = 1.0 / (1.0 + np.exp(-logit))

        # Barrido de umbrales en caliente
        for t_idx, thr in enumerate(THRESHOLDS):
            cc_cand = ambiguous & (prob_cc > thr)
            labeled, n = ndimage.label(cc_cand)
            sizes = np.bincount(labeled.ravel()) if n > 0 else np.array([])
            
            for m_idx, mv in enumerate(MIN_VOXELS):
                if n > 0:
                    keep = sizes > mv
                    keep[0] = False
                    cc_final = keep[labeled]
                else:
                    cc_final = np.zeros_like(cc_cand)
                    
                pred_net = ambiguous & ~cc_final
                pred_cc  = ambiguous & cc_final
                
                net_scores[valid_cases-1, t_idx, m_idx] = dice(pred_net, gt == 2)
                cc_scores[valid_cases-1, t_idx, m_idx]  = dice(pred_cc, gt == 3)

        if idx % 10 == 0 or idx == n_cases:
            print(f"    [{idx}/{n_cases}] Casos evaluados...")

    if valid_cases == 0:
        raise SystemExit(" [!] No se pudo cargar ningún paciente completo (revisa las rutas de GT e IMAGES).")

    # --- Resultados ---
    mean_net_ref = np.nanmean(net_ref[:valid_cases])
    mean_cc_ref  = np.nanmean(cc_ref[:valid_cases])
    
    print("\n================= [ REFERENCIA (SIN SEPARAR CC) ] =================")
    print(f" -> NETC (Core): {mean_net_ref:.4f} | CC (Quiste): {mean_cc_ref:.4f}\n")

    print(f"{'thr':>5} {'minVox':>7} {'DSC_NETC':>10} {'DSC_CC':>8} {'balance':>9}")
    print("-" * 55)
    
    best = None
    best_row = (-1, -1)
    
    for t_idx, thr in enumerate(THRESHOLDS):
        for m_idx, mv in enumerate(MIN_VOXELS):
            m_net = np.nanmean(net_scores[:valid_cases, t_idx, m_idx])
            m_cc  = np.nanmean(cc_scores[:valid_cases, t_idx, m_idx])
            balance = (m_cc - mean_cc_ref) - (mean_net_ref - m_net)
            
            if best is None or balance > best[0]:
                best = (balance, thr, mv, m_net, m_cc)
                best_row = (t_idx, m_idx)

    for t_idx, thr in enumerate(THRESHOLDS):
        for m_idx, mv in enumerate(MIN_VOXELS):
            m_net = np.nanmean(net_scores[:valid_cases, t_idx, m_idx])
            m_cc  = np.nanmean(cc_scores[:valid_cases, t_idx, m_idx])
            balance = (m_cc - mean_cc_ref) - (mean_net_ref - m_net)
            
            flag = " <-- MEJOR" if (t_idx, m_idx) == best_row else ""
            print(f"{thr:>5.2f} {mv:>7d} {m_net:>10.4f} {m_cc:>8.4f} {balance:>+9.4f}{flag}")

    print("-" * 55)
    b = best
    print(f"\nMEJOR CONFIGURACIÓN: thr={b[1]}, minVox={b[2]}")
    print(f" -> NETC={b[3]:.4f}  CC={b[4]:.4f}  Balance={b[0]:+.4f}")
    print(f"\n[!] REGLA DE ORO: La referencia de NETC es {mean_net_ref:.4f}.")
    print("    Si al separar el quiste (CC) el NETC cae radicalmente por debajo de ese número, no compensa en el leaderboard global.")

if __name__ == "__main__":
    main()
