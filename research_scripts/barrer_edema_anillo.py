#!/usr/bin/env python3
"""
barrer_edema_anillo.py (Versión Hyper-Low-RAM < 50MB)
=====================================================

Recuperación de Edema (ED, label 4) en el anillo peritumoral exterior a WT.
Extracción por coordenadas: jamás apila volúmenes 4D ni satura la RAM del servidor.
"""

import os, glob, gc
import numpy as np
import nibabel as nib
from scipy import ndimage

# ---- RUTAS EN ASIMOV --------------------------------------------------------
PREDS_VAL = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation"
IMAGES    = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/imagesTr"
GT_DIR    = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/labelsTr"
CHANNELS  = ["0000", "0001", "0002", "0003"]   # t1n, t1c, t2w, t2f

# ---- Clasificador de edema (comparación B, 4 canales) -----------------------
COEF = np.array([-0.0024, -0.7011, 0.4016, 1.8768], dtype=np.float32)
INTERCEPT = -3.2951

# ---- Rejilla de búsqueda ----------------------------------------------------
DILATACIONES = [2, 3, 5, 8]
UMBRALES     = [0.70, 0.85, 0.95]
MIN_VOXELS   = [30, 100]

def dice(pred, gt):
    ps, gs = pred.sum(), gt.sum()
    if gs == 0 and ps == 0: return np.nan
    if gs == 0 or ps == 0: return 0.0
    return 2.0 * np.logical_and(pred, gt).sum() / (ps + gs)

def main():
    files = sorted(glob.glob(os.path.join(PREDS_VAL, "*.nii.gz")))
    if not files:
        raise SystemExit(f" [!] No hay predicciones en {PREDS_VAL}")

    n_cases = len(files)
    n_dil = len(DILATACIONES)
    n_thr = len(UMBRALES)
    n_mv  = len(MIN_VOXELS)

    print(f"-> Evaluando rejilla en {n_cases} pacientes (Modo Coordenadas < 50MB RAM)...\n")

    wt_scores  = np.zeros((n_cases, n_dil, n_thr, n_mv), dtype=np.float32)
    net_scores = np.zeros((n_cases, n_dil, n_thr, n_mv), dtype=np.float32)
    ed_scores  = np.zeros((n_cases, n_dil, n_thr, n_mv), dtype=np.float32)

    wt_ref, net_ref, ed_ref = [], [], []
    valid_cases = 0

    for idx, pf in enumerate(files, 1):
        cid = os.path.basename(pf).replace(".nii.gz", "")
        gp = os.path.join(GT_DIR, cid + ".nii.gz")
        chp = [os.path.join(IMAGES, f"{cid}_{c}.nii.gz") for c in CHANNELS]
        
        if not (os.path.exists(gp) and all(os.path.exists(p) for p in chp)):
            continue
            
        valid_cases += 1
        seg = np.asanyarray(nib.load(pf).dataobj).astype(np.uint8)
        gt  = np.asanyarray(nib.load(gp).dataobj).astype(np.uint8)

        # 1. Referencia base (sin añadir edema)
        pred_wt_base  = (seg >= 1)
        pred_net_base = (seg == 2) | (seg == 1)
        
        wt_ref.append(dice(pred_wt_base, gt >= 1))
        net_ref.append(dice(pred_net_base, gt == 2))
        ed_ref.append(dice(np.zeros_like(seg, dtype=bool), gt == 4))

        # 2. Obtener máscara cerebral ligera desde T1n
        t1n_vol = np.asanyarray(nib.load(chp[0]).dataobj).astype(np.float32)
        brain_mask = t1n_vol > 0
        del t1n_vol

        # 3. Encontrar índices del anillo máximo (dilatación 8) para no apilar 4D
        max_dil_mask = ndimage.binary_dilation(pred_wt_base, iterations=max(DILATACIONES))
        max_anillo   = max_dil_mask & ~pred_wt_base & brain_mask
        del max_dil_mask, brain_mask

        idx_anillo = np.where(max_anillo.ravel())[0]
        del max_anillo

        if len(idx_anillo) == 0:
            for d_i in range(n_dil):
                for t_i in range(n_thr):
                    for m_i in range(n_mv):
                        wt_scores[valid_cases-1, d_i, t_i, m_i]  = wt_ref[-1]
                        net_scores[valid_cases-1, d_i, t_i, m_i] = net_ref[-1]
                        ed_scores[valid_cases-1, d_i, t_i, m_i]  = ed_ref[-1]
            del seg, gt, pred_wt_base, pred_net_base, idx_anillo
            gc.collect()
            print(f"   [{idx}/{n_cases}] Caso {cid} sin anillo (ok)...")
            continue

        # 4. Calcular logística canal por canal (RAM ínfima)
        logit_anillo = np.full(len(idx_anillo), INTERCEPT, dtype=np.float32)
        for ch_idx, p_ch in enumerate(chp):
            vol = np.asanyarray(nib.load(p_ch).dataobj).astype(np.float32)
            mask_pos = vol > 0
            if mask_pos.sum() > 0:
                mu, sig = vol[mask_pos].mean(), vol[mask_pos].std()
                vals = vol.ravel()[idx_anillo]
                vals_z = (vals - mu) / (sig + 1e-6)
                logit_anillo += COEF[ch_idx] * vals_z
                del vals, vals_z
            del vol, mask_pos
            gc.collect()

        prob_anillo = 1.0 / (1.0 + np.exp(-logit_anillo))
        del logit_anillo

        # 5. Barrido de la rejilla reutilizando un único buffer 3D
        cand_3d = np.zeros_like(seg, dtype=bool)

        for d_idx, dil in enumerate(DILATACIONES):
            dil_mask = ndimage.binary_dilation(pred_wt_base, iterations=dil)
            en_dil = dil_mask.ravel()[idx_anillo]
            del dil_mask

            for t_idx, thr in enumerate(UMBRALES):
                mask_valid = en_dil & (prob_anillo > thr)
                if mask_valid.sum() == 0:
                    for m_idx in range(n_mv):
                        wt_scores[valid_cases-1, d_idx, t_idx, m_idx]  = wt_ref[-1]
                        net_scores[valid_cases-1, d_idx, t_idx, m_idx] = net_ref[-1]
                        ed_scores[valid_cases-1, d_idx, t_idx, m_idx]  = ed_ref[-1]
                    continue

                cand_3d.ravel()[idx_anillo[mask_valid]] = True
                lab, n = ndimage.label(cand_3d)
                sizes = np.bincount(lab.ravel()) if n > 0 else np.empty(0, dtype=int)

                for m_idx, mv in enumerate(MIN_VOXELS):
                    if n > 0:
                        keep = sizes > mv
                        keep[0] = False
                        out_ed = keep[lab]
                    else:
                        out_ed = np.zeros_like(seg, dtype=bool)

                    pred_wt_final = pred_wt_base | out_ed
                    wt_scores[valid_cases-1, d_idx, t_idx, m_idx]  = dice(pred_wt_final, gt >= 1)
                    net_scores[valid_cases-1, d_idx, t_idx, m_idx] = net_ref[-1]
                    ed_scores[valid_cases-1, d_idx, t_idx, m_idx]  = dice(out_ed, gt == 4)
                    del out_ed, pred_wt_final

                del lab, sizes
                cand_3d.fill(False)

        # Limpieza total del paciente
        del seg, gt, pred_wt_base, pred_net_base, idx_anillo, prob_anillo, cand_3d
        gc.collect()
        print(f"   [{idx}/{n_cases}] Caso {cid} procesado (RAM limpia)...")

    if valid_cases == 0:
        raise SystemExit(" [!] No se procesó ningún caso válido. Revisa las rutas.")

    # --- Resultados ---
    WT_REF  = np.nanmean(wt_ref[:valid_cases])
    NET_REF = np.nanmean(net_ref[:valid_cases])
    ED_REF  = np.nanmean(ed_ref[:valid_cases])

    print("\n=================== [ REFERENCIA (SIN EDEMA) ] ===================")
    print(f" -> WT={WT_REF:.4f} | NETC={NET_REF:.4f} | ED={ED_REF:.4f}\n")

    print(f"{'dil':>4}{'thr':>6}{'minVox':>8}{'WT':>10}{'NETC':>10}{'ED':>10}{'dWT':>9}")
    print("-" * 60)

    resultados = []
    for d_idx, dil in enumerate(DILATACIONES):
        for t_idx, thr in enumerate(UMBRALES):
            for m_idx, mv in enumerate(MIN_VOXELS):
                m_wt  = np.nanmean(wt_scores[:valid_cases, d_idx, t_idx, m_idx])
                m_net = np.nanmean(net_scores[:valid_cases, d_idx, t_idx, m_idx])
                m_ed  = np.nanmean(ed_scores[:valid_cases, d_idx, t_idx, m_idx])
                dwt   = m_wt - WT_REF
                
                resultados.append((dil, thr, mv, m_wt, m_net, m_ed, dwt))
                flag = "  <-- WT OK" if dwt > -0.003 and m_ed > 0.02 else ""
                print(f"{dil:>4}{thr:>6.2f}{mv:>8d}{m_wt:>10.4f}{m_net:>10.4f}{m_ed:>10.4f}{dwt:>+9.4f}{flag}")

    print("-" * 60)
    print(f"\nReferencia WT = {WT_REF:.4f}")
    print("Regla de oro: El WT no debe caer más de 0.003 respecto a la referencia.")

    validas = [r for r in resultados if r[6] > -0.003 and r[5] > 0.02]
    if validas:
        best = max(validas, key=lambda r: r[5])
        print(f"\nMEJOR CONFIGURACIÓN (WT protegido): dil={best[0]}, thr={best[1]}, minVox={best[2]}")
        print(f" -> WT={best[3]:.4f} | NETC={best[4]:.4f} | ED={best[5]:.4f} | dWT={best[6]:+.4f}")
    else:
        print("\n[!] Ninguna configuración recupera Edema sin degradar el Whole Tumor (WT).")
        print("    Decisión científica: El edema no es recuperable sin canibalizar el contorno.")

if __name__ == "__main__":
    main()
