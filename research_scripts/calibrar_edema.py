#!/usr/bin/env python3
"""
calibrar_edema.py (Versión Low-RAM)
==================================

Evalúa la separabilidad del Edema Pediátrico (ED, label 4) mediante
la intensidad de FLAIR (z-score) con gestión estricta de memoria.
"""

import os, glob, gc
import numpy as np
import nibabel as nib
from scipy.stats import mannwhitneyu

# ---- RUTAS EN ASIMOV --------------------------------------------------------
IMAGES = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/imagesTr"
GT_DIR = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/labelsTr"
FLAIR_CH = "0003"  # FLAIR en el estándar BraTS

def zscore_brain(vol):
    mask = vol > 0
    if mask.sum() == 0:
        return np.zeros_like(vol, dtype=np.float32)
    mu, sigma = vol[mask].mean(), vol[mask].std()
    z = np.zeros_like(vol, dtype=np.float32)
    z[mask] = (vol[mask] - mu) / (sigma + 1e-6)
    return z

def main():
    gts = sorted(glob.glob(os.path.join(GT_DIR, "*.nii.gz")))
    print(f"-> Analizando {len(gts)} casos (modo ultra-eficiente RAM)...\n")

    ed_flair = []        # FLAIR z en edema real (gt==4)
    otro_flair = []      # FLAIR z en el resto del tumor (gt en 1,2,3)
    casos_con_ed = 0
    casos_sin_ed = 0

    for idx, gp in enumerate(gts, 1):
        cid = os.path.basename(gp).replace(".nii.gz", "")
        fp = os.path.join(IMAGES, f"{cid}_{FLAIR_CH}.nii.gz")
        if not os.path.exists(fp):
            continue
            
        gt_img = nib.load(gp)
        gt = np.asarray(gt_img.dataobj).astype(np.uint8)
        
        flair_img = nib.load(fp)
        flair_z = zscore_brain(np.asarray(flair_img.dataobj).astype(np.float32))

        ed_mask = (gt == 4)
        otro_mask = np.isin(gt, [1, 2, 3])

        if ed_mask.sum() > 0:
            casos_con_ed += 1
            vals_ed = flair_z[ed_mask]
            if vals_ed.size > 1000:
                vals_ed = np.random.choice(vals_ed, 1000, replace=False)
            ed_flair.append(vals_ed)
        else:
            casos_sin_ed += 1
            
        if otro_mask.sum() > 0:
            vals_otro = flair_z[otro_mask]
            if vals_otro.size > 1000:
                vals_otro = np.random.choice(vals_otro, 1000, replace=False)
            otro_flair.append(vals_otro)

        # Liberar memoria de objetos pesados explícitamente
        del gt, flair_z, ed_mask, otro_mask, gt_img, flair_img
        if idx % 10 == 0:
            gc.collect()

        if idx % 20 == 0 or idx == len(gts):
            print(f"   [{idx}/{len(gts)}] Volúmenes procesados de forma limpia...")

    ed_flair = np.concatenate(ed_flair) if ed_flair else np.array([])
    otro_flair = np.concatenate(otro_flair) if otro_flair else np.array([])

    print("\n" + "=" * 60)
    print(f"Casos CON edema (gt==4 presente): {casos_con_ed}")
    print(f"Casos SIN edema:                  {casos_sin_ed}")
    print("=" * 60)

    if ed_flair.size == 0:
        print("\n[!] NO hay vóxeles de edema en el ground truth de este dataset.")
        print("    El edema está 100% ausente -> Se confirma la invariancia de la clase.")
        return

    print(f"\nMuestra vóxeles edema:    {ed_flair.size:,}")
    print(f"Muestra vóxeles no-edema: {otro_flair.size:,}\n")
    
    print("FLAIR z-score:")
    print(f"  EDEMA     -> media={ed_flair.mean():.2f} | p25={np.percentile(ed_flair,25):.2f} "
          f"p50={np.median(ed_flair):.2f} p75={np.percentile(ed_flair,75):.2f}")
    print(f"  NO-EDEMA  -> media={otro_flair.mean():.2f} | p25={np.percentile(otro_flair,25):.2f} "
          f"p50={np.median(otro_flair):.2f} p75={np.percentile(otro_flair,75):.2f}")

    # AUC vía Mann-Whitney U
    u, _ = mannwhitneyu(ed_flair, otro_flair, alternative='two-sided')
    auc = u / (ed_flair.size * otro_flair.size)
    print("\n------------------------------------------------------------")
    print(f"  AUC (Edema vs Resto por intensidad FLAIR): {auc:.4f}")
    print("------------------------------------------------------------")
    
    if auc > 0.75:
        print("  -> SEÑAL FUERTE: FLAIR discrimina el edema pediátrico con alta precisión.")
    elif auc > 0.65:
        print("  -> SEÑAL MODERADA: Separabilidad débil. Riesgo de falsos positivos.")
    else:
        print("  -> SEÑAL NULA (~0.50): El edema no es diferenciable por FLAIR.")
        print("     Decisión científica confirmada: mantener la clase cerrada.")

if __name__ == "__main__":
    main()
