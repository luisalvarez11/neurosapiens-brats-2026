#!/usr/bin/env python3
"""
calibrar_edema_multicanal.py (Versión Hyper-Low-RAM)
====================================================

Extracción de características secuencial por coordenadas.
Jamás carga más de 1 canal MRI en memoria simultáneamente.
"""

import os, glob, gc
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.stats import mannwhitneyu

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

# ---- RUTAS EN ASIMOV --------------------------------------------------------
IMAGES = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/imagesTr"
GT_DIR = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/labelsTr"
CHANNELS = ["0000", "0001", "0002", "0003"]  # t1n, t1c, t2w, t2f
CH_NAMES = ["t1n", "t1c", "t2w", "t2f"]
MAX_VOX_PER_CASE = 1500  # Vóxeles máximos por clase y paciente

def zscore_brain(vol):
    m = vol > 0
    if m.sum() == 0:
        return np.zeros_like(vol, dtype=np.float32)
    mu, sig = vol[m].mean(), vol[m].std()
    z = np.zeros_like(vol, dtype=np.float32)
    z[m] = (vol[m] - mu) / (sig + 1e-6)
    return z

def get_sampled_indices(mask_3d, max_samples):
    idx = np.where(mask_3d.ravel())[0]
    if idx.size > max_samples:
        idx = np.random.choice(idx, max_samples, replace=False)
    return idx

def main():
    gts = sorted(glob.glob(os.path.join(GT_DIR, "*.nii.gz")))
    print(f"-> Analizando {len(gts)} casos multicanal (Modo Hyper-Low-RAM < 200MB)...\n")

    ed_feats, tumor_feats, sano_feats = [], [], []
    casos_con_ed = 0

    for idx, gp in enumerate(gts, 1):
        cid = os.path.basename(gp).replace(".nii.gz", "")
        paths = [os.path.join(IMAGES, f"{cid}_{c}.nii.gz") for c in CHANNELS]
        if not all(os.path.exists(p) for p in paths):
            continue
            
        gt_img = nib.load(gp)
        gt = np.asarray(gt_img.dataobj).astype(np.uint8)
        if (gt == 4).sum() == 0:
            del gt, gt_img
            continue  # Saltamos si no hay Edema
            
        casos_con_ed += 1

        # 1. Definir coordenadas de interés usando SOLO el GT y una máscara cerebral rápida
        t1n_path = os.path.join(IMAGES, f"{cid}_0000.nii.gz")
        t1n_quick = np.asarray(nib.load(t1n_path).dataobj) > 0
        
        whole_tumor = np.isin(gt, [1, 2, 3, 4])
        dilated = ndimage.binary_dilation(whole_tumor, iterations=5)
        sano_mask = dilated & ~whole_tumor & t1n_quick

        ed_idx = get_sampled_indices(gt == 4, MAX_VOX_PER_CASE)
        tumor_idx = get_sampled_indices(np.isin(gt, [1, 2, 3]), MAX_VOX_PER_CASE)
        sano_idx = get_sampled_indices(sano_mask, MAX_VOX_PER_CASE)

        # Destruir máscaras 3D de inmediato
        del gt, gt_img, t1n_quick, whole_tumor, dilated, sano_mask
        
        # 2. Extraer valores canal a canal (secuencialmente)
        p_ed = np.zeros((len(ed_idx), 4), dtype=np.float32)
        p_tumor = np.zeros((len(tumor_idx), 4), dtype=np.float32)
        p_sano = np.zeros((len(sano_idx), 4), dtype=np.float32)

        for ch_i, p in enumerate(paths):
            vol = zscore_brain(np.asarray(nib.load(p).dataobj).astype(np.float32))
            vol_flat = vol.ravel()
            
            if len(ed_idx) > 0: p_ed[:, ch_i] = vol_flat[ed_idx]
            if len(tumor_idx) > 0: p_tumor[:, ch_i] = vol_flat[tumor_idx]
            if len(sano_idx) > 0: p_sano[:, ch_i] = vol_flat[sano_idx]
            
            # Destruir el volumen 3D antes de pasar al siguiente canal
            del vol, vol_flat

        if len(ed_idx) > 0: ed_feats.append(p_ed)
        if len(tumor_idx) > 0: tumor_feats.append(p_tumor)
        if len(sano_idx) > 0: sano_feats.append(p_sano)

        # Recolección de basura estricta por paciente
        del ed_idx, tumor_idx, sano_idx, p_ed, p_tumor, p_sano
        if idx % 5 == 0:
            gc.collect()

        if idx % 10 == 0 or idx == len(gts):
            print(f"   [{idx}/{len(gts)}] Casos edematosos escaneados limpiamente...")

    ed = np.vstack(ed_feats) if ed_feats else np.empty((0, 4))
    tumor = np.vstack(tumor_feats) if tumor_feats else np.empty((0, 4))
    sano = np.vstack(sano_feats) if sano_feats else np.empty((0, 4))

    print(f"\n============================================================")
    print(f"Casos con edema utilizados para calibrar: {casos_con_ed}")
    print(f"Muestras: Edema={len(ed):,} | Tumor={len(tumor):,} | Sano={len(sano):,}")
    print("============================================================\n")

    if len(ed) == 0:
        print("[!] No se encontró señal de Edema en los casos escaneados.")
        return

    # --- Medias por canal ---
    print("Medias z-score por canal:")
    print(f"{'':10}{'t1n':>8}{'t1c':>8}{'t2w':>8}{'t2f':>8}")
    for name, arr in [("EDEMA", ed), ("TUMOR", tumor), ("SANO", sano)]:
        print(f"{name:10}" + "".join(f"{arr[:,i].mean():>8.2f}" for i in range(4)))
    print()

    # --- AUC por canal individual ---
    print("AUC por canal individual (EDEMA vs SANO peritumoral):")
    for i, nm in enumerate(CH_NAMES):
        u, _ = mannwhitneyu(ed[:, i], sano[:, i], alternative='two-sided')
        auc = u / (len(ed) * len(sano))
        print(f"  -> {nm}: {auc:.4f}")
    print()

    # --- Logística multi-canal ---
    def evaluar(pos, neg, etiqueta):
        X = np.vstack([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        if HAVE_SK:
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
            clf.fit(X, y)
            print(f"[{etiqueta}] AUC 4-canales (CV): {auc:.4f}")
            print(f"    Coeficientes: {dict(zip(CH_NAMES, np.round(clf.coef_[0],4)))}")
            print(f"    Intercepto:   {clf.intercept_[0]:.4f}")
        else:
            print(f"[{etiqueta}] sklearn no disponible; instala scikit-learn.")
        print()

    print("------------------------------------------------------------")
    evaluar(ed, tumor, "A) Edema vs TUMOR (Interno)")
    evaluar(ed, sano,  "B) Edema vs SANO peritumoral (Frontera biológica)")
    print("------------------------------------------------------------")
    print("INTERPRETACIÓN:")
    print(" -> Si la comparación B supera AUC > 0.75, podemos recuperar puntos de edema")
    print("    mediante un posprocesado del anillo peritumoral sano.")
    print(" -> Si ambas salen < 0.65, queda cerrado y justificado científicamente.")

if __name__ == "__main__":
    main()
