import os, glob
import nibabel as nib
import numpy as np
from scipy import ndimage

IMAGES_VAL = "/workspace/imagesVal"      # imágenes validation formato nnU-Net
PREDS_VAL  = "/workspace/predsVal"       # predicción CRUDA de nnU-Net
PREDS_SUB  = "/workspace/predsVal_final" # salida lista para Synapse
os.makedirs(PREDS_SUB, exist_ok=True)

# --- Clasificador calibrado (Fase 2) ---
COEF_T1C = -0.5844
COEF_T2W =  1.0382
INTERCEPT = -2.2613
MIN_CC_VOXELS = 15   # descarta clusters diminutos

def zscore_brain(vol):
    mask = vol > 0
    if mask.sum() == 0: return np.zeros_like(vol, dtype=np.float32)
    mu, sigma = vol[mask].mean(), vol[mask].std()
    z = np.zeros_like(vol, dtype=np.float32)
    z[mask] = (vol[mask] - mu) / (sigma + 1e-6)
    return z

files = sorted(glob.glob(os.path.join(PREDS_VAL, "*.nii.gz")))
print(f"-> Procesando {len(files)} pacientes con regresión logística balanceada...")

for idx, pf in enumerate(files, 1):
    cid = os.path.basename(pf).replace(".nii.gz", "")
    seg_img = nib.load(pf)
    seg = np.asarray(seg_img.dataobj).astype(np.uint8)

    t1c_path = os.path.join(IMAGES_VAL, f"{cid}_0001.nii.gz")
    t2w_path = os.path.join(IMAGES_VAL, f"{cid}_0002.nii.gz")
    
    if not os.path.exists(t1c_path) or not os.path.exists(t2w_path):
        print(f" [!] Error: No se encuentran las imágenes para {cid}")
        continue

    t1c = np.asarray(nib.load(t1c_path).dataobj).astype(np.float32)
    t2w = np.asarray(nib.load(t2w_path).dataobj).astype(np.float32)
    t1c_z, t2w_z = zscore_brain(t1c), zscore_brain(t2w)

    out = np.zeros_like(seg)
    out[seg == 1] = 4   # WT-solo de nnU-Net -> Edema (ED) oficial
    out[seg == 3] = 1   # ET de nnU-Net -> Realce (ET) oficial

    # Core ambiguo (NET + CC fusionados)
    ambiguous = (seg == 2)

    # Regresión Logística: prob(CC) = sigmoide(COEF·z + intercepto)
    logit = COEF_T1C * t1c_z + COEF_T2W * t2w_z + INTERCEPT
    prob_cc = 1.0 / (1.0 + np.exp(-logit))
    cc_candidate = ambiguous & (prob_cc > 0.5)

    # Filtrar clusters diminutos de CC de forma vectorizada (muy rápido)
    labeled, n = ndimage.label(cc_candidate)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        mask_sizes = sizes > MIN_CC_VOXELS
        mask_sizes[0] = False # Ignorar el fondo
        cc_final = mask_sizes[labeled]
    else:
        cc_final = np.zeros_like(cc_candidate)

    out[ambiguous & cc_final]  = 3   # Componente Quístico (CC)
    out[ambiguous & ~cc_final] = 2   # Core No Realzado (NETC)

    nib.save(nib.Nifti1Image(out, seg_img.affine, seg_img.header),
             os.path.join(PREDS_SUB, os.path.basename(pf)))
    
    if idx % 10 == 0 or idx == len(files):
        print(f"   [{idx}/{len(files)}] Completados...")

print("\nHecho ->", PREDS_SUB)
print("Ficheros generados:", len(glob.glob(os.path.join(PREDS_SUB, '*.nii.gz'))))
