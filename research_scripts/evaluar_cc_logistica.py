import os, glob, random
import nibabel as nib
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.ndimage import label
from sklearn.linear_model import LogisticRegression

# --- CONFIGURACIÓN EN ASIMOV ---
TRAIN_IMG = '/workspace/nnUNet_data/nnUNet_raw/Dataset502_BraTSPED/imagesTr'
TRAIN_LBL = '/workspace/nnUNet_data/nnUNet_raw/Dataset502_BraTSPED/labelsTr'
MIN_CC_VOXELS = 50       # Filtro para eliminar ruido/clusters enanos de quiste
VAL_SPLIT_RATIO = 0.20   # 20% de pacientes para validación honesta (Hold-out)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def zscore_brain(vol):
    mask = vol > 0
    if mask.sum() == 0: return np.zeros_like(vol, dtype=np.float32)
    mu, sigma = vol[mask].mean(), vol[mask].std()
    z = np.zeros_like(vol, dtype=np.float32)
    z[mask] = (vol[mask] - mu) / (sigma + 1e-6)
    return z

# 1. RECUPERAR ARCHIVOS Y DIVIDIR PACIENTES (Evitar fuga de datos por vóxel)
all_files = sorted(glob.glob(os.path.join(TRAIN_LBL, '*.nii.gz')))
random.shuffle(all_files)
val_size = int(len(all_files) * VAL_SPLIT_RATIO)
val_files = all_files[:val_size]
train_files = all_files[val_size:]

print(f"-> Total pacientes: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

# 2. EXTRAER VÓXELES DEL SPLIT DE ENTRENAMIENTO (80%)
print("\n[Fase 1/3] Extrayendo vóxeles de calibración (80% Train)...")
net_t1c, net_t2w, cc_t1c, cc_t2w = [], [], [], []

for lbl_path in train_files:
    cid = os.path.basename(lbl_path).replace('.nii.gz', '')
    gt = np.asarray(nib.load(lbl_path).dataobj).astype(np.uint8)
    
    img_t1c_path = os.path.join(TRAIN_IMG, f'{cid}_0001.nii.gz')
    img_t2w_path = os.path.join(TRAIN_IMG, f'{cid}_0002.nii.gz')
    if not os.path.exists(img_t1c_path) or not os.path.exists(img_t2w_path): continue
    
    t1c_z = zscore_brain(np.asarray(nib.load(img_t1c_path).dataobj).astype(np.float32))
    t2w_z = zscore_brain(np.asarray(nib.load(img_t2w_path).dataobj).astype(np.float32))
    
    if (gt == 2).any():
        net_t1c.append(t1c_z[gt == 2])
        net_t2w.append(t2w_z[gt == 2])
    if (gt == 3).any():
        cc_t1c.append(t1c_z[gt == 3])
        cc_t2w.append(t2w_z[gt == 3])

net_t1c = np.concatenate(net_t1c) if net_t1c else np.array([])
net_t2w = np.concatenate(net_t2w) if net_t2w else np.array([])
cc_t1c  = np.concatenate(cc_t1c) if cc_t1c else np.array([])
cc_t2w  = np.concatenate(cc_t2w) if cc_t2w else np.array([])

print(f" -> Vóxeles NET (Core): {len(net_t1c):,} | Vóxeles CC (Quiste): {len(cc_t1c):,}")

# 3. ANÁLISIS DE SEPARABILIDAD (AUC Mann-Whitney U)
print("\n[Fase 2/3] Evaluando AUC por modalidad individual...")
for name, net_arr, cc_arr in [("T1c", net_t1c, cc_t1c), ("T2w", net_t2w, cc_t2w)]:
    if len(cc_arr) == 0 or len(net_arr) == 0: continue
    u, _ = mannwhitneyu(cc_arr, net_arr, alternative='two-sided')
    auc = u / (len(cc_arr) * len(net_arr))
    print(f" -> AUC {name} (separando CC de NET): {auc:.4f}")

# 4. ENTRENAR REGRESIÓN LOGÍSTICA BALANCEADA
X_train = np.column_stack([np.concatenate([net_t1c, cc_t1c]), np.concatenate([net_t2w, cc_t2w])])
y_train = np.concatenate([np.zeros(len(net_t1c)), np.ones(len(cc_t1c))]) # 0=NET, 1=CC

clf = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
print("\n -> Regresión Logística Balanceada Ajustada:")
print(f"    Coeficientes [T1c, T2w]: {clf.coef_[0].round(4)}")
print(f"    Intercepto: {clf.intercept_[0]:.4f}")

# 5. VALIDACIÓN HONESTA A NIVEL DE CASO (20% Val)
print("\n[Fase 3/3] Midiendo Dice real de CC a nivel de caso en el 20% Hold-out...")
dice_cc_scores = []

for lbl_path in val_files:
    cid = os.path.basename(lbl_path).replace('.nii.gz', '')
    gt = np.asarray(nib.load(lbl_path).dataobj).astype(np.uint8)
    
    # Solo evaluamos donde hay Core (2) o Quiste (3) en la máscara
    region_mask = (gt == 2) | (gt == 3)
    if not (gt == 3).any(): continue # Saltar casos que no tienen quiste en GT para no alterar media
    
    img_t1c_path = os.path.join(TRAIN_IMG, f'{cid}_0001.nii.gz')
    img_t2w_path = os.path.join(TRAIN_IMG, f'{cid}_0002.nii.gz')
    if not os.path.exists(img_t1c_path) or not os.path.exists(img_t2w_path): continue
    
    t1c_z = zscore_brain(np.asarray(nib.load(img_t1c_path).dataobj).astype(np.float32))
    t2w_z = zscore_brain(np.asarray(nib.load(img_t2w_path).dataobj).astype(np.float32))
    
    # Predecir sobre los vóxeles ambiguos con el modelo de regresión
    coords = np.where(region_mask)
    X_val = np.column_stack([t1c_z[coords], t2w_z[coords]])
    preds = clf.predict(X_val)
    
    pred_cc_mask = np.zeros_like(gt, dtype=bool)
    pred_cc_mask[coords] = (preds == 1)
    
    # Posprocesado espacial: Eliminar componentes conectados pequeños (ruido)
    labeled_vol, num_features = label(pred_cc_mask)
    for feat_id in range(1, num_features + 1):
        if (labeled_vol == feat_id).sum() < MIN_CC_VOXELS:
            pred_cc_mask[labeled_vol == feat_id] = False
            
    # Calcular Dice espacial exactamente como hace Synapse
    gt_cc_mask = (gt == 3)
    intersection = (pred_cc_mask & gt_cc_mask).sum()
    total_voxels = pred_cc_mask.sum() + gt_cc_mask.sum()
    dice = (2.0 * intersection) / (total_voxels + 1e-6)
    dice_cc_scores.append(dice)

mean_dice = np.mean(dice_cc_scores) if dice_cc_scores else 0.0
median_dice = np.median(dice_cc_scores) if dice_cc_scores else 0.0
print(f"\n=================== [ RESULTADO DEFINITIVO ] ===================")
print(f" -> Dice medio para CC (Quiste):   {mean_dice:.4f}")
print(f" -> Dice mediana para CC (Quiste): {median_dice:.4f} (sobre {len(dice_cc_scores)} casos con quiste)")
print("==================================================================")
