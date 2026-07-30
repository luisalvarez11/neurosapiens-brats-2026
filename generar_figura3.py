import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
# 1. El FLAIR (Original limpio)
ruta_flair = "/workspace/nnUNet_data/nnUNet_raw/Dataset502_BraTSPED/imagesTr/BraTS-PED-00039-000_0003.nii.gz"

# 2. El Ground Truth (La máscara experta)
ruta_gt = "/workspace/nnUNet_data/nnUNet_raw/Dataset502_BraTSPED/labelsTr/BraTS-PED-00039-000.nii.gz"

# 3. La predicción del Baseline (Dataset 502)
ruta_baseline = "/workspace/nnUNet_data/nnUNet_results/Dataset502_BraTSPED/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation_v2/BraTS-PED-00039-000.nii.gz"

# 4. La predicción del Ensemble (Dataset 501 con su trainer de 250 épocas)
ruta_ensemble = "/workspace/nnUNet_data/nnUNet_results/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation_v2/BraTS-PED-00039-000.nii.gz"
slice_z = 75 # Elige el slice donde el baseline falle y el ensemble acierte

def cargar_slice(ruta, z):
    return np.rot90(nib.load(ruta).get_fdata()[:, :, z])

flair = cargar_slice(ruta_flair, slice_z)
gt = cargar_slice(ruta_gt, slice_z)
baseline = cargar_slice(ruta_baseline, slice_z)
ensemble = cargar_slice(ruta_ensemble, slice_z)

# Crear figura
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titulos = ["FLAIR", "Ground Truth", "Baseline (Hard)", "Ours (Ensemble)"]
imagenes = [None, gt, baseline, ensemble]

for i in range(4):
    # Pintar el cerebro de fondo siempre
    axes[i].imshow(flair, cmap='gray')
    
    # Pintar la máscara por encima si no es la primera columna
    if i > 0:
        # Colorear: 1 (ET, Amarillo), 2 (NET, Rojo), 3 (CC, Verde)
        # Esto asume las etiquetas originales de BraTS
        mascara = np.ma.masked_where(imagenes[i] == 0, imagenes[i])
        axes[i].imshow(mascara, cmap='Set1', alpha=0.6, vmin=1, vmax=4)
        
    axes[i].set_title(titulos[i], fontsize=14, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("Figura3_Resultados.pdf", bbox_inches='tight', dpi=300)
print("¡Figura 3 guardada!")