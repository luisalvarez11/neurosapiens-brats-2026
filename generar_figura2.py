import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# --- CONFIGURACIÓN ---
ruta_flair = "/workspace/nnUNet_data/nnUNet_raw/Dataset502_BraTSPED/imagesTr/BraTS-PED-00002-000_0003.nii.gz"
ruta_gt = "/workspace/nnUNet_data/nnUNet_raw/Dataset502_BraTSPED/labelsTr/BraTS-PED-00002-000.nii.gz"
slice_z = 75 # Cambia este número hasta que veas el tumor bien grande
lambda_val = 5.7 # El valor medio de vuestro paper

# 1. Cargar datos
flair = nib.load(ruta_flair).get_fdata()
gt = nib.load(ruta_gt).get_fdata()

flair_slice = flair[:, :, slice_z]
gt_slice = gt[:, :, slice_z]

# 2. Recrear vuestra Soft Label (Fisher-KPP)
# Sacamos la máscara binaria del core (simplificado para la visualización)
mascara_binaria = (gt_slice > 0).astype(float)

# Distancia Euclidiana (negativa dentro, positiva fuera)
dist_fuera = distance_transform_edt(mascara_binaria == 0)
dist_dentro = distance_transform_edt(mascara_binaria)
d_x = dist_fuera - dist_dentro

# Fórmula de vuestro paper
d_0 = lambda_val * np.log(np.sqrt(2) - 1)
soft_label = np.power(1 + np.exp((d_x - d_0) / lambda_val), -2)

# 3. Plotear la figura (1 fila, 3 columnas)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(np.rot90(flair_slice), cmap='gray')
axes[0].set_title("FLAIR MRI")
axes[0].axis('off')

axes[1].imshow(np.rot90(mascara_binaria), cmap='gray')
axes[1].set_title("Hard Label (Binary)")
axes[1].axis('off')

# Usamos el colormap 'magma' para que el degradado parezca un mapa de calor
im = axes[2].imshow(np.rot90(soft_label), cmap='magma')
axes[2].set_title("Fisher-KPP Soft Label")
axes[2].axis('off')

plt.tight_layout()
# Guardamos en PDF recortado y a máxima calidad
plt.savefig("Figura2_SoftLabels.pdf", bbox_inches='tight', dpi=300)
print("¡Figura 2 guardada!")