import os, glob, zipfile, numpy as np, nibabel as nib

zip_path = '/workspace/submission_soft_val_CC090.zip'
nii_dir = '/workspace/soft_submission_pipeline/final_nii_for_zip'

print('=== 🔍 AUDITORÍA DE CALIDAD PRE-SUMISIÓN SYNAPSE ===\n')

# 1. VERIFICACIÓN DE INTEGRIDAD DEL ZIP
if not os.path.exists(zip_path):
    print('❌ ERROR CRÍTICO: No se encuentra el archivo .zip.')
    exit()

with zipfile.ZipFile(zip_path, 'r') as z:
    file_list = z.namelist()
    
print(f'📦 Archivo ZIP: {os.path.basename(zip_path)} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)')
print(f'📄 Total de archivos dentro del ZIP: {len(file_list)}')

# Verificar que sea un ZIP plano (sin subcarpetas)
has_folders = any('/' in f or '\\' in f for f in file_list)
if has_folders:
    print('❌ ALERTA DE FORMATO: El ZIP contiene subcarpetas. Synapse exige un ZIP plano.')
else:
    print('✅ Estructura plana del ZIP: Correcta (sin subcarpetas).')

# 2. AUDITORÍA DE ETIQUETAS Y VOLÚMENES
files = sorted(glob.glob(os.path.join(nii_dir, '*.nii.gz')))
if len(files) == 0:
    print('❌ No hay archivos .nii.gz en la carpeta de salida para auditar.')
    exit()

print(f'\n🧠 Auditando anatomía y etiquetas en {len(files)} casos...')

all_unique = set()
class_counts = {1: 0, 2: 0, 3: 0, 4: 0} # 1:ET, 2:NET, 3:CC, 4:ED
empty_cases = []

for f in files:
    img = nib.load(f)
    data = np.asanyarray(img.dataobj)
    
    if np.sum(data) == 0:
        empty_cases.append(os.path.basename(f))
        
    uniques = np.unique(data)
    all_unique.update(uniques)
    
    for c in [1, 2, 3, 4]:
        class_counts[c] += np.sum(data == c)

print('\n--- RESULTADOS DE LA AUDITORÍA ---')
print(f'🔢 Valores únicos encontrados en todo el dataset: {sorted(list(all_unique))}')

valid_set = {0, 1, 2, 3, 4}
if not all_unique.issubset(valid_set):
    print(f'❌ ERROR CRÍTICO: Se han detectado etiquetas ilegales: {all_unique - valid_set}. Synapse dará error 0.')
else:
    print('✅ Esquema de etiquetas oficial BraTS-PEDs 2026: Intacto [0, 1, 2, 3, 4].')

num_cases = len(files)
print('\n📊 Promedio de vóxeles predichos por paciente:')
print(f'   • [Etiqueta 1 - ET (Enhancing)]       : {class_counts[1]/num_cases:10.1f} voxels')
print(f'   • [Etiqueta 2 - NET (Non-enhancing)]  : {class_counts[2]/num_cases:10.1f} voxels')
print(f'   • [Etiqueta 3 - CC (Cystic Component)]: {class_counts[3]/num_cases:10.1f} voxels')
print(f'   • [Etiqueta 4 - ED (Edema)]           : {class_counts[4]/num_cases:10.1f} voxels')

if class_counts[3] > 0:
    print('\n🎯 CONFIRMACIÓN LOGÍSTICA: ¡La clase 3 (CC) tiene volumen! La regresión logística ha separado quistes con éxito.')
else:
    print('\n⚠️ ADVERTENCIA: La clase 3 (CC) está en 0 en todo el dataset. Revisa los umbrales de la logística.')

if empty_cases:
    print(f'\n⚠️ ALERTA: Hay {len(empty_cases)} pacientes donde no se predijo ningún tumor: {empty_cases[:3]}...')
else:
    print('\n✅ Sanity Check total: 100% de los pacientes tienen tumor predicho sin máscaras vacías.')

print('\n🚀 DICTAMEN: Si todo está en verde arriba, el archivo está BLOQUEADO Y LISTO PARA SUBIR.')
