import os
import sys
import glob
import numpy as np
import nibabel as nib

def calcular_metricas_edema(ruta_ref, ruta_pred):
    # Cargar las imágenes como arrays de numpy
    ref_data = np.asanyarray(nib.load(ruta_ref).dataobj)
    pred_data = np.asanyarray(nib.load(ruta_pred).dataobj)
    
    # La etiqueta del Edema es 4 según tu predict.py
    LBL_EDEMA = 4
    
    # Crear máscaras booleanas (True donde hay Edema, False en el resto)
    mask_ref = (ref_data == LBL_EDEMA)
    mask_pred = (pred_data == LBL_EDEMA)
    
    # Verdaderos Positivos: Es 4 en la referencia Y es 4 en la predicción
    tp = np.sum(mask_ref & mask_pred)
    
    # Extras para contexto
    fp = np.sum((~mask_ref) & mask_pred) # Falsos Positivos
    fn = np.sum(mask_ref & (~mask_pred)) # Falsos Negativos
    
    return tp, fp, fn

def evaluar_directorios(dir_ref, dir_pred):
    archivos_ref = sorted(glob.glob(os.path.join(dir_ref, "*.nii.gz")))
    
    if not archivos_ref:
        print(f"No se encontraron archivos .nii.gz en {dir_ref}")
        return
        
    print("Calculando métricas para EDEMA (Label 4)...")
    print("-" * 70)
    
    tp_total = 0
    
    for ruta_ref in archivos_ref:
        nombre_archivo = os.path.basename(ruta_ref)
        ruta_pred = os.path.join(dir_pred, nombre_archivo)
        
        if not os.path.exists(ruta_pred):
            print(f"[!] Falta {nombre_archivo} en la carpeta de Docker.")
            continue
            
        tp, fp, fn = calcular_metricas_edema(ruta_ref, ruta_pred)
        tp_total += tp
        
        print(f"Caso: {nombre_archivo}")
        print(f"  -> Verdaderos Positivos (TP) : {tp} voxeles idénticos")
        if fp > 0 or fn > 0:
            print(f"  -> Discrepancias encontradas: Falsos Positivos={fp} | Falsos Negativos={fn}")
            
    print("-" * 70)
    print(f"TOTAL Verdaderos Positivos (Edema) acumulados: {tp_total}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python tp_edema.py <dir_servidor> <dir_docker>")
        sys.exit(1)
        
    evaluar_directorios(sys.argv[1], sys.argv[2])