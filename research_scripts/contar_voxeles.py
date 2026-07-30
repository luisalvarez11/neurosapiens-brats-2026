import sys
import numpy as np
import nibabel as nib

def contar_voxeles(nii_path):
    try:
        # Cargar el archivo NIfTI
        img = nib.load(nii_path)
        data = np.asanyarray(img.dataobj)
        
        # Obtener las etiquetas únicas y el número total de voxeles por etiqueta
        etiquetas, conteos = np.unique(data, return_counts=True)
        
        # Mapeo de etiquetas según tu predict.py
        nombres_etiquetas = {
            0: "Background",
            1: "ET (Enhancing Tumor)",
            2: "NET (Non-Enhancing Tumor)",
            3: "CC (Cystic Component)",
            4: "ED (Edema)"
        }
        
        print(f"--- Conteo de voxeles para: {nii_path} ---")
        for etiqueta, cantidad in zip(etiquetas, conteos):
            nombre = nombres_etiquetas.get(int(etiqueta), "Etiqueta desconocida")
            print(f"Label {int(etiqueta)} [{nombre}]: {cantidad} voxeles")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error al procesar el archivo {nii_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python contar_voxeles.py <ruta_al_archivo.nii.gz>")
        sys.exit(1)
        
    # Permite pasar varios archivos a la vez separados por espacio
    for archivo in sys.argv[1:]:
        contar_voxeles(archivo)