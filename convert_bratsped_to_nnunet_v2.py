#!/usr/bin/env python3
"""
convert_bratsped_to_nnunet.py
=============================

Convierte BraTS-PEDs al formato nnU-Net v2.

BraTS-PEDs viene como una carpeta por caso:
    BraTS-PED-00001-000/
        BraTS-PED-00001-000-t1n.nii.gz   (T1 native)
        BraTS-PED-00001-000-t1c.nii.gz   (T1 contrast)
        BraTS-PED-00001-000-t2w.nii.gz   (T2)
        BraTS-PED-00001-000-t2f.nii.gz   (FLAIR)
        BraTS-PED-00001-000-seg.nii.gz   (consenso)

nnU-Net v2 espera:
    nnUNet_raw/Dataset501_BraTSPED/
        imagesTr/BraTS-PED-00001-000_0000.nii.gz  (canal 0 = t1n)
        imagesTr/BraTS-PED-00001-000_0001.nii.gz  (canal 1 = t1c)
        imagesTr/BraTS-PED-00001-000_0002.nii.gz  (canal 2 = t2w)
        imagesTr/BraTS-PED-00001-000_0003.nii.gz  (canal 3 = t2f)
        labelsTr/BraTS-PED-00001-000.nii.gz
        dataset.json

IMPORTANTE sobre las etiquetas BraTS: los valores originales son
    1 = ET (enhancing tumor)
    2 = NET (non-enhancing tumor)
    3 = CC (cystic component)
    4 = ED (edema)
    [CORREGIDO tras bug de evaluacion - ver dataset.json regions_class_order]
nnU-Net puede entrenarlas tal cual con "region-based training" para predecir
las regiones anidadas WT/TC/ET que evalúa el challenge. Aquí montamos el
dataset.json con regiones para que las métricas cuadren con Synapse.

Uso:
    python convert_bratsped_to_nnunet.py \
        --src /ruta/a/BraTS-PEDs/training \
        --dataset_id 501 \
        --dataset_name BraTSPED

Requiere: nnUNet_raw exportado como variable de entorno (o pásalo con --raw).
"""
import os, sys, json, shutil, argparse, glob
import numpy as np
import nibabel as nib


# canal -> sufijo de modalidad BraTS. El ORDEN define los _0000.._0003
MODALITIES = ['t1n', 't1c', 't2w', 't2f']


def find_cases(src_paths):
    """Lista carpetas de caso que contengan las 4 modalidades + seg desde una o varias carpetas."""
    # Por seguridad: si nos pasan un string suelto, lo convertimos en lista
    if isinstance(src_paths, str):
        src_paths = [src_paths]
        
    cases = []
    for src in src_paths:
        print(f"  -> Explorando directorio: {src} ...")
        for d in sorted(glob.glob(os.path.join(src, '*'))):
            if not os.path.isdir(d):
                continue
            cid = os.path.basename(d)
            # tolera tanto -t2f.nii.gz como _t2f.nii.gz (bug de guion bajo)
            def grab(mod):
                for pat in (f'*{mod}.nii.gz', f'*{mod}_nii.gz'):
                    hits = glob.glob(os.path.join(d, pat))
                    if hits:
                        return hits[0]
                return None
            mods = {m: grab(m) for m in MODALITIES}
            seg = grab('seg')
            if all(mods.values()) and seg:
                cases.append((cid, mods, seg))
            else:
                missing = [m for m in MODALITIES if not mods[m]] + ([] if seg else ['seg'])
                print(f"  AVISO: {cid} en {src} sin {missing}, se salta")
    return cases


def safe_load_save(src_path, dst_path):
    """Carga y re-guarda un NIfTI (repara el problema de extensión _nii.gz)."""
    img = nib.load(src_path)
    nib.save(img, dst_path)


def convert_label(seg_path, dst_path):
    """Copia la segmentación. Mantiene los enteros originales de BraTS (1/2/3/4).
    Verifica qué labels hay realmente (útil para detectar CC ausente)."""
    img = nib.load(seg_path)
    data = np.asarray(img.dataobj)
    present = sorted(int(x) for x in np.unique(data))
    nib.save(img, dst_path)
    return present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, nargs='+', help='Carpetas training de BraTS-PEDs (acepta varias separadas por espacio)')
    ap.add_argument('--dataset_id', type=int, default=501)
    ap.add_argument('--dataset_name', default='BraTSPED')
    ap.add_argument('--raw', default=os.environ.get('nnUNet_raw'),
                    help='carpeta nnUNet_raw (o export nnUNet_raw)')
    args = ap.parse_args()

    if not args.raw:
        sys.exit("ERROR: define nnUNet_raw (export nnUNet_raw=/ruta) o pasa --raw")

    ds_folder = os.path.join(args.raw, f"Dataset{args.dataset_id:03d}_{args.dataset_name}")
    imagesTr = os.path.join(ds_folder, 'imagesTr')
    labelsTr = os.path.join(ds_folder, 'labelsTr')
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    print(f"Buscando casos en {args.src} ...")
    cases = find_cases(args.src)
    print(f"Encontrados {len(cases)} casos válidos.\n")
    if not cases:
        sys.exit("No hay casos. Revisa rutas y nombres de archivo.")

    all_labels = set()
    for i, (cid, mods, seg) in enumerate(cases, 1):
        for ch, mod in enumerate(MODALITIES):
            dst = os.path.join(imagesTr, f"{cid}_{ch:04d}.nii.gz")
            safe_load_save(mods[mod], dst)
        lbl_dst = os.path.join(labelsTr, f"{cid}.nii.gz")
        present = convert_label(seg, lbl_dst)
        all_labels.update(present)
        if i % 25 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)} convertidos")

    print(f"\nLabels presentes en el dataset: {sorted(all_labels)}")

    # dataset.json con region-based training para WT/TC/ET
    # labels BraTS: 1=NETC, 2=ED, 3=ET, 4=CC (si existe)
    # Regiones anidadas evaluadas por el challenge:
    #   WT (whole tumor)  = 1,2,3(,4)
    #   TC (tumor core)   = 1,3(,4)
    #   ET (enhancing)    = 3
    has_cc = 4 in all_labels
    # Esquema oficial BraTS-PEDs 2026 CORREGIDO para Region-Based Training
    # Esquema oficial BraTS-PEDs 2026: ET=1, NET=2, CC=3, ED=4
    labels = {
        "background": 0,
        "WT": [1, 2, 3, 4],   # ET + NET + CC + ED
        "TC": [1, 2, 3],      # ET + NET + CC
        "ET": [1],            # solo enhancing
    }
    dataset_json = {
        "channel_names": {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"},
        "labels": labels,
        "regions_class_order": [4, 2, 1],   # WT-solo=ED(4), TC-solo=NET+CC(2), ET=ET(1) real
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(os.path.join(ds_folder, 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset creado en:\n  {ds_folder}")
    print("dataset.json:")
    print(json.dumps(dataset_json, indent=2))
    print("\nSIGUIENTE PASO:")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()