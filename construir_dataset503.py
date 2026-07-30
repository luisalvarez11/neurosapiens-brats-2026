#!/usr/bin/env python3
"""
construir_dataset503.py
=======================

Construye el Dataset 503: imagenes CON craneo (del 501) + el soft label WT
como canal extra (_0004), para el fine-tuning con soft labels de Fisher-KPP.

Estructura resultante:
  Dataset503_BraTSPED/
    imagesTr/
      BraTS-PED-XXXXX-YYY_0000.nii.gz  (t1n)
      BraTS-PED-XXXXX-YYY_0001.nii.gz  (t1c)
      BraTS-PED-XXXXX-YYY_0002.nii.gz  (t2w)
      BraTS-PED-XXXXX-YYY_0003.nii.gz  (t2f)
      BraTS-PED-XXXXX-YYY_0004.nii.gz  (soft label WT)  <-- NUEVO
    labelsTr/
      BraTS-PED-XXXXX-YYY.nii.gz       (labels duras 1/2/3/4, sin cambios)
    dataset.json   (5 canales; el canal 4 marcado noNorm)
"""

import os, glob, json, shutil
import numpy as np
import nibabel as nib

# ---- RUTAS (AJUSTAR) --------------------------------------------------------
RAW = "/workspace/nnUNet_data/nnUNet_raw"
DS501 = os.path.join(RAW, "Dataset501_BraTSPED")      # imagenes con craneo + labels
SOFT_DIR = os.path.join(RAW, "Dataset502_BraTSPED", "labelsTr_soft")  # soft labels ya generados
DS503 = os.path.join(RAW, "Dataset503_BraTSPED")      # destino

SOFT_SUFFIX = "-softlabel.nii.gz"   # como se llaman los soft labels


def main():
    img_out = os.path.join(DS503, "imagesTr")
    lbl_out = os.path.join(DS503, "labelsTr")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # Casos: a partir de las labels del 501 (los que tienen GT)
    labels = sorted(glob.glob(os.path.join(DS501, "labelsTr", "*.nii.gz")))
    print(f"Casos en el 501: {len(labels)}")

    n_ok, sin_soft, sin_mod = 0, [], []

    for lp in labels:
        cid = os.path.basename(lp).replace(".nii.gz", "")

        # --- comprobar que existen las 4 modalidades ---
        mods = [os.path.join(DS501, "imagesTr", f"{cid}_{i:04d}.nii.gz") for i in range(4)]
        if not all(os.path.exists(m) for m in mods):
            sin_mod.append(cid); continue

        # --- comprobar que existe el soft label ---
        soft_p = os.path.join(SOFT_DIR, cid + SOFT_SUFFIX)
        if not os.path.exists(soft_p):
            sin_soft.append(cid); continue

        # --- copiar las 4 modalidades tal cual ---
        for i, m in enumerate(mods):
            shutil.copy2(m, os.path.join(img_out, f"{cid}_{i:04d}.nii.gz"))

        # --- el soft label pasa a ser el canal _0004 ---
        ref = nib.load(mods[0])
        soft_img = nib.load(soft_p)
        soft = np.asanyarray(soft_img.dataobj).astype(np.float32)

        if soft.shape != ref.shape:
            print(f"  [!] {cid}: shape soft {soft.shape} != img {ref.shape}. Se salta.")
            for i in range(4):
                p = os.path.join(img_out, f"{cid}_{i:04d}.nii.gz")
                if os.path.exists(p): os.remove(p)
            continue

        # Guardar el soft usando el affine/header de la imagen de referencia
        nib.save(nib.Nifti1Image(soft, ref.affine, ref.header),
                 os.path.join(img_out, f"{cid}_0004.nii.gz"))

        # --- copiar la label dura sin cambios ---
        shutil.copy2(lp, os.path.join(lbl_out, f"{cid}.nii.gz"))

        n_ok += 1
        if n_ok % 25 == 0:
            print(f"  {n_ok} casos construidos...")

    # --- dataset.json ---
    dataset_json = {
        "channel_names": {
            "0": "t1n",
            "1": "t1c",
            "2": "t2w",
            "3": "t2f",
            "4": "noNorm"          # <-- soft label WT, sin normalizacion
        },
        "labels": {
            "background": 0,
            "WT": [1, 2, 3, 4],
            "TC": [1, 2, 3],
            "ET": [1]
        },
        "regions_class_order": [1, 2, 3],
        "numTraining": n_ok,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    with open(os.path.join(DS503, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("\n" + "=" * 55)
    print(f"Dataset 503 construido: {n_ok} casos")
    print(f"  imagesTr: {img_out}")
    print(f"  labelsTr: {lbl_out}")
    if sin_mod:
        print(f"  [!] {len(sin_mod)} casos sin las 4 modalidades")
    if sin_soft:
        print(f"  [!] {len(sin_soft)} casos SIN soft label:")
        for c in sin_soft[:10]:
            print(f"        {c}")
    print("=" * 55)
    print("\nSiguiente paso:")
    print("  1. Mover los planes del 502 al 503 (compatibilidad de pesos)")
    print("  2. Preprocesar el 503")
    print("  Comandos:")
    print("    nnUNetv2_extract_fingerprint -d 503")
    print("    nnUNetv2_move_plans_between_datasets -s 502 -t 503 -sp nnUNetPlans -tp nnUNetPlans")
    print("    nnUNetv2_preprocess -d 503 -c 3d_fullres -plans_name nnUNetPlans")


if __name__ == "__main__":
    main()
