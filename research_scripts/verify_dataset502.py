#!/usr/bin/env python3
"""
Verificacion de integridad del Dataset 502 (BraTS-PEDs, post skull-stripping).

Comprueba que:
  1. Cada caso en imagesTr/ tiene sus 4 canales completos (_0000.._0003).
  2. Cada caso con imagenes tiene su label correspondiente en labelsTr/.
  3. No hay labels "huerfanas" (sin imagenes asociadas) - normalmente indica
     un caso que fallo en el stripper mientras aun conserva label copiada.

Util tras el paso 4 (reconstruccion de Dataset 502), por si algun caso se
cae durante el skull-stripping en CPU y hay que decidir si se descarta del
entrenamiento o se reprocesa.

Uso:
    python verify_dataset502.py /ruta/Dataset502
"""

import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_CHANNELS = {"0000", "0001", "0002", "0003"}  # t1n, t1c, t2w, t2f


def collect_image_cases(images_dir: Path):
    """Devuelve {sub_id: set(canales encontrados)}."""
    cases = defaultdict(set)
    for f in images_dir.glob("*.nii.gz"):
        stem = f.name.replace(".nii.gz", "")
        try:
            sub_id, channel = stem.rsplit("_", 1)
        except ValueError:
            print(f"  [aviso] nombre inesperado en imagesTr, se ignora: {f.name}")
            continue
        cases[sub_id].add(channel)
    return cases


def collect_label_ids(labels_dir: Path):
    """Devuelve el set de sub_id con label (asume nombre = sub_id + .nii.gz)."""
    return {f.name.replace(".nii.gz", "") for f in labels_dir.glob("*.nii.gz")}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    for d in (images_dir, labels_dir):
        if not d.exists():
            print(f"Error: no existe {d}")
            sys.exit(1)

    image_cases = collect_image_cases(images_dir)
    label_ids = collect_label_ids(labels_dir)
    image_ids = set(image_cases.keys())

    incomplete = {
        sub_id: EXPECTED_CHANNELS - channels
        for sub_id, channels in image_cases.items()
        if channels != EXPECTED_CHANNELS
    }
    missing_labels = sorted(image_ids - label_ids)
    orphan_labels = sorted(label_ids - image_ids)
    complete_ok = sorted(
        sub_id for sub_id in image_ids
        if sub_id not in incomplete and sub_id in label_ids
    )

    print(f"Casos con imagenes: {len(image_ids)}")
    print(f"Casos con labels:   {len(label_ids)}")
    print(f"Casos completos y consistentes: {len(complete_ok)}\n")

    if incomplete:
        print(f"[!] {len(incomplete)} caso(s) con canales incompletos:")
        for sub_id, faltantes in sorted(incomplete.items()):
            print(f"    {sub_id}: faltan canales {sorted(faltantes)}")
        print()

    if missing_labels:
        print(f"[!] {len(missing_labels)} caso(s) con imagenes pero SIN label:")
        for sub_id in missing_labels:
            print(f"    {sub_id}")
        print()

    if orphan_labels:
        print(f"[!] {len(orphan_labels)} label(s) sin imagenes asociadas (huerfanas):")
        for sub_id in orphan_labels:
            print(f"    {sub_id}")
        print()

    if not incomplete and not missing_labels and not orphan_labels:
        print("Todo consistente: ningun caso incompleto ni huerfano.")

    # Sugerencia de numTraining para el dataset.json
    print(f"\nnumTraining sugerido para dataset.json: {len(complete_ok)}")


if __name__ == "__main__":
    main()
