#!/usr/bin/env python3
"""
Renombrado de casos BraTS-PEDs 2026 para el pipeline de skull-stripping
(d3b-center/peds-brain-auto-skull-strip) y vuelta al formato nnU-Net.

Mapeo de canales (IMPORTANTE):
  El stripper exige nombres [subID]_[imageID].nii.gz con imageID en
  {FL, T1, T1CE, T2}. Ese naming es independiente del orden de canales
  de NUESTRO dataset.json, que es: 0=t1n, 1=t1c, 2=t2w, 3=t2f.
"""

import shutil
import sys
from pathlib import Path

# BraTS -> sufijo que exige el stripper
BRATS_TO_STRIPPER = {
    "t1n":  "T1",
    "t1c":  "T1CE",
    "t2w":  "T2",
    "t2f":  "FL",
}

# Sufijo del stripper -> canal de NUESTRO dataset.json (0=t1n,1=t1c,2=t2w,3=t2f)
STRIPPER_TO_NNUNET = {
    "T1":   "0000",
    "T1CE": "0001",
    "T2":   "0002",
    "FL":   "0003",
}

# Carpetas basura que hay que ignorar al escanear casos
IGNORE_DIRS = {".ipynb_checkpoints", "__pycache__", ".git"}


def to_stripper(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_files, n_cases = 0, 0
    for case_dir in sorted(input_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        if case_dir.name in IGNORE_DIRS:
            continue
        sub_id = case_dir.name
        found = 0
        for modality, suffix in BRATS_TO_STRIPPER.items():
            matches = list(case_dir.glob(f"*-{modality}.nii.gz"))
            if not matches:
                print(f"  [aviso] no encontrado {modality} en {sub_id}")
                continue
            shutil.copy2(matches[0], output_dir / f"{sub_id}_{suffix}.nii.gz")
            n_files += 1
            found += 1
        if found == 4:
            n_cases += 1
        else:
            print(f"  [AVISO] {sub_id}: solo {found}/4 modalidades")
    print(f"\nListo: {n_files} ficheros ({n_cases} casos completos) -> {output_dir}")


def from_stripper(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    n, ignored = 0, []

    for f in sorted(input_dir.glob("*.nii.gz")):
        stem = f.name.replace(".nii.gz", "")
        try:
            sub_id, suffix = stem.rsplit("_", 1)
        except ValueError:
            ignored.append(f.name)
            continue
        if suffix not in STRIPPER_TO_NNUNET:
            ignored.append(f.name)
            continue
        dst = output_dir / f"{sub_id}_{STRIPPER_TO_NNUNET[suffix]}.nii.gz"
        shutil.copy2(f, dst)
        n += 1

    print(f"\nCopiados: {n} ficheros -> {output_dir}")

    if ignored:
        print(f"\n[!] IGNORADOS {len(ignored)} ficheros:")
        for name in ignored[:10]:
            print(f"    {name}")
        if len(ignored) > 10:
            print(f"    ... y {len(ignored) - 10} mas")

    # FALLO RUIDOSO: si no se copio nada, el naming de salida no coincide
    if n == 0:
        print("\n" + "=" * 60)
        print("[ERROR] No se copio NINGUN fichero.")
        print("El naming de salida del stripper no coincide con lo esperado")
        print(f"(sufijos esperados: {sorted(STRIPPER_TO_NNUNET)}).")
        print("Mira los nombres reales en la carpeta de salida y ajusta")
        print("STRIPPER_TO_NNUNET en este script.")
        print("=" * 60)
        sys.exit(1)


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("to_stripper", "from_stripper"):
        print(__doc__)
        sys.exit(1)

    mode, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

    if not src.exists():
        sys.exit(f"Error: no existe {src}")

    if mode == "to_stripper":
        to_stripper(src, dst)
    else:
        from_stripper(src, dst)


if __name__ == "__main__":
    main()