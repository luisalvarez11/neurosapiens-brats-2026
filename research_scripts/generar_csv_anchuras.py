#!/usr/bin/env python3
"""
generar_csv_anchuras.py
=======================

Recorre los casos de BraTS-PEDs y genera un CSV con la anchura de transicion
medida por paciente (transition_width_mm), usando run_cohort_nifti del
distance_profile.py de Jesus.

Ese CSV es lo que necesitan los soft labels de Fisher-KPP.

Uso:
    python generar_csv_anchuras.py \
        --src /workspace/Dataset501 \
        --out /workspace/anchuras_294.csv

Requiere distance_profile.py en el mismo directorio (o en el PYTHONPATH).
"""

import argparse
import csv
import glob
import os
import sys

# Importa la funcion del script de Jesus
try:
    from distance_profile import run_cohort_nifti
except ImportError:
    sys.exit("ERROR: no encuentro distance_profile.py. Ponlo en la misma carpeta.")


def find_pairs(src):
    """
    Para cada carpeta de caso, localiza el FLAIR (t2f) y la segmentacion (seg).
    Devuelve lista de (patient_id, flair_path, seg_path).
    """
    pairs = []
    sin_flair, sin_seg = [], []

    for case_dir in sorted(glob.glob(os.path.join(src, "*"))):
        if not os.path.isdir(case_dir):
            continue
        cid = os.path.basename(case_dir)
        if cid.startswith("."):          # ignora .ipynb_checkpoints etc
            continue

        flair = glob.glob(os.path.join(case_dir, "*-t2f.nii.gz"))
        seg   = glob.glob(os.path.join(case_dir, "*-seg.nii.gz"))

        if not flair:
            sin_flair.append(cid); continue
        if not seg:
            sin_seg.append(cid); continue

        pairs.append((cid, flair[0], seg[0]))

    if sin_flair:
        print(f"  [aviso] {len(sin_flair)} casos sin FLAIR (t2f)")
    if sin_seg:
        print(f"  [aviso] {len(sin_seg)} casos sin seg")

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="carpeta con las carpetas de caso BraTS-PED-XXXXX-YYY/")
    ap.add_argument("--out", default="anchuras.csv",
                    help="ruta del CSV de salida")
    args = ap.parse_args()

    print(f"Buscando casos en {args.src} ...")
    pairs = find_pairs(args.src)
    print(f"Casos con FLAIR + seg: {len(pairs)}")

    if not pairs:
        sys.exit("No hay casos validos. Revisa la ruta --src.")

    print("Midiendo anchuras (esto tarda un poco, es CPU)...")
    # run_cohort_nifti devuelve (rows, profiles) segun el script de Jesus
    rows, profiles = run_cohort_nifti(pairs)

    if not rows:
        sys.exit("run_cohort_nifti no devolvio filas. Revisa distance_profile.py.")

    # Escribir CSV con todas las columnas que devuelva
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nCSV escrito: {args.out}")
    print(f"Filas: {len(rows)}")
    print(f"Columnas: {fieldnames}")

    # Resumen rapido de la anchura si existe la columna
    width_col = None
    for cand in ("transition_width_mm", "width_gradient_mm", "width_10_90"):
        if cand in fieldnames:
            width_col = cand
            break
    if width_col:
        import statistics
        vals = [float(r[width_col]) for r in rows
                if r[width_col] not in (None, "", "nan")]
        if vals:
            print(f"\n{width_col}:")
            print(f"  n={len(vals)}  min={min(vals):.2f}  "
                  f"mediana={statistics.median(vals):.2f}  max={max(vals):.2f}")
            print("  (el paper reportaba mediana ~5.7mm, rango 1.4-20.8mm)")


if __name__ == "__main__":
    main()
