#!/usr/bin/env python3
"""
make_patient_splits.py
======================

Genera un splits_final.json para nnU-Net v2 agrupando por PACIENTE, no por
caso, para evitar fuga de datos (data leakage) entre train y validacion.

EL PROBLEMA
-----------
BraTS-PEDs 2026 mezcla dos cohortes:

  Batch 1 (treatment-naive):
      BraTS-PED-00001-000      <- un timepoint por paciente

  Batch 2 (post-treatment, longitudinal):
      BraTS-PED-00025-100      <- MISMO paciente (00025),
      BraTS-PED-00025-101         distintos timepoints
      BraTS-PED-00025-102
      BraTS-PED-00025-103

Si nnU-Net hace el split 5-fold por CASO (que es lo que hace por defecto),
BraTS-PED-00025-100 puede caer en train y BraTS-PED-00025-101 en validacion.
Es el mismo cerebro del mismo nino unas semanas despues: el modelo ya ha
visto ese tumor. La metrica de validacion sale inflada y NO refleja la
capacidad de generalizar a un paciente nuevo.

LA SOLUCION
-----------
Agrupar por ID de paciente (los primeros campos, ignorando el sufijo de
timepoint) y repartir PACIENTES entre folds, no casos. Todos los timepoints
de un paciente van juntos al mismo lado.

Uso:
    python make_patient_splits.py \
        --dataset_dir $nnUNet_preprocessed/Dataset501_BraTSPED \
        --n_folds 5 \
        --seed 42

Escribe splits_final.json en dataset_dir. nnU-Net lo detecta y lo usa en
lugar de generar su propio split.

IMPORTANTE: ejecutar DESPUES de nnUNetv2_plan_and_preprocess (la carpeta de
preprocessed debe existir) y ANTES de nnUNetv2_train.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def patient_id_from_case(case_id):
    """
    Extrae el ID de paciente de un case_id de BraTS-PEDs.

        BraTS-PED-00001-000  ->  BraTS-PED-00001
        BraTS-PED-00025-101  ->  BraTS-PED-00025

    El ultimo campo (-000, -100, -101...) es el timepoint y se descarta.
    """
    m = re.match(r"^(.*)-(\d+)$", case_id)
    if m:
        return m.group(1)
    # Si el patron no encaja, tratar el caso como su propio paciente
    # (mas seguro que agrupar mal).
    print(f"  [aviso] patron inesperado, se trata como paciente unico: {case_id}")
    return case_id


def collect_cases(dataset_dir):
    """
    Lista los case_id del dataset preprocesado.

    nnU-Net guarda un .npz por caso en la carpeta de datos preprocesados.
    Buscamos ahi. Si no, caemos al gt_segmentations.
    """
    # Los datos preprocesados viven en subcarpetas tipo nnUNetPlans_3d_fullres
    candidates = list(dataset_dir.glob("nnUNetPlans_*/*.npz"))
    if candidates:
        return sorted({f.stem for f in candidates})

    # Fallback: la carpeta gt_segmentations siempre existe tras preprocesar
    gt_dir = dataset_dir / "gt_segmentations"
    if gt_dir.exists():
        return sorted({
            f.name.replace(".nii.gz", "")
            for f in gt_dir.glob("*.nii.gz")
        })

    return []


def make_splits(cases, n_folds, seed):
    """
    Reparte PACIENTES en n_folds, y expande a casos.

    Devuelve la lista de dicts {'train': [...], 'val': [...]} que espera
    nnU-Net en splits_final.json.
    """
    # Agrupar casos por paciente
    by_patient = defaultdict(list)
    for case in cases:
        by_patient[patient_id_from_case(case)].append(case)

    patients = sorted(by_patient.keys())

    print(f"\nCasos totales:     {len(cases)}")
    print(f"Pacientes unicos:  {len(patients)}")

    # Cuantos pacientes tienen mas de un timepoint (los peligrosos)
    multi = {p: c for p, c in by_patient.items() if len(c) > 1}
    if multi:
        print(f"Pacientes con multiples timepoints: {len(multi)}")
        print("  (estos son los que causarian fuga de datos con el split por defecto)")
        for p, c in sorted(multi.items())[:5]:
            print(f"    {p}: {len(c)} timepoints")
        if len(multi) > 5:
            print(f"    ... y {len(multi) - 5} mas")
    else:
        print("No hay pacientes con multiples timepoints.")
        print("  (el split por defecto de nnU-Net habria sido seguro, pero este")
        print("   script no hace dano)")

    # Barajar pacientes de forma reproducible
    rng = np.random.RandomState(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)

    # Repartir pacientes en n_folds de forma equilibrada
    fold_patients = [[] for _ in range(n_folds)]
    for i, patient in enumerate(shuffled):
        fold_patients[i % n_folds].append(patient)

    # Construir los splits: para cada fold, val = ese fold, train = el resto
    splits = []
    for k in range(n_folds):
        val_patients = set(fold_patients[k])
        train_patients = set(patients) - val_patients

        val_cases = sorted(
            c for p in val_patients for c in by_patient[p]
        )
        train_cases = sorted(
            c for p in train_patients for c in by_patient[p]
        )

        splits.append({"train": train_cases, "val": val_cases})

    return splits, by_patient


def verify_no_leakage(splits, by_patient):
    """Comprueba que ningun paciente aparece a ambos lados de un split."""
    case_to_patient = {
        case: patient
        for patient, cases in by_patient.items()
        for case in cases
    }

    ok = True
    for k, split in enumerate(splits):
        train_patients = {case_to_patient[c] for c in split["train"]}
        val_patients = {case_to_patient[c] for c in split["val"]}
        overlap = train_patients & val_patients
        if overlap:
            ok = False
            print(f"\n[ERROR] Fold {k}: FUGA DE DATOS. Pacientes en ambos lados:")
            for p in sorted(overlap):
                print(f"    {p}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True,
                    help="carpeta del dataset en nnUNet_preprocessed")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42,
                    help="semilla para reproducibilidad")
    ap.add_argument("--dry_run", action="store_true",
                    help="no escribe el fichero, solo muestra el reparto")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        sys.exit(f"Error: no existe {dataset_dir}\n"
                 f"Ejecuta primero nnUNetv2_plan_and_preprocess.")

    cases = collect_cases(dataset_dir)
    if not cases:
        sys.exit(f"Error: no se encontraron casos en {dataset_dir}\n"
                 f"Ejecuta primero nnUNetv2_plan_and_preprocess.")

    splits, by_patient = make_splits(cases, args.n_folds, args.seed)

    print(f"\nReparto en {args.n_folds} folds:")
    for k, split in enumerate(splits):
        print(f"  Fold {k}: {len(split['train']):>4} train, "
              f"{len(split['val']):>3} val")

    # Verificacion critica
    if not verify_no_leakage(splits, by_patient):
        sys.exit("\nABORTADO: el split tiene fuga de datos. Esto es un bug.")

    print("\n[OK] Verificado: ningun paciente aparece en train y val a la vez.")

    if args.dry_run:
        print("\n(dry run: no se ha escrito nada)")
        return

    out_path = dataset_dir / "splits_final.json"

    if out_path.exists():
        print(f"\n[aviso] {out_path} ya existe y se va a SOBRESCRIBIR.")
        print("        (si nnU-Net ya entreno con el split viejo, los")
        print("         resultados no seran comparables)")

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nEscrito: {out_path}")
    print("\nnnU-Net usara este split en lugar de generar el suyo.")
    print("Lanza el entrenamiento normalmente:")
    print("  nnUNetv2_train 501 3d_fullres 0 --npz")


if __name__ == "__main__":
    main()
