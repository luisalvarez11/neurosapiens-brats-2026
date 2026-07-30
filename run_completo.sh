#!/bin/bash
set -euo pipefail

export nnUNet_raw="/workspace/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_data/nnUNet_results"
export nnUNet_n_proc_DA=12 

echo "=== PASO 1: Preparando formato para el stripper ==="
mkdir -p /workspace/stripper_input
python /workspace/rename_bratspeds.py to_stripper /workspace/Dataset501 /workspace/stripper_input

echo "=== PASO 2: Instalando y ejecutando HD-BET (Nativo) ==="
# Instalamos HD-BET
pip install git+https://github.com/MIC-DKFZ/HD-BET
mkdir -p /workspace/stripper_output

# Limpiamos los cerebros usando la tarjeta gráfica
hd-bet -i /workspace/stripper_input -o /workspace/stripper_output -device cuda:0

echo "=== PASO 3: Reconstruyendo Dataset 502 directo a nnU-Net ==="
mkdir -p $nnUNet_raw/Dataset502_BraTSPED/imagesTr
mkdir -p $nnUNet_raw/Dataset502_BraTSPED/labelsTr

python /workspace/rename_bratspeds.py from_stripper /workspace/stripper_output $nnUNet_raw/Dataset502_BraTSPED/imagesTr
cp /workspace/Dataset501/*/*-seg.nii.gz $nnUNet_raw/Dataset502_BraTSPED/labelsTr/
cp /workspace/Dataset501/dataset.json $nnUNet_raw/Dataset502_BraTSPED/ || echo "Aviso: No se encontró dataset.json"

echo "=== PASO 4: Verificando la integridad del Dataset 502 ==="
python /workspace/verify_dataset502.py $nnUNet_raw/Dataset502_BraTSPED

echo "=== PASO 5: Preprocesamiento de nnU-Net ==="
pip install -e /opt/nnUNet_source >/dev/null 2>&1 || true
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity

echo "=== PASO 5.5: Generando splits por paciente ==="
python /workspace/make_patient_splits.py --dataset_dir $nnUNet_preprocessed/Dataset502_BraTSPED --n_folds 5 --seed 42

echo "=== PASO 6: Entrenamiento Final ==="
nnUNetv2_train 502 3d_fullres 0 --npz