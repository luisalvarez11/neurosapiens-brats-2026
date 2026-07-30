#!/bin/bash
# preparar_modelos.sh
# ===================
# Copia SOLO los ficheros necesarios de los modelos a la carpeta ./models/
# (checkpoint_final.pth de cada fold + dataset.json + plans.json), para que
# la imagen Docker sea pequena. Ejecutar en el pod ANTES del docker build.
#
# Uso: bash preparar_modelos.sh

set -e

RESULTS="/workspace/nnUNet_data/nnUNet_results"
DEST="./models"

# --- Modelo ENSEMBLE hard (Dataset 501, 5 folds) ---
SRC_HARD="$RESULTS/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
DST_HARD="$DEST/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"

echo "=== Copiando modelo ensemble (5 folds) ==="
mkdir -p "$DST_HARD"
# ficheros de config (a nivel del trainer)
for f in dataset.json plans.json dataset_fingerprint.json; do
    [ -f "$SRC_HARD/$f" ] && cp "$SRC_HARD/$f" "$DST_HARD/"
done
# checkpoint_final.pth de cada fold (SOLO ese, no los intermedios)
for FOLD in 0 1 2 3 4; do
    mkdir -p "$DST_HARD/fold_$FOLD"
    cp "$SRC_HARD/fold_$FOLD/checkpoint_final.pth" "$DST_HARD/fold_$FOLD/"
    echo "  fold $FOLD copiado"
done

# --- Modelo SOFT (Dataset 503, fold 0) - OPCIONAL ---
SRC_SOFT="$RESULTS/Dataset503_BraTSPED/nnUNetTrainerSoftWT_lowLR__nnUNetPlans__3d_fullres"
DST_SOFT="$DEST/Dataset503_BraTSPED/nnUNetTrainerSoftWT_lowLR__nnUNetPlans__3d_fullres"
if [ -d "$SRC_SOFT" ]; then
    echo "=== Copiando modelo soft (fold 0) ==="
    mkdir -p "$DST_SOFT/fold_0"
    for f in dataset.json plans.json dataset_fingerprint.json; do
        [ -f "$SRC_SOFT/$f" ] && cp "$SRC_SOFT/$f" "$DST_SOFT/"
    done
    cp "$SRC_SOFT/fold_0/checkpoint_final.pth" "$DST_SOFT/fold_0/"
    echo "  soft fold 0 copiado"

    # IMPORTANTE: el dataset.json del soft debe estar en 4 canales para inferencia
    # (el trainer separa el 5o canal, pero el predictor lee el dataset.json).
    # Si el guardado tiene 5 canales, ajustarlo:
    python3 -c "
import json, os
p='$DST_SOFT/dataset.json'
d=json.load(open(p))
if '4' in d.get('channel_names',{}):
    d['channel_names']={k:v for k,v in d['channel_names'].items() if k!='4'}
    json.dump(d, open(p,'w'), indent=4)
    print('  dataset.json del soft ajustado a 4 canales')
"
else
    echo "=== Modelo soft no encontrado; el Docker usara solo el WT hard ==="
fi

# --- Trainers custom (para el soft) ---
mkdir -p ./trainers
NNUNET_DIR=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
for t in nnUNetTrainerSoftWT.py nnUNetTrainerSoftWT_lowLR.py; do
    if [ -f "$NNUNET_DIR/training/nnUNetTrainer/$t" ]; then
        cp "$NNUNET_DIR/training/nnUNetTrainer/$t" ./trainers/
        echo "  trainer $t copiado"
    fi
done
# placeholder para que COPY trainers/ no falle si no hay ninguno
touch ./trainers/.keep

echo ""
echo "=== Listo. Tamano de ./models: ==="
du -sh "$DEST"
echo "Ahora: docker build -t bratsped-neurosapiens:latest ."
