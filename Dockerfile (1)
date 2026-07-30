# BraTS-PEDs 2026 Task 2 - NeuroSapiens
# Ensemble 5-fold (cascada skull-strip -> fine-tune) + WT combinado + CC logistica
#
# Base con CUDA 12.8 (<= 13.0, cumple requisito del challenge) y PyTorch 2.8

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# nnU-Net necesita estas variables aunque en inferencia no se usen todas
ENV nnUNet_raw=/opt/nnunet/raw
ENV nnUNet_preprocessed=/opt/nnunet/preprocessed
ENV nnUNet_results=/opt/models

# Dependencias del sistema minimas
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# nnU-Net v2.8.1 y dependencias (versiones fijadas, sin red en runtime)
RUN pip install --no-cache-dir \
        nnunetv2==2.8.1 \
        SimpleITK==2.5.5 \
        nibabel==5.4.2 \
        scipy==1.17.1 \
        blosc2==4.9.1

# --- Copiar los pesos del modelo DENTRO de la imagen ---
# Solo los checkpoint_final.pth de cada fold + los ficheros de config necesarios
# (dataset.json, plans.json). NO copiar checkpoints intermedios ni validation/
# para mantener la imagen pequena.
COPY models/ /opt/models/

# --- Trainers custom (si el modelo soft se incluye) ---
# El trainer nnUNetTrainer_250epochs es nativo de nnU-Net; no hace falta copiarlo.
# Si se incluye el modelo soft, copiar sus trainers:
COPY trainers/ /opt/custom_trainers/
RUN if [ -f /opt/custom_trainers/nnUNetTrainerSoftWT.py ]; then \
        NNUNET_DIR=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))") && \
        cp /opt/custom_trainers/*.py "$NNUNET_DIR/training/nnUNetTrainer/" ; \
    fi

# --- Script de inferencia ---
COPY predict.py /opt/predict.py

# Entry point: ejecuta la inferencia sobre /input -> /output
ENTRYPOINT ["python", "/opt/predict.py"]
