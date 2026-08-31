# BraTS-PEDs 2026 Task 2 - NeuroSapiens
# 5-fold Ensemble (skull-strip cascade -> fine-tune) + combined WT + CC logistic
# Base with CUDA 12.8 (<= 13.0, meets challenge requirements) and PyTorch 2.8

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# nnU-Net requires these variables even if not all are used during inference
ENV nnUNet_raw=/opt/nnunet/raw
ENV nnUNet_preprocessed=/opt/nnunet/preprocessed
ENV nnUNet_results=/opt/models

# Minimal system dependencies and cache cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# nnU-Net v2.8.1 and dependencies (pinned versions, offline runtime)
RUN pip install --no-cache-dir \
        nnunetv2==2.8.1 \
        SimpleITK==2.5.5 \
        nibabel==5.4.2 \
        scipy==1.17.1 \
        blosc2==4.9.1

# --- Copy model weights INSIDE the image ---
# Only the checkpoint_final.pth of each fold + necessary config files
# (dataset.json, plans.json). DO NOT copy intermediate checkpoints or validation/
# to keep the image size minimal.
COPY models/ /opt/models/

# --- Custom trainers (if soft model is included) ---
# The nnUNetTrainer_250epochs trainer is native to nnU-Net; no need to copy it.
# If the soft model is included, copy its trainers:
COPY trainers/ /opt/custom_trainers/
RUN if [ -f /opt/custom_trainers/nnUNetTrainerSoftWT.py ]; then \
        NNUNET_DIR=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))") && \
        cp /opt/custom_trainers/*.py "$NNUNET_DIR/training/nnUNetTrainer/" ; \
    fi

# --- Inference script ---
COPY predict.py /opt/predict.py

# Entry point: runs inference on /input -> /output
ENTRYPOINT ["python", "/opt/predict.py"]
