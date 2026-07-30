#!/usr/bin/env python3
"""
predict.py - Inferencia BraTS-PEDs 2026 Task 2 (NeuroSapiens)
=============================================================

Pipeline (replica la submission 9774214):
  1. Lee cada carpeta de caso en /input (4 modalidades: t1n, t1c, t2w, t2f).
  2. Las convierte al formato nnU-Net (_0000.._0003) en un tmp.
  3. Inferencia con el ENSEMBLE 5-fold (fine-tuning cascada), con probabilidades.
  4. Reconstruye regiones (WT, TC, ET) a 0.5.
  5. WT combinado: (WT_hard) OR (TC_soft > 0.5)  [si hay modelo soft disponible]
  6. Remapeo a etiquetas oficiales: ET=1, NET=2, CC=3, ED=4.
  7. CC: logistica de intensidad thr 0.90 sobre el NET.
  8. Escribe cada prediccion en /output (estructura plana).

Requisitos del challenge:
  - /input read-only (nunca se escribe ahi)
  - /output plano (sin subcarpetas)
  - network none (todo autocontenido)
"""

import os
import sys
import glob
import shutil
import tempfile
import numpy as np
import nibabel as nib
from scipy import ndimage

# ---------------------------------------------------------------------------
INPUT_DIR = "/input"
OUTPUT_DIR = "/output"

# Modelo ensemble (dentro de la imagen)
MODEL_DIR = "/opt/models/Dataset501_BraTSPED/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
FOLDS = (0, 1, 2, 3, 4)
TRAINER = "nnUNetTrainer_250epochs"

# Modelo soft (opcional; si no esta, se usa solo el WT hard)
SOFT_MODEL_DIR = "/opt/models/Dataset503_BraTSPED/nnUNetTrainerSoftWT_lowLR__nnUNetPlans__3d_fullres"
USE_SOFT = os.path.isdir(SOFT_MODEL_DIR)

# Parametros (identicos a 9774214)
THR_SOFT = 0.5
COEF_T1C, COEF_T2W, INTERCEPT = -0.5844, 1.0382, -2.2613
THR_CC, MIN_CC = 0.90, 100
MIN_COMPONENT = 50

# Mapeo modalidad -> canal nnU-Net
MODALITY_TO_CHANNEL = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}


def log(msg):
    print(f"[predict] {msg}", flush=True)


def zscore_brain(vol):
    m = vol > 0
    if m.sum() == 0:
        return np.zeros_like(vol, dtype=np.float32)
    mu, s = vol[m].mean(), vol[m].std()
    z = np.zeros_like(vol, dtype=np.float32)
    z[m] = (vol[m] - mu) / (s + 1e-6)
    return z


def filtrar(mask, minv):
    if mask.sum() == 0:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (sizes > minv)[lab]


def find_case_folders(input_dir):
    """Devuelve las carpetas de caso dentro de /input."""
    entries = []
    for name in sorted(os.listdir(input_dir)):
        full = os.path.join(input_dir, name)
        if os.path.isdir(full):
            entries.append((name, full))
    return entries


def stage_case(case_id, case_dir, tmp_in):
    """Copia las 4 modalidades al formato nnU-Net en tmp_in. Devuelve el path de referencia."""
    ref_path = None
    for mod, chan in MODALITY_TO_CHANNEL.items():
        # buscar el fichero de esa modalidad (flexible con el naming)
        candidates = glob.glob(os.path.join(case_dir, f"*{mod}.nii.gz"))
        if not candidates:
            log(f"  [!] {case_id}: falta modalidad {mod}")
            return None
        src = candidates[0]
        dst = os.path.join(tmp_in, f"{case_id}_{chan}.nii.gz")
        shutil.copy2(src, dst)
        if mod == "t1n":
            ref_path = dst
    return ref_path


def find_any_modality_file(case_dir):
    """Busca cualquier modalidad presente en el caso, para usar como referencia de forma/affine
    cuando el caso no se puede procesar normalmente (falta alguna modalidad u otro fallo)."""
    for mod in MODALITY_TO_CHANNEL:
        candidates = glob.glob(os.path.join(case_dir, f"*{mod}.nii.gz"))
        if candidates:
            return candidates[0]
    return None


def write_empty_output(case_id, case_dir):
    """Escribe una segmentacion vacia (todo ceros) para este caso, para que SIEMPRE haya un
    fichero por caso en /output (un caso fallido no debe invalidar el batch completo).
    Devuelve True si se pudo escribir, False si no habia ninguna imagen de referencia disponible."""
    ref_file = find_any_modality_file(case_dir)
    if ref_file is None:
        log(f"  [ERROR] {case_id}: no hay NINGUNA modalidad disponible, no se puede ni generar un output vacio")
        return False
    ref_img = nib.load(ref_file)
    out = np.zeros(ref_img.shape, dtype=np.uint8)
    out_path = os.path.join(OUTPUT_DIR, f"{case_id}.nii.gz")
    nib.save(nib.Nifti1Image(out, ref_img.affine, ref_img.header), out_path)
    log(f"  -> {out_path}  [VACIO: caso no procesado, puntuara 0 pero no invalida el batch]")
    return True


def regiones_desde_probs(prob, ref_shape):
    """Del ensemble region-based (canales [WT, TC, ET]), reconstruye mascaras a 0.5."""
    chans = []
    for i in range(prob.shape[0]):
        c = prob[i]
        if c.shape != ref_shape:
            c = np.transpose(c, (2, 1, 0))
        chans.append(c)
    wt = chans[0] > 0.5
    tc = chans[1] > 0.5
    et = chans[2] > 0.5
    tc = tc & wt
    et = et & tc
    return wt, tc, et


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cases = find_case_folders(INPUT_DIR)
    log(f"Casos encontrados: {len(cases)}")
    if not cases:
        log("No hay casos en /input. Abortando.")
        sys.exit(1)

    # importaciones pesadas aqui (tras confirmar que hay trabajo)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch
    import nnunetv2

    # --- log de entorno (reproducibilidad) ---
    log(f"torch: {torch.__version__}")
    log(f"cuda disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"nnunetv2: {nnunetv2.__version__}")
    log(f"USE_SOFT: {USE_SOFT}")

    # --- predictor ENSEMBLE hard ---
    log("Inicializando predictor ensemble (5-fold)...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device("cuda", 0),
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        MODEL_DIR, use_folds=FOLDS, checkpoint_name="checkpoint_final.pth"
    )

    # --- predictor SOFT (opcional) ---
    soft_predictor = None
    if USE_SOFT:
        log("Inicializando predictor soft (fold 0)...")
        soft_predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            device=torch.device("cuda", 0), verbose=False, allow_tqdm=False,
        )
        soft_predictor.initialize_from_trained_model_folder(
            SOFT_MODEL_DIR, use_folds=(0,), checkpoint_name="checkpoint_final.pth"
        )

    for case_id, case_dir in cases:
        log(f"Procesando {case_id}")
        tmp_in = tempfile.mkdtemp(prefix="in_")
        tmp_out = tempfile.mkdtemp(prefix="out_")
        try:
            ref_path = stage_case(case_id, case_dir, tmp_in)
            if ref_path is None:
                write_empty_output(case_id, case_dir)
                continue
            ref = nib.load(ref_path)
            ref_shape = ref.shape

            # --- inferencia ensemble con probabilidades ---
            predictor.predict_from_files(
                tmp_in, tmp_out,
                save_probabilities=True,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
            )
            npz = os.path.join(tmp_out, f"{case_id}.npz")
            prob = np.load(npz)["probabilities"]
            wt_hard, tc_hard, et_hard = regiones_desde_probs(prob, ref_shape)

            # --- soft TC (opcional) ---
            soft_add = np.zeros(ref_shape, dtype=bool)
            if soft_predictor is not None:
                tmp_out_s = tempfile.mkdtemp(prefix="outs_")
                try:
                    soft_predictor.predict_from_files(
                        tmp_in, tmp_out_s, save_probabilities=True, overwrite=True,
                        num_processes_preprocessing=2, num_processes_segmentation_export=2,
                    )
                    npz_s = os.path.join(tmp_out_s, f"{case_id}.npz")
                    tc_soft = np.load(npz_s)["probabilities"][1]
                    if tc_soft.shape != ref_shape:
                        tc_soft = np.transpose(tc_soft, (2, 1, 0))
                    soft_add = tc_soft > THR_SOFT
                finally:
                    shutil.rmtree(tmp_out_s, ignore_errors=True)

            # --- WT combinado ---
            wt = wt_hard | soft_add
            wt = filtrar(wt, MIN_COMPONENT)
            tc = tc_hard & wt
            et = et_hard & tc
            edema = wt & ~tc
            net = tc & ~et

            # --- CC logistica sobre el NET ---
            cc = np.zeros(ref_shape, dtype=bool)
            t1c_c = glob.glob(os.path.join(case_dir, "*t1c.nii.gz"))
            t2w_c = glob.glob(os.path.join(case_dir, "*t2w.nii.gz"))
            if t1c_c and t2w_c and net.sum() > 0:
                t1c_z = zscore_brain(np.asanyarray(nib.load(t1c_c[0]).dataobj).astype(np.float32))
                t2w_z = zscore_brain(np.asanyarray(nib.load(t2w_c[0]).dataobj).astype(np.float32))
                logit = COEF_T1C * t1c_z + COEF_T2W * t2w_z + INTERCEPT
                prob_cc = 1.0 / (1.0 + np.exp(-logit))
                cc = filtrar(net & (prob_cc > THR_CC), MIN_CC)

            # --- ensamblar etiquetas: ET=1, NET=2, CC=3, ED=4 ---
            out = np.zeros(ref_shape, dtype=np.uint8)
            out[edema] = 4
            out[net] = 2
            out[cc] = 3
            out[et] = 1

            # --- escribir en /output (plano) ---
            out_path = os.path.join(OUTPUT_DIR, f"{case_id}.nii.gz")
            nib.save(nib.Nifti1Image(out, ref.affine, ref.header), out_path)
            log(f"  -> {out_path}  (WT={int(wt.sum())} vox)")

        except Exception as e:
            log(f"  [ERROR] {case_id}: {e}")
            import traceback
            traceback.print_exc()
            # no dejar el batch sin este caso: escribir un output vacio para no invalidar el resto
            write_empty_output(case_id, case_dir)
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)

    # --- validacion final: un .nii.gz por caso, o abortar con error ---
    esperados = len(cases)
    escritos = len(glob.glob(os.path.join(OUTPUT_DIR, "*.nii.gz")))
    log(f"Casos esperados: {esperados} | ficheros escritos en /output: {escritos}")
    if escritos != esperados:
        log(f"[ERROR] Faltan salidas: se esperaban {esperados} y se escribieron {escritos}.")
        sys.exit(1)

    log("Inferencia completada.")


if __name__ == "__main__":
    main()
