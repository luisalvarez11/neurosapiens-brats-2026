#!/usr/bin/env python3
"""
entrenar_reclasificador_edema.py
=================================

Entrena un clasificador que, para cada vóxel HOY etiquetado como Edema
(ED = WT \\ TC del pipeline de ensemble), predice su clase real:
    0 = Fondo sano   1 = ET   2 = NETC   3 = Quiste (CC)   4 = Edema real

Usa los 294 casos de training con ground truth conocido.

REQUISITO CLAVE: TRAIN_PROBS debe contener probabilidades *out-of-fold*
(la predicción de cada caso hecha por el fold de nnU-Net que NO lo vio
en entrenamiento). Si usas las probabilidades "in-fold" el reclasificador
va a aprender a corregir errores que en producción (validation oficial)
no va a tener forma de detectar -- exactamente el mismo motivo por el
que ya haces ensemble de 5-fold para la submission final.

Ajusta las rutas en CONFIG antes de correr.
"""

import os, glob, json
import numpy as np
import nibabel as nib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from brats_postproc_common import (
    zscore_brain, construir_wt_tc_et_edema, features_voxeles_candidatos, FEATURE_NAMES
)

# ------------------------------- CONFIG (ajusta) -------------------------------
TRAIN_IMAGES     = "/workspace/imagesTr"                    # 4 modalidades _0000.._0003
TRAIN_LABELS     = "/workspace/labelsTr"                    # {cid}.nii.gz, ET=1 NETC=2 CC=3 ED=4
TRAIN_PROBS      = "/workspace/preds_ensemble_train"        # {cid}.npz out-of-fold, 5-fold
TRAIN_PROBS_SOFT = "/workspace/preds_soft_train"            # opcional; pon None si no aplica

TC_SOFT_CH, THR_SOFT, MIN_COMPONENT = 1, 0.5, 50

MODEL_OUT  = "/workspace/modelo_reclasificador_edema.joblib"
REPORT_OUT = "/workspace/reporte_reclasificador_edema.json"
N_SPLITS_CV = 5
# --------------------------------------------------------------------------------


def cargar_caso(cid):
    ref_p = os.path.join(TRAIN_IMAGES, f"{cid}_0000.nii.gz")
    ref_shape = nib.load(ref_p).shape

    nombres = ["t1", "t1c", "t2", "fl"]  # 0000,0001,0002,0003 (mismo orden que tu script)
    imgs = {}
    for ch, nombre in enumerate(nombres):
        p = os.path.join(TRAIN_IMAGES, f"{cid}_000{ch}.nii.gz")
        vol = np.asanyarray(nib.load(p).dataobj).astype(np.float32)
        imgs[nombre] = zscore_brain(vol)

    lab_p = os.path.join(TRAIN_LABELS, f"{cid}.nii.gz")
    gt = np.asanyarray(nib.load(lab_p).dataobj).astype(np.uint8)

    prob_ens = np.load(os.path.join(TRAIN_PROBS, f"{cid}.npz"))["probabilities"]

    prob_soft = None
    if TRAIN_PROBS_SOFT:
        sp = os.path.join(TRAIN_PROBS_SOFT, f"{cid}.npz")
        if os.path.exists(sp):
            prob_soft = np.load(sp)["probabilities"]

    return ref_shape, imgs, gt, prob_ens, prob_soft


def construir_dataset():
    cids = sorted(
        os.path.basename(f).replace(".npz", "")
        for f in glob.glob(os.path.join(TRAIN_PROBS, "*.npz"))
    )
    print(f"Casos de entrenamiento encontrados: {len(cids)}")

    X_all, y_all, g_all = [], [], []
    for n, cid in enumerate(cids, 1):
        try:
            ref_shape, imgs, gt, prob_ens, prob_soft = cargar_caso(cid)
        except FileNotFoundError as e:
            print(f"  [!] {cid}: falta un archivo ({e}); se salta")
            continue

        wt, tc, et, edema, chans = construir_wt_tc_et_edema(
            prob_ens, ref_shape, prob_soft, TC_SOFT_CH, THR_SOFT, MIN_COMPONENT
        )

        idx = np.where(edema)
        if idx[0].size == 0:
            continue

        feats = features_voxeles_candidatos(
            idx, chans, imgs["t1"], imgs["t1c"], imgs["t2"], imgs["fl"], tc, wt
        )
        labels = gt[idx]

        X_all.append(feats)
        y_all.append(labels)
        g_all.append(np.full(labels.shape, n, dtype=np.int32))

        if n % 20 == 0:
            total = sum(a.shape[0] for a in X_all)
            print(f"  {n}/{len(cids)} casos, {total:,} vóxeles candidatos hasta ahora")

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    g = np.concatenate(g_all)

    print(f"\nDataset final: {X.shape[0]:,} vóxeles, {X.shape[1]} features, "
          f"{len(np.unique(g))} casos")
    print("Distribución de clases (0=fondo,1=ET,2=NETC,3=CC,4=ED-real):")
    for c in sorted(np.unique(y)):
        print(f"  {c}: {(y == c).sum():,} ({(y == c).mean()*100:.1f}%)")
    return X, y, g


def evaluar_cv(X, y, g, n_splits=N_SPLITS_CV):
    """Validación cruzada partida POR CASO (GroupKFold), no por vóxel."""
    gkf = GroupKFold(n_splits=n_splits)
    modelos = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, multi_class="multinomial"),
        ),
        "hgb": HistGradientBoostingClassifier(max_iter=300, random_state=0),
    }
    resultados = {}
    nombres_clases = ["fondo", "ET", "NETC", "CC", "ED"]
    for nombre, modelo in modelos.items():
        print(f"\n=== Validación cruzada por caso — {nombre} ===")
        y_pred_oof = np.zeros_like(y)
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups=g)):
            modelo.fit(X[tr], y[tr])
            y_pred_oof[va] = modelo.predict(X[va])
            print(f"  fold {fold + 1}/{n_splits} listo")
        etiquetas_presentes = sorted(np.unique(y))
        print(classification_report(
            y, y_pred_oof, labels=etiquetas_presentes, digits=3,
            target_names=[nombres_clases[c] for c in etiquetas_presentes]))
        print("Matriz de confusión (filas=real, columnas=predicho):")
        print(confusion_matrix(y, y_pred_oof, labels=etiquetas_presentes))
        resultados[nombre] = classification_report(
            y, y_pred_oof, labels=etiquetas_presentes, digits=3,
            target_names=[nombres_clases[c] for c in etiquetas_presentes],
            output_dict=True)
    return resultados


def main():
    X, y, g = construir_dataset()

    resultados = evaluar_cv(X, y, g)
    with open(REPORT_OUT, "w") as f:
        json.dump(resultados, f, indent=2)

    # Modelo final entrenado con todos los datos.
    # Cambia a la pipeline "logreg" si prefieres algo más interpretable/conservador
    # una vez compares los reportes de evaluar_cv() en el JSON.
    modelo_final = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    modelo_final.fit(X, y)
    joblib.dump({"modelo": modelo_final, "features": FEATURE_NAMES}, MODEL_OUT)

    print(f"\nModelo guardado en: {MODEL_OUT}")
    print(f"Reporte de validación (JSON) guardado en: {REPORT_OUT}")
    print("\nRevisa sobre todo: precision de la clase 'ED' en el reporte.")
    print("Si baja mucho respecto al recall de NETC/CC, sube UMBRAL_FLIP")
    print("en construir_submission_final_ensemble_v2.py para ser más conservador.")


if __name__ == "__main__":
    main()
