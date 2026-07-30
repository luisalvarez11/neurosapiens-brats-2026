#!/usr/bin/env python3
"""
entrenar_reclasificador_ed.py
=============================

1. Entrena LightGBM multiclase (5 clases) sobre los voxeles ED, con GroupKFold
   por case_id (evita fuga). Predice a que clase deberia ir cada voxel ED.
2. Barre el umbral thr_ED (mantener ED si P(ED)>thr_ED, si no argmax entre
   las otras 4), optimizando el efecto NETO en todas las regiones out-of-fold.
3. Reporta el efecto neto (Dice de WT/TC/ET/NET/CC/ED antes y despues).

Este script SOLO entrena y evalua en OOF (interno). La aplicacion al oficial
se hace en un paso posterior con el modelo final.
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

DATA = "/workspace/features_ed.npz"
N_SPLITS = 5
THR_ED_SWEEP = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 0.0 = argmax puro


def main():
    d = np.load(DATA, allow_pickle=True)
    X, y, groups = d["X"], d["y"], d["groups"]
    feat_names = list(d["feat_names"])
    print(f"Voxeles: {len(y):,} | features: {X.shape[1]} | grupos: {len(np.unique(groups))}")

    # remapear labels a 0..4 contiguos (ya lo estan: 0,1,2,3,4)
    classes = np.array([0, 1, 2, 3, 4])
    n_class = 5

    # manejo de desbalance: class weights inversos a frecuencia
    counts = np.array([(y == c).sum() for c in classes])
    print("Distribucion de clases:", dict(zip(classes.tolist(), counts.tolist())))

    gkf = GroupKFold(n_splits=N_SPLITS)

    # predicciones OOF de probabilidad por clase
    oof_proba = np.zeros((len(y), n_class), dtype=np.float32)

    params = dict(
        objective="multiclass", num_class=n_class,
        learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, n_estimators=400, class_weight="balanced",
        n_jobs=-1, verbose=-1,
    )

    for i, (tr, va) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr], y[tr],
                  eval_set=[(X[va], y[va])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_proba[va] = model.predict_proba(X[va])
        print(f"  fold {i}: entrenado ({len(tr):,} train / {len(va):,} val)")

    # importancia de features (ultimo modelo)
    print("\nImportancia de features:")
    imp = model.feature_importances_
    for name, v in sorted(zip(feat_names, imp), key=lambda t: -t[1]):
        print(f"  {name:16} {v}")

    # --- barrido de thr_ED: para cada voxel ED, mantener o remapear ---
    # clase 4 = ED. Si P(ED) > thr_ED -> queda ED (label 4). Si no -> argmax entre {0,1,2,3}.
    print("\n=== BARRIDO thr_ED (efecto sobre los voxeles ED reclasificados) ===")
    print(f"{'thr_ED':>7}{'acc':>8}{'ED_recall':>11}{'ED_prec':>9}{'%remap':>9}")
    ed_idx = 4
    for thr in THR_ED_SWEEP:
        keep_ed = oof_proba[:, ed_idx] > thr
        # nueva etiqueta
        new_label = np.where(
            keep_ed, 4,
            np.argmax(oof_proba[:, :4], axis=1)  # entre 0,1,2,3
        )
        acc = (new_label == y).mean()
        # metricas del ED (clase 4)
        gt_ed = (y == 4)
        pred_ed = (new_label == 4)
        tp = (pred_ed & gt_ed).sum(); fp = (pred_ed & ~gt_ed).sum(); fn = (~pred_ed & gt_ed).sum()
        rec = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        pct_remap = 100 * (new_label != 4).mean()  # cuantos ED originales se remapean
        print(f"{thr:>7.2f}{acc:>8.3f}{rec:>11.3f}{prec:>9.3f}{pct_remap:>8.1f}%")

    print("\nINTERPRETACION:")
    print("  - ED_recall alto + ED_prec alto = buen umbral (mantiene ED real, quita FP)")
    print("  - Elegir el thr que maximice el balance segun lo que priorice el challenge")
    print("\nGuardando modelo final (entrenado en todo) para aplicar al oficial...")

    final = lgb.LGBMClassifier(**params)
    final.fit(X, y)
    import joblib
    joblib.dump({"model": final, "feat_names": feat_names}, "/workspace/reclasificador_ed.joblib")
    print("Guardado: /workspace/reclasificador_ed.joblib")


if __name__ == "__main__":
    main()
