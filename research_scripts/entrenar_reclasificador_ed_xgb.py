#!/usr/bin/env python3
"""
entrenar_reclasificador_ed_xgb.py
=================================

Versión adaptada a XGBoost con hiperparámetros regularizados para evitar 
el sobreajuste en los bordes del edema peritumoral.

Requiere /workspace/features_ed.npz y las segmentaciones OOF en PRED_OOF_DIR.
"""

import os, numpy as np, nibabel as nib
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

DATA = "/workspace/features_ed.npz"
GT_DIR = "/workspace/nnUNet_data/nnUNet_raw/Dataset501_BraTSPED/labelsTr"
N_SPLITS = 5
THR_ED_SWEEP = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PRED_OOF_DIR = "/workspace/preds_oof_construidas"

def dice(a, b):
    sa, sb = a.sum(), b.sum()
    if sa == 0 and sb == 0: return np.nan
    if sa == 0 or sb == 0: return 0.0
    return 2 * np.logical_and(a, b).sum() / (sa + sb)

def main():
    d = np.load(DATA, allow_pickle=True)
    X, y, groups = d["X"], d["y"], d["groups"]
    feat_names = list(d["feat_names"])
    has_coords = "coords" in d and "case_ids" in d
    
    if has_coords:
        coords = d["coords"]
        case_ids = d["case_ids"]
        
    print(f"Vóxeles: {len(y):,} | features: {X.shape[1]} | grupos: {len(np.unique(groups))}")

    classes = np.array([0, 1, 2, 3, 4])
    counts = np.array([(y == c).sum() for c in classes])
    print("Distribución de clases:", dict(zip(classes.tolist(), counts.tolist())))

    # --- NUEVOS HIPERPARÁMETROS XGBOOST ---
    params = dict(
        objective="multi:softprob",
        num_class=len(classes),
        learning_rate=0.02,        # Más lento y seguro
        max_depth=5,               # Regularización de ramas (equivalente a bajar num_leaves)
        min_child_weight=500,      # Ignorar componentes conexas ruidosas muy pequeñas
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1500,         # Subimos margen por el learning_rate bajo
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss"
    )

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_proba = np.zeros((len(y), len(classes)), dtype=np.float32)
    best_iters = []

    for i, (tr, va) in enumerate(gkf.split(X, y, groups)):
        # Sub-split de tr para early stopping, SIN tocar va
        inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_in, es_in = next(inner.split(X[tr], y[tr], groups[tr]))
        X_tr, y_tr = X[tr][tr_in], y[tr][tr_in]
        X_es, y_es = X[tr][es_in], y[tr][es_in]

        # Calcular pesos de clase manualmente para XGBoost
        pesos_tr = compute_sample_weight(class_weight='balanced', y=y_tr)
        pesos_es = compute_sample_weight(class_weight='balanced', y=y_es)

        # 1. Añadimos el early_stopping_rounds directamente a los parámetros del modelo
        params_con_es = dict(params)
        params_con_es["early_stopping_rounds"] = 40
        
        model = xgb.XGBClassifier(**params_con_es)
        
        # 2. Entrenamos quitando el argumento early_stopping_rounds del fit()
        model.fit(
            X_tr, y_tr,
            sample_weight=pesos_tr,
            eval_set=[(X_es, y_es)],
            sample_weight_eval_set=[pesos_es],
            verbose=False
        )
        
        # Guardar la mejor iteración
        best_iter = model.best_iteration if model.best_iteration is not None else params["n_estimators"]
        best_iters.append(best_iter)
        
        # Predecir sobre el fold de validación OOF real
        oof_proba[va] = model.predict_proba(X[va])
        print(f"  Fold {i}: best_iter={best_iter}  ({len(tr):,} tr / {len(va):,} va)")

    print("\nImportancia de features (XGBoost):")
    importancias = model.feature_importances_
    for name, v in sorted(zip(feat_names, importancias), key=lambda t: -t[1]):
        print(f"  {name:16} {v:.5f}")

    # --- EFECTO NETO ---
    if not has_coords or not os.path.isdir(PRED_OOF_DIR):
        print("\n[AVISO] Falta coords/case_ids o PRED_OOF_DIR; se omite el efecto neto.")
    else:
        print("\n=== EFECTO NETO sobre las 5 regiones (OOF completo) ===")
        uniq_cases = np.unique(case_ids)
        base_seg, gt_seg = {}, {}
        for cid in uniq_cases:
            bp = os.path.join(PRED_OOF_DIR, f"{cid}.nii.gz")
            gp = os.path.join(GT_DIR, f"{cid}.nii.gz")
            if os.path.exists(bp) and os.path.exists(gp):
                base_seg[cid] = np.asanyarray(nib.load(bp).dataobj).astype(np.uint8)
                gt_seg[cid] = np.asanyarray(nib.load(gp).dataobj).astype(np.uint8)

        regions = {"WT": lambda s: s>=1, "TC": lambda s: np.isin(s,[1,2,3]),
                   "ET": lambda s: s==1, "NETC": lambda s: s==2,
                   "CC": lambda s: s==3, "ED": lambda s: s==4}

        def eval_all(seg_dict):
            out = {}
            for rn, rf in regions.items():
                ds = [dice(rf(seg_dict[c]), rf(gt_seg[c])) for c in seg_dict if c in gt_seg]
                out[rn] = np.nanmean(ds)
            return out

        base_metrics = eval_all(base_seg)
        print("BASE (sin remapeo):", {k: round(v,4) for k,v in base_metrics.items()})
        print(f"\n{'thr_ED':>7} " + " ".join(f"{r:>7}" for r in regions))
        
        for thr in THR_ED_SWEEP:
            keep_ed = oof_proba[:, 4] > thr
            new_label = np.where(keep_ed, 4, np.argmax(oof_proba[:, :4], axis=1))
            seg_mod = {c: base_seg[c].copy() for c in base_seg}
            for j in range(len(y)):
                cid = case_ids[j]
                if cid not in seg_mod: continue
                z, yy, xx = coords[j]
                seg_mod[cid][z, yy, xx] = new_label[j]
            m = eval_all(seg_mod)
            print(f"{thr:>7.2f} " + " ".join(f"{m[r]:>7.4f}" for r in regions))

    # --- MODELO FINAL ---
    n_final = int(np.mean(best_iters))
    print(f"\nModelo final: n_estimators = {n_final} (promedio de best_iter de los folds)")
    
    params_final = dict(params)
    params_final["n_estimators"] = n_final
    
    # Pesos balanceados para el dataset completo
    pesos_totales = compute_sample_weight(class_weight='balanced', y=y)
    
    final_model = xgb.XGBClassifier(**params_final)
    final_model.fit(X, y, sample_weight=pesos_totales)
    
    import joblib
    joblib.dump({"model": final_model, "feat_names": feat_names, "n_estimators": n_final},
                "/workspace/reclasificador_ed_xgb.joblib")
    print("Guardado: /workspace/reclasificador_ed_xgb.joblib")

if __name__ == "__main__":
    main()