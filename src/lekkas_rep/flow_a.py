"""Flujo A: firma radiómica de 16 rasgos + gradient boosting [Lekkas et al., 2025]."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .radiomics_features import select_features


def make_gbm(cfg: dict, seed: int):
    fa = cfg["flow_a"]
    backend = fa.get("backend", "sklearn")
    if backend == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=fa["max_iter"], learning_rate=fa["learning_rate"], max_depth=fa["max_depth"],
                             reg_lambda=fa["l2_regularization"], subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=seed)
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=fa["max_iter"], learning_rate=fa["learning_rate"], max_depth=fa["max_depth"],
                              reg_lambda=fa["l2_regularization"], random_state=seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=fa["learning_rate"], max_iter=fa["max_iter"],
                                          max_depth=fa["max_depth"], l2_regularization=fa["l2_regularization"],
                                          min_samples_leaf=fa.get("min_samples_leaf", 20), random_state=seed)


def run_flow_a(feats: pd.DataFrame, manifest: pd.DataFrame, cfg: dict) -> dict:
    """Validación cruzada en desarrollo (selección anidada) + modelo final evaluado en la prueba externa.

    Devuelve predicciones fuera de fold (dev), predicciones externas y los rasgos finales.
    """
    seed = cfg["data"]["seed"]
    m = manifest[["PANORAMA_study_id", "split", "fold", "y"]].rename(columns={"PANORAMA_study_id": "sid"})
    m["sid"] = m["sid"].astype(str)
    df = m.merge(feats, on="sid", how="inner")
    feat_cols = [c for c in feats.columns if c != "sid"]
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    ext = df[df["split"] == "external"].reset_index(drop=True)

    oof = np.full(len(dev), np.nan)
    fold_feats = {}
    for k in sorted(dev["fold"].unique()):
        tr, va = dev["fold"] != k, dev["fold"] == k
        sel, _ = select_features(dev.loc[tr, feat_cols], dev.loc[tr, "y"].values, cfg, seed=seed + int(k))
        clf = make_gbm(cfg, seed).fit(dev.loc[tr, sel].values, dev.loc[tr, "y"].values)
        oof[va.values] = clf.predict_proba(dev.loc[va, sel].values)[:, 1]
        fold_feats[int(k)] = sel

    final_sel, sel_log = select_features(dev[feat_cols], dev["y"].values, cfg, seed=seed)
    clf = make_gbm(cfg, seed).fit(dev[final_sel].values, dev["y"].values)
    p_ext = clf.predict_proba(ext[final_sel].values)[:, 1] if len(ext) else np.array([])
    return {
        "dev": pd.DataFrame({"sid": dev["sid"], "y": dev["y"], "fold": dev["fold"], "prob": oof}),
        "external": pd.DataFrame({"sid": ext["sid"], "y": ext["y"], "prob": p_ext}),
        "selected_features": final_sel,
        "selection_log": sel_log,
        "fold_features": fold_feats,
        "model": clf,
    }
