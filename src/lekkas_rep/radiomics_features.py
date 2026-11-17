"""Extracción radiómica (OE2) y selección en cascada de 16 rasgos estables.

Selección [Lekkas et al., 2025]: varianza → ANOVA → poda por correlación → estabilidad con bootstrap.
La selección se ajusta SOLO con datos de desarrollo; dentro de la validación cruzada se repite en cada fold
para no filtrar información de la partición de validación.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

log = logging.getLogger("lekkas_rep")


def extract_case(sid: str, processed_dir: str | Path, params_file: str) -> dict:
    from radiomics import featureextractor  # importación tardía: solo se necesita en este paso
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    processed_dir = Path(processed_dir)
    img = processed_dir / f"{sid}_hu.nii.gz"
    roi = processed_dir / f"{sid}_roi.nii.gz"
    ext = featureextractor.RadiomicsFeatureExtractor(params_file)
    try:
        res = ext.execute(str(img), str(roi))
    except Exception as e:  # máscara vacía u otro problema: se registra y se omite
        log.warning("Radiómica falló en %s: %s", sid, e)
        return {"sid": sid}
    feats = {k: float(np.real(v)) for k, v in res.items() if not k.startswith("diagnostics_")}
    feats["sid"] = sid
    return feats


def clean_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Quita columnas no numéricas o con valores faltantes/infinitos."""
    num = df.drop(columns=["sid"]).apply(pd.to_numeric, errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.loc[:, num.notna().all()]
    num.insert(0, "sid", df["sid"].values)
    return num


def _anova_rank(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        F, p = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0)
    p = np.nan_to_num(p, nan=1.0)
    return F, p


def select_features(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int = 0) -> tuple[list[str], dict]:
    """Devuelve los nombres de los rasgos seleccionados y un registro de cada etapa."""
    rc = cfg["radiomics"]
    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    log_info = {"start": len(cols)}

    # 1) varianza
    var = X.var(axis=0).values
    cols = [c for c, v in zip(cols, var) if v > rc["variance_threshold"]]
    log_info["variance"] = len(cols)

    # 2) ANOVA
    F, p = _anova_rank(X[cols].values, y)
    order = np.argsort(-F)
    keep = [cols[i] for i in order if p[i] <= rc["anova_p_max"]][: rc["anova_top_k"]]
    if len(keep) < rc["n_final_features"]:  # si hay pocas significativas, completar por F
        keep = [cols[i] for i in order[: max(rc["anova_top_k"], rc["n_final_features"])]]
    cols = keep
    log_info["anova"] = len(cols)

    # 3) poda por correlación: recorre por F descendente y descarta redundantes
    F_map = dict(zip(cols, _anova_rank(X[cols].values, y)[0]))
    corr = X[cols].corr(method="spearman").abs().fillna(0).values
    kept_idx = []
    for i in np.argsort([-F_map[c] for c in cols]):
        if all(corr[i, j] < rc["corr_threshold"] for j in kept_idx):
            kept_idx.append(i)
    cols = [cols[i] for i in kept_idx]
    log_info["correlation"] = len(cols)

    # 4) estabilidad con bootstrap: frecuencia de aparición en el top-k por ANOVA
    k = min(rc["n_final_features"], len(cols))
    counts = pd.Series(0, index=cols, dtype=float)
    n = len(y)
    for _ in range(rc["bootstrap_rounds"]):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        Fb, _ = _anova_rank(X[cols].values[idx], y[idx])
        top = [cols[i] for i in np.argsort(-Fb)[:k]]
        counts[top] += 1
    counts = counts / max(rc["bootstrap_rounds"], 1)
    selected = list(counts.sort_values(ascending=False).index[:k])
    log_info["stability"] = len(selected)
    if len(selected) < rc["n_final_features"]:
        log.warning("Solo %d rasgos sobrevivieron a la selección (se esperaban %d).", len(selected), rc["n_final_features"])
    log_info["stability_freq"] = counts[selected].round(3).to_dict()
    return selected, log_info
