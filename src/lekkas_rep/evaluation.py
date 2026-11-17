"""Métricas, intervalos de confianza, prueba de DeLong y tabla comparativa con el estudio original."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score, roc_curve


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def metrics_at(y, p, thr: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "threshold": thr, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": len(y),
    }


def bootstrap_ci(y, p, thr: float, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """IC 95 % por bootstrap estratificado (remuestreo dentro de cada clase)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keys = ["auc", "sensitivity", "specificity", "accuracy", "f1", "brier"]
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = metrics_at(y[idx], p[idx], thr)
        for k in keys:
            vals[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for k, v in vals.items()}


# --- Prueba de DeLong para AUC correlacionados (Sun y Xu, 2014: implementación rápida) ---
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2) -> dict:
    """Compara dos AUC calculados sobre los mismos casos. Devuelve AUCs, diferencia y valor p bilateral."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")  # positivos primero
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(preds, m)
    l = np.array([1, -1])
    var = float(l @ cov @ l.T)
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(aucs[0] - aucs[1]),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def comparison_table(results: dict, cfg: dict) -> pd.DataFrame:
    """Tabla 'original frente a replicación' con el criterio de éxito definido en la configuración."""
    o, tol = cfg["evaluation"]["original"], cfg["evaluation"]["success_tolerance_auc"]
    rows = []

    def add(ind, orig, val, ci, ok):
        rows.append({"Indicador": ind, "Original": orig,
                     "Replicación": "—" if val is None else f"{val:.3f}",
                     "IC 95 %": "—" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                     "¿Reproducido?": "—" if ok is None else ("Sí" if ok else "No")})

    def auc_ok(val, ci, ref):
        return abs(val - ref) <= tol or (ci[0] <= ref <= ci[1])

    for flow, key in (("A", "A_auc"), ("B", "B_auc"), ("C", "C_auc")):
        r = results.get(flow)
        if r is None:
            add(f"{flow} · AUC externo", f"{o[key]}", None, None, None)
            continue
        v, ci = r["metrics"]["auc"], r["ci"]["auc"]
        add(f"{flow} · AUC externo", f"{o[key]}", v, ci, auc_ok(v, ci, o[key]))
    c = results.get("C")
    if c:
        m, ci = c["metrics"], c["ci"]
        add("C · Sensibilidad", f"> {o['C_sens_min']}", m["sensitivity"], ci["sensitivity"], m["sensitivity"] > o["C_sens_min"])
        add("C · Especificidad", f"> {o['C_spec_min']}", m["specificity"], ci["specificity"], m["specificity"] > o["C_spec_min"])
        add("C · Brier", f"< {o['C_brier_max']}", m["brier"], ci["brier"], m["brier"] < o["C_brier_max"])
    return pd.DataFrame(rows)


def plot_roc_and_calibration(curves: dict, path_prefix: str) -> None:
    """curves = {nombre: (y, p)}. Guarda ROC y diagrama de confiabilidad."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    colors = {"A": "#000000", "B": "#A6A6A6", "C": "#FF3A00"}
    fig, ax = plt.subplots(figsize=(5, 5))
    for k, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=colors.get(k), lw=2, label=f"{k} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#BBBBBB", lw=1)
    ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad"); ax.set_title("Curvas ROC · prueba externa")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(f"{path_prefix}_roc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for k, (y, p) in curves.items():
        fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "o-", color=colors.get(k), label=k)
    ax.plot([0, 1], [0, 1], "--", color="#BBBBBB", lw=1)
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Fracción observada de PDAC")
    ax.set_title("Diagrama de confiabilidad · prueba externa"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{path_prefix}_calibration.png", dpi=150); plt.close(fig)
