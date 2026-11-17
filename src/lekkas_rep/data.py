"""Manifiesto de casos y partición por fabricante del tomógrafo.

La tabla clínica de PANORAMA (clinical_information.xlsx en DIAGNijmegen/panorama_labels)
incluye la columna `scanner`. Siguiendo a Lekkas et al. (2025):
  - desarrollo = SIEMENS + Philips  (1 257 estudios)
  - prueba externa = TOSHIBA        (651 estudios)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def read_clinical_table(labels_repo: str | Path) -> pd.DataFrame:
    labels_repo = Path(labels_repo)
    for name in ("clinical_information.xlsx", "clinical_information.csv"):
        p = labels_repo / name
        if p.exists():
            df = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p)
            break
    else:
        raise FileNotFoundError(f"No se encontró clinical_information.(xlsx|csv) en {labels_repo}")
    df["scanner"] = df["scanner"].fillna("NA").astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["y"] = (df["label"].str.upper() == "PDAC").astype(int)
    return df


def assign_split(df: pd.DataFrame, dev_scanners, external_scanners, n_folds: int, seed: int,
                 exclude_cross_split_patients: bool = False) -> pd.DataFrame:
    """Asigna 'dev' o 'external' según el fabricante y crea folds estratificados en 'dev'.

    Los folds se agrupan por paciente (PANORAMA_patient_id) cuando la columna existe, para que un mismo
    paciente no quede a la vez en entrenamiento y validación. En PANORAMA, 3 pacientes (9 estudios) tienen estudios en
    desarrollo y en la prueba externa; con exclude_cross_split_patients=True se excluyen sus estudios.
    """
    dev_set = {s.upper() for s in dev_scanners}
    ext_set = {s.upper() for s in external_scanners}
    up = df["scanner"].str.upper()
    df = df.copy()
    df["split"] = np.where(up.isin(dev_set), "dev", np.where(up.isin(ext_set), "external", "excluded"))
    has_pid = "PANORAMA_patient_id" in df.columns
    if has_pid:
        sp = df.groupby("PANORAMA_patient_id")["split"].agg(lambda s: set(s) >= {"dev", "external"})
        cross = set(sp[sp].index)
        df["cross_split_patient"] = df["PANORAMA_patient_id"].isin(cross)
        if exclude_cross_split_patients:
            df.loc[df["cross_split_patient"], "split"] = "excluded"
    df["fold"] = -1
    dev_idx = df.index[df["split"] == "dev"]
    if len(dev_idx) >= n_folds:
        if has_pid:
            skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            it = skf.split(dev_idx, df.loc[dev_idx, "y"], groups=df.loc[dev_idx, "PANORAMA_patient_id"])
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            it = skf.split(dev_idx, df.loc[dev_idx, "y"])
        for k, (_, te) in enumerate(it):
            df.loc[dev_idx[te], "fold"] = k
    return df


def find_image(images_dir: str | Path, sid: str, patterns) -> str | None:
    images_dir = Path(images_dir)
    for pat in patterns:
        hits = list(images_dir.rglob(pat.format(sid=sid)))
        if hits:
            return str(hits[0])
    return None


def find_label(labels_repo: str | Path, sid: str) -> tuple[str | None, str | None]:
    """Devuelve (etiqueta automática, etiqueta manual) si existen."""
    labels_repo = Path(labels_repo)
    auto = labels_repo / "automatic_labels" / f"{sid}.nii.gz"
    man = labels_repo / "manual_labels" / f"{sid}.nii.gz"
    return (str(auto) if auto.exists() else None, str(man) if man.exists() else None)


def build_manifest(cfg: dict, require_images: bool = True) -> pd.DataFrame:
    """Une tabla clínica, partición, rutas de imagen y de etiquetas."""
    d = cfg["data"]
    df = read_clinical_table(cfg["paths"]["labels_repo"])
    df = assign_split(df, d["dev_scanners"], d["external_scanners"], d["n_folds"], d["seed"],
                      d.get("exclude_cross_split_patients", False))
    excl = set(map(str, d.get("exclude_studies", [])))
    df.loc[df["PANORAMA_study_id"].astype(str).isin(excl), "split"] = "excluded"
    sids = df["PANORAMA_study_id"].astype(str)
    df["image_path"] = [find_image(cfg["paths"]["images_dir"], s, d["image_patterns"]) for s in sids]
    labs = [find_label(cfg["paths"]["labels_repo"], s) for s in sids]
    df["auto_label_path"] = [a for a, _ in labs]
    df["manual_label_path"] = [m for _, m in labs]
    if require_images:
        df = df[df["image_path"].notna()]
    return df.reset_index(drop=True)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab([df["split"], df["scanner"]], df["label"], margins=True)
