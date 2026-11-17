"""Genera un mini-dataset sintético con la misma estructura que PANORAMA, para probar el pipeline sin GPU.

No reemplaza a los datos reales: sirve solo para verificar que todo el código se ejecuta de punta a punta.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

VENDOR_NOISE = {"SIEMENS": 12.0, "Philips": 18.0, "TOSHIBA": 25.0}


def ellipsoid(shape, center, radii):
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (((zz - center[0]) / radii[0]) ** 2 + ((yy - center[1]) / radii[1]) ** 2 +
            ((xx - center[2]) / radii[2]) ** 2) <= 1.0


def make_case(rng, pdac: bool, vendor: str, shape=(40, 64, 64), spacing=(1.5, 1.5, 2.5)):
    img = np.full(shape, -900.0, np.float32)                      # aire
    body = ellipsoid(shape, (20, 32, 32), (18, 28, 30))
    img[body] = 40.0                                               # tejido blando
    img[ellipsoid(shape, (20, 32, 32), (16, 25, 27)) & ~ellipsoid(shape, (20, 32, 32), (14, 22, 24))] = -100.0  # grasa
    c = (20 + rng.integers(-2, 3), 30 + rng.integers(-3, 4), 32 + rng.integers(-3, 4))
    panc = ellipsoid(shape, c, (6, 7, 14))
    lab = np.zeros(shape, np.uint8)
    lab[panc] = 4
    duct = ellipsoid(shape, c, (1.2, 1.2, 11))
    lab[duct] = 5
    img[panc] = 110.0
    img[duct] = 20.0
    if pdac:
        lc = (c[0], c[1] + rng.integers(-2, 3), c[2] + rng.integers(-6, 7))
        les = ellipsoid(shape, lc, (3.5, 3.5, 4.5)) & panc
        lab[les] = 1
        img[les] = 45.0 + rng.normal(0, 15, les.sum())             # lesión hipodensa y heterogénea
    img += rng.normal(0, VENDOR_NOISE[vendor], shape)              # ruido propio del "fabricante"
    si = sitk.GetImageFromArray(img)
    si.SetSpacing(spacing[::-1])
    sl = sitk.GetImageFromArray(lab)
    sl.CopyInformation(si)
    return si, sl


def main(root: str, n_dev: int = 30, n_ext: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    img_dir = root / "images"
    lab_repo = root / "panorama_labels"
    (lab_repo / "automatic_labels").mkdir(parents=True, exist_ok=True)
    (lab_repo / "manual_labels").mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    vendors = ["SIEMENS", "Philips"] * (n_dev // 2) + ["TOSHIBA"] * n_ext + ["GE MEDICAL SYSTEMS"] * 2
    for i, v in enumerate(vendors):
        sid = f"{900000 + i}_00001"
        pdac = bool((i // 2) % 2)  # equilibrado dentro de cada fabricante
        si, sl = make_case(rng, pdac, v if v in VENDOR_NOISE else "SIEMENS")
        sitk.WriteImage(si, str(img_dir / f"{sid}_0000.nii.gz"), useCompression=True)
        sitk.WriteImage(sl, str(lab_repo / "automatic_labels" / f"{sid}.nii.gz"), useCompression=True)
        if pdac and i % 4 == 1:
            sitk.WriteImage(sl, str(lab_repo / "manual_labels" / f"{sid}.nii.gz"), useCompression=True)
        rows.append({"PANORAMA_patient_id": 900000 + i, "PANORAMA_study_id": sid, "anonymized_study_date": "2020-01-01",
                     "patient_age": "060Y", "patient_sex": "F", "scanner": v, "label": "PDAC" if pdac else "non-PDAC",
                     "level": "radiology"})
    pd.DataFrame(rows).to_excel(lab_repo / "clinical_information.xlsx", index=False)
    print(f"Dataset sintético: {len(rows)} casos en {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    a = ap.parse_args()
    main(a.root)
