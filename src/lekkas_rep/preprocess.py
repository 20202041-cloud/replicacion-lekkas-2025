"""Preprocesamiento de las TC (OE1).

Pasos [Lekkas et al., 2025]:
  1. reorientación del volumen
  2. remuestreo a 1 mm isotrópico (lineal para la imagen, vecino más cercano para la máscara)
  3. ventaneo a [−100, 600] HU
  4. normalización z-score
  5. recorte de tamaño fijo centrado en el páncreas

Salidas por estudio (en processed_dir):
  - {sid}_img.npy      volumen normalizado (float16) de tamaño crop_size, orden (z, y, x)
  - {sid}_mask.npz     máscara (uint8, comprimida) con las etiquetas originales de PANORAMA
  - {sid}_hu.nii.gz    imagen remuestreada y ventaneada en HU, recortada alrededor del páncreas (PyRadiomics)
  - {sid}_roi.nii.gz   máscara binaria del páncreas en el mismo recorte (PyRadiomics)

Guardar solo recortes permite procesar PANORAMA lote por lote y conservar las salidas en Google Drive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def reorient(img: sitk.Image, orientation: str = "LPS") -> sitk.Image:
    return sitk.DICOMOrient(img, orientation)


def resample(img: sitk.Image, spacing, is_label: bool) -> sitk.Image:
    spacing = [float(s) for s in spacing]
    in_sp, in_sz = img.GetSpacing(), img.GetSize()
    out_sz = [int(round(sz * sp / ns)) for sz, sp, ns in zip(in_sz, in_sp, spacing)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(spacing)
    r.SetSize(out_sz)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0 if is_label else -1024)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return r.Execute(img)


def window_hu(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m, s = float(arr.mean()), float(arr.std())
    return (arr - m) / (s + eps)


def bbox_center(mask: np.ndarray) -> tuple[int, int, int]:
    """Centro del cuadro delimitador de la máscara; centro del volumen si está vacía."""
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return tuple(int(s // 2) for s in mask.shape)
    lo, hi = idx.min(0), idx.max(0)
    return tuple(int(v) for v in ((lo + hi) // 2))


def crop_or_pad(arr: np.ndarray, center, size, fill) -> np.ndarray:
    """Recorta un bloque de `size` centrado en `center`; rellena con `fill` fuera del volumen."""
    out = np.full(size, fill, dtype=arr.dtype)
    src_lo = [c - s // 2 for c, s in zip(center, size)]
    for_src, for_dst = [], []
    for d in range(3):
        a0 = max(src_lo[d], 0)
        a1 = min(src_lo[d] + size[d], arr.shape[d])
        b0 = a0 - src_lo[d]
        for_src.append(slice(a0, a1))
        for_dst.append(slice(b0, b0 + max(a1 - a0, 0)))
    out[tuple(for_dst)] = arr[tuple(for_src)]
    return out


def load_mask(processed_dir: str | Path, sid: str) -> np.ndarray:
    return np.load(Path(processed_dir) / f"{sid}_mask.npz")["mask"]


def is_processed(processed_dir: str | Path, sid: str) -> bool:
    d = Path(processed_dir)
    return all((d / f"{sid}{s}").exists() for s in ("_img.npy", "_mask.npz", "_hu.nii.gz", "_roi.nii.gz"))


def preprocess_case(image_path: str, label_path: str | None, manual_label_path: str | None,
                    sid: str, out_dir: str | Path, pp: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = reorient(sitk.ReadImage(image_path, sitk.sitkFloat32), pp["orientation"])
    img = resample(img, pp["spacing_mm"], is_label=False)

    if label_path:
        lab = reorient(sitk.ReadImage(label_path, sitk.sitkUInt8), pp["orientation"])
        if manual_label_path:  # la lesión manual reemplaza a la automática cuando existe
            man = reorient(sitk.ReadImage(manual_label_path, sitk.sitkUInt8), pp["orientation"])
            la, ma = sitk.GetArrayFromImage(lab), sitk.GetArrayFromImage(man)
            if la.shape == ma.shape:
                la = la.copy()
                la[la == pp["lesion_label"]] = 4  # la lesión automática pasa a parénquima
                la[ma == pp["lesion_label"]] = pp["lesion_label"]
                lab2 = sitk.GetImageFromArray(la)
                lab2.CopyInformation(lab)
                lab = lab2
        lab = sitk.Resample(lab, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        lab_arr = sitk.GetArrayFromImage(lab)
    else:
        lab_arr = np.zeros(sitk.GetArrayFromImage(img).shape, np.uint8)

    lo, hi = pp["hu_window"]
    hu = window_hu(sitk.GetArrayFromImage(img), lo, hi).astype(np.float32)  # (z, y, x)
    roi = np.isin(lab_arr, pp["pancreas_labels"]).astype(np.uint8)

    # Para PyRadiomics: imagen HU ventaneada y máscara del páncreas, recortadas al páncreas + margen.
    hu_img = sitk.GetImageFromArray(hu)
    hu_img.CopyInformation(img)
    roi_img = sitk.GetImageFromArray(roi)
    roi_img.CopyInformation(img)
    idx = np.argwhere(roi > 0)
    if idx.size:
        mg = int(round(pp.get("crop_margin_mm", 10) / float(pp["spacing_mm"][0])))
        lo_zyx = np.maximum(idx.min(0) - mg, 0)
        hi_zyx = np.minimum(idx.max(0) + mg + 1, np.array(roi.shape))
        start, size = [int(v) for v in lo_zyx[::-1]], [int(v) for v in (hi_zyx - lo_zyx)[::-1]]
        hu_img = sitk.RegionOfInterest(hu_img, size, start)
        roi_img = sitk.RegionOfInterest(roi_img, size, start)
    sitk.WriteImage(hu_img, str(out_dir / f"{sid}_hu.nii.gz"), useCompression=True)
    sitk.WriteImage(roi_img, str(out_dir / f"{sid}_roi.nii.gz"), useCompression=True)

    # Para la CNN: recorte fijo centrado en el páncreas, luego z-score.
    size_zyx = tuple(int(s) for s in pp["crop_size"][::-1])
    c = bbox_center(roi)
    hu_crop = crop_or_pad(hu, c, size_zyx, fill=float(lo))
    lab_crop = crop_or_pad(lab_arr, c, size_zyx, fill=0)
    x = zscore(hu_crop).astype(np.float16)
    np.save(out_dir / f"{sid}_img.npy", x)
    np.savez_compressed(out_dir / f"{sid}_mask.npz", mask=lab_crop.astype(np.uint8))
    return {"sid": sid, "roi_voxels": int(roi.sum()), "lesion_voxels": int((lab_crop == pp["lesion_label"]).sum()),
            "shape_zyx": list(x.shape)}
