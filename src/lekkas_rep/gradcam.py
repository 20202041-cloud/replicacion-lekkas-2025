"""Grad-CAM 3D y concordancia con la lesión anotada.

Métricas de concordancia (sobre casos PDAC con lesión anotada):
  - energía dentro de la lesión: fracción de la activación del mapa que cae dentro de la máscara dilatada
  - pointing game: si el vóxel de máxima activación cae dentro de la máscara dilatada
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation


def gradcam_3d(model, x: torch.Tensor, target_class: int = 1, fusion_r: torch.Tensor | None = None,
               layer: str = "layer3") -> np.ndarray:
    """Calcula el mapa Grad-CAM para un volumen x de forma (1, 1, D, H, W). Devuelve (D, H, W) en [0, 1].

    Por defecto usa layer3 (resolución 1/16) en vez de layer4 (1/32), para obtener mapas menos gruesos en 3D.
    """
    model.eval()
    backbone = model.backbone if fusion_r is not None else model
    acts = {}

    def hook(_, __, out):
        out.retain_grad()
        acts["a"] = out

    h = getattr(backbone, layer).register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # garantiza gradientes aunque la CNN esté congelada
            if fusion_r is not None:
                prev = model.mode
                model.mode = "end_to_end"  # para propagar gradientes por la CNN
                logits = model(x, fusion_r)[0]
                model.mode = prev
            else:
                logits = model(x)[0]
            model.zero_grad(set_to_none=True)
            logits[0, target_class].backward()
        a, g = acts["a"], acts["a"].grad
        w = g.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
    finally:
        h.remove()
    cam -= cam.min()
    return cam / (cam.max() + 1e-8)


def lesion_agreement(cam: np.ndarray, lesion: np.ndarray, dilate_vox: int = 5) -> dict:
    if lesion.sum() == 0:
        return {"energy_in_lesion": np.nan, "pointing_hit": np.nan}
    m = binary_dilation(lesion > 0, iterations=dilate_vox)
    energy = float(cam[m].sum() / (cam.sum() + 1e-8))
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {"energy_in_lesion": energy, "pointing_hit": float(bool(m[peak]))}


def save_overlay(x: np.ndarray, cam: np.ndarray, lesion: np.ndarray, path: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = int(np.argmax(lesion.sum(axis=(1, 2)))) if lesion.sum() else x.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(x[z], cmap="gray")
    if lesion.sum():
        ax[0].contour(lesion[z] > 0, levels=[0.5], colors="lime", linewidths=1)
    ax[0].set_title("TC + lesión anotada")
    ax[1].imshow(x[z], cmap="gray")
    ax[1].imshow(cam[z], cmap="jet", alpha=0.4)
    ax[1].set_title("Grad-CAM")
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
