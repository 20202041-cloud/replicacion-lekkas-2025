"""Entrenamiento de la CNN 3D (flujo B) y de la fusión (flujo C).

- Entropía cruzada ponderada por clase y precisión mixta (AMP) [Lekkas et al., 2025].
- Selección del mejor modelo por AUC en el fold de validación (nunca en la prueba externa).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger("lekkas_rep")


class CTDataset(Dataset):
    """Carga volúmenes preprocesados (.npy) y, opcionalmente, la firma radiómica estandarizada."""

    def __init__(self, sids, labels, processed_dir, radiomic: np.ndarray | None = None, augment: bool = False,
                 seed: int = 0):
        self.sids = list(sids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.dir = Path(processed_dir)
        self.radiomic = None if radiomic is None else np.asarray(radiomic, dtype=np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.sids)

    def _aug(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:                       # volteo izquierda-derecha
            x = x[:, :, ::-1]
        sh = [int(self.rng.integers(-6, 7)) for _ in range(3)]  # pequeño desplazamiento
        x = np.roll(x, sh, axis=(0, 1, 2))
        x = x * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.1, 0.1)  # intensidad
        return np.ascontiguousarray(x)

    def __getitem__(self, i):
        x = np.load(self.dir / f"{self.sids[i]}_img.npy").astype(np.float32)
        if self.augment:
            x = self._aug(x)
        item = {"x": torch.from_numpy(x[None]), "y": torch.tensor(self.labels[i]), "sid": self.sids[i]}
        if self.radiomic is not None:
            item["r"] = torch.from_numpy(self.radiomic[i])
        return item


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=2).astype(np.float32)
    w = counts.sum() / (2.0 * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, fusion: bool = False) -> pd.DataFrame:
    dev = _device()
    model.eval()
    rows = []
    for b in loader:
        x = b["x"].to(dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            logits = model(x, b["r"].to(dev))[0] if fusion else model(x)[0]
        p = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
        rows += [{"sid": s, "y": int(y), "prob": float(pp)} for s, y, pp in zip(b["sid"], b["y"].numpy(), p)]
    return pd.DataFrame(rows)


def fit(model, train_ds, val_ds, tc: dict, out_path: str | Path, fusion: bool = False,
        param_groups=None) -> dict:
    """Bucle genérico de entrenamiento con parada temprana por AUC de validación."""
    dev = _device()
    model.to(dev)
    tl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc.get("num_workers", 0),
                    drop_last=len(train_ds) > tc["batch_size"])
    vl = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=tc.get("num_workers", 0))
    crit = nn.CrossEntropyLoss(weight=class_weights(train_ds.labels).to(dev))
    params = param_groups or [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tc["epochs"], 1))
    use_amp = bool(tc.get("amp", True)) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_ep, wait, hist = -1.0, -1, 0, []
    for ep in range(tc["epochs"]):
        model.train()
        if fusion and getattr(model, "mode", "") == "frozen":
            model.backbone.eval()
        tot, n = 0.0, 0
        for b in tl:
            x, y = b["x"].to(dev), b["y"].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x, b["r"].to(dev))[0] if fusion else model(x)[0]
                loss = crit(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach().item() * len(y)
            n += len(y)
        sched.step()
        pv = predict(model, vl, fusion)
        auc = roc_auc_score(pv["y"], pv["prob"]) if pv["y"].nunique() == 2 else float("nan")
        hist.append({"epoch": ep, "loss": tot / max(n, 1), "val_auc": auc})
        log.info("época %d | pérdida %.4f | AUC val %.4f", ep, tot / max(n, 1), auc)
        score = -np.inf if np.isnan(auc) else auc
        if score > best:
            best, best_ep, wait = score, ep, 0
            torch.save(model.state_dict(), out_path)
        else:
            wait += 1
            if wait >= tc["patience"]:
                log.info("Parada temprana en la época %d", ep)
                break
    if best_ep < 0:  # sin AUC válido: guardar el último estado
        torch.save(model.state_dict(), out_path)
    model.load_state_dict(torch.load(out_path, map_location=dev))
    return {"best_val_auc": best, "best_epoch": best_ep, "history": hist}
