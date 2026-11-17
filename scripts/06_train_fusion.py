"""Paso 6: flujo C — fusión deep-radiómica con compuerta squeeze-and-excitation.

--mode frozen      CNN congelada (pesos del flujo B); se entrenan adaptador, compuerta y clasificador.
--mode end_to_end  la CNN se ajusta junto con el adaptador (tasa de aprendizaje menor para la CNN).
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from _common import out, parse

from lekkas_rep.models import CBAMResNet18_3D, DeepRadiomicFusion
from lekkas_rep.training import CTDataset, fit, predict
from lekkas_rep.utils import save_json


def main():
    args, cfg, log = parse(__doc__, lambda ap: ap.add_argument("--mode", choices=["frozen", "end_to_end"], default=None))
    fc = cfg["fusion"]
    mode = args.mode or fc["mode"]
    m = pd.read_csv(out(cfg, "manifest.csv"))
    m["sid"] = m["PANORAMA_study_id"].astype(str)
    feats = pd.read_csv(out(cfg, "radiomics_features.csv"))
    feats["sid"] = feats["sid"].astype(str)
    sel = json.load(open(out(cfg, "flow_a_features.json"), encoding="utf-8"))["selected_features"]
    m = m.merge(feats[["sid"] + sel], on="sid", how="inner")

    vf, pdir = cfg["data"]["val_fold"], cfg["paths"]["processed_dir"]
    dev = m[m["split"] == "dev"]
    tr, va, ext = dev[dev["fold"] != vf], dev[dev["fold"] == vf], m[m["split"] == "external"]
    mu, sd = tr[sel].mean(), tr[sel].std().replace(0, 1)  # estandarización ajustada solo en entrenamiento
    R = lambda d: ((d[sel] - mu) / sd).fillna(0).values.astype(np.float32)

    backbone = CBAMResNet18_3D(base=cfg["cnn"]["base_channels"])
    backbone.load_state_dict(torch.load(out(cfg, "models", "flow_b_cnn.pt"), map_location="cpu"))
    model = DeepRadiomicFusion(backbone, len(sel), fc["radiomic_proj_dim"], fc["se_reduction"], fc["dropout"], mode)
    groups = None
    if mode == "end_to_end":
        head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        groups = [{"params": head, "lr": fc["lr"]}, {"params": model.backbone.parameters(), "lr": fc["backbone_lr"]}]
    seed = cfg["data"]["seed"]
    ds_tr = CTDataset(tr["sid"], tr["y"], pdir, R(tr), augment=cfg["cnn"]["augment"], seed=seed)
    ds_va = CTDataset(va["sid"], va["y"], pdir, R(va))
    tc = {**fc, "num_workers": cfg["cnn"].get("num_workers", 0)}
    info = fit(model, ds_tr, ds_va, tc, out(cfg, "models", f"flow_c_{mode}.pt"), fusion=True, param_groups=groups)
    save_json({**info, "mode": mode}, out(cfg, f"flow_c_{mode}_training.json"))
    bs, nw = fc["batch_size"], tc["num_workers"]
    predict(model, DataLoader(ds_va, batch_size=bs, num_workers=nw), fusion=True).to_csv(out(cfg, "preds", f"C_{mode}_dev_val.csv"), index=False)
    ds_ext = CTDataset(ext["sid"], ext["y"], pdir, R(ext))
    predict(model, DataLoader(ds_ext, batch_size=bs, num_workers=nw), fusion=True).to_csv(out(cfg, "preds", f"C_{mode}_external.csv"), index=False)
    save_json({"mean": mu.to_dict(), "std": sd.to_dict(), "features": sel}, out(cfg, "models", "fusion_radiomic_scaler.json"))
    log.info("Flujo C (%s) listo. Mejor AUC de validación: %.4f", mode, info["best_val_auc"])


if __name__ == "__main__":
    main()
