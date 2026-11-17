"""Paso 5: flujo B — CBAM-ResNet-18 3D (entropía cruzada ponderada, precisión mixta)."""
import pandas as pd

from _common import out, parse

from lekkas_rep.models import CBAMResNet18_3D
from lekkas_rep.training import CTDataset, fit, predict
from lekkas_rep.utils import save_json
from torch.utils.data import DataLoader


def main():
    args, cfg, log = parse(__doc__)
    m = pd.read_csv(out(cfg, "manifest.csv"))
    m["sid"] = m["PANORAMA_study_id"].astype(str)
    pdir, tc, vf = cfg["paths"]["processed_dir"], cfg["cnn"], cfg["data"]["val_fold"]
    dev = m[m["split"] == "dev"]
    tr, va, ext = dev[dev["fold"] != vf], dev[dev["fold"] == vf], m[m["split"] == "external"]
    seed = cfg["data"]["seed"]
    ds_tr = CTDataset(tr["sid"], tr["y"], pdir, augment=tc["augment"], seed=seed)
    ds_va = CTDataset(va["sid"], va["y"], pdir)
    model = CBAMResNet18_3D(base=tc["base_channels"])
    info = fit(model, ds_tr, ds_va, tc, out(cfg, "models", "flow_b_cnn.pt"))
    save_json(info, out(cfg, "flow_b_training.json"))
    bs, nw = tc["batch_size"], tc.get("num_workers", 0)
    predict(model, DataLoader(ds_va, batch_size=bs, num_workers=nw)).to_csv(out(cfg, "preds", "B_dev_val.csv"), index=False)
    ds_ext = CTDataset(ext["sid"], ext["y"], pdir)
    predict(model, DataLoader(ds_ext, batch_size=bs, num_workers=nw)).to_csv(out(cfg, "preds", "B_external.csv"), index=False)
    log.info("Flujo B listo. Mejor AUC de validación: %.4f", info["best_val_auc"])


if __name__ == "__main__":
    main()
