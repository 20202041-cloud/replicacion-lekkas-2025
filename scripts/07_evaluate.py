"""Paso 7: evaluación en la prueba externa (Toshiba), comparación entre flujos y con el estudio original.

- Umbral de decisión: Youden sobre predicciones de desarrollo (nunca sobre la prueba externa).
- IC 95 % por bootstrap estratificado y prueba de DeLong para comparar AUC entre flujos.
- Grad-CAM: concordancia con la lesión anotada en casos PDAC externos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import out, parse

from lekkas_rep.evaluation import (bootstrap_ci, comparison_table, delong_test, metrics_at,
                                   plot_roc_and_calibration, youden_threshold)
from lekkas_rep.gradcam import gradcam_3d, lesion_agreement, save_overlay
from lekkas_rep.models import CBAMResNet18_3D, DeepRadiomicFusion
from lekkas_rep.preprocess import load_mask
from lekkas_rep.utils import save_json


def _read(cfg, name):
    p = out(cfg, "preds", name)
    return pd.read_csv(p, dtype={"sid": str}) if p.exists() else None


def main():
    args, cfg, log = parse(__doc__, lambda ap: (ap.add_argument("--mode", default=None),
                                                ap.add_argument("--gradcam-max", type=int, default=200)))
    ev = cfg["evaluation"]
    mode = args.mode or cfg["fusion"]["mode"]
    flows = {"A": ("A_dev_oof.csv", "A_external.csv"), "B": ("B_dev_val.csv", "B_external.csv"),
             "C": (f"C_{mode}_dev_val.csv", f"C_{mode}_external.csv")}
    results, ext_preds = {}, {}
    for k, (fd, fe) in flows.items():
        d, e = _read(cfg, fd), _read(cfg, fe)
        if d is None or e is None or e.empty:
            log.warning("Faltan predicciones del flujo %s; se omite.", k)
            continue
        d = d.dropna(subset=["prob"])
        thr = youden_threshold(d["y"], d["prob"])
        results[k] = {"threshold": thr, "metrics": metrics_at(e["y"], e["prob"], thr),
                      "ci": bootstrap_ci(e["y"], e["prob"], thr, ev["n_bootstrap"], cfg["data"]["seed"])}
        ext_preds[k] = e.set_index("sid")

    # DeLong por pares sobre los casos comunes
    tests = {}
    ks = list(ext_preds)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ext_preds[ks[i]], ext_preds[ks[j]]
            common = a.index.intersection(b.index)
            if len(common) and a.loc[common, "y"].nunique() == 2:
                tests[f"{ks[i]}_vs_{ks[j]}"] = delong_test(a.loc[common, "y"].values, a.loc[common, "prob"].values,
                                                           b.loc[common, "prob"].values)

    table = comparison_table(results, cfg)
    res_dir = Path(cfg["paths"]["outputs_dir"]) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(res_dir / "tabla_original_vs_replicacion.csv", index=False)
    (res_dir / "tabla_original_vs_replicacion.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    save_json({"results": results, "delong": tests, "fusion_mode": mode}, res_dir / "metricas.json")
    if ext_preds:
        plot_roc_and_calibration({k: (v["y"].values, v["prob"].values) for k, v in ext_preds.items()},
                                 str(res_dir / "externa"))
    print(table.to_string(index=False))
    for k, t in tests.items():
        print(f"DeLong {k}: ΔAUC = {t['diff']:.3f}, p = {t['p_value']:.4f}")

    # Grad-CAM (flujos B y C) en casos PDAC externos con lesión anotada
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdir = Path(cfg["paths"]["processed_dir"])
    m = pd.read_csv(out(cfg, "manifest.csv"))
    m["sid"] = m["PANORAMA_study_id"].astype(str)
    cases = [s for s in m.loc[(m["split"] == "external") & (m["y"] == 1), "sid"]
             if (pdir / f"{s}_mask.npz").exists()][: args.gradcam_max]
    agree = []
    b_path, c_path = out(cfg, "models", "flow_b_cnn.pt"), out(cfg, "models", f"flow_c_{mode}.pt")
    models = {}
    if b_path.exists():
        mb = CBAMResNet18_3D(base=cfg["cnn"]["base_channels"])
        mb.load_state_dict(torch.load(b_path, map_location=dev))
        models["B"] = (mb.to(dev), None)
    sc_path = out(cfg, "models", "fusion_radiomic_scaler.json")
    if c_path.exists() and sc_path.exists():
        sc = json.load(open(sc_path, encoding="utf-8"))
        feats = pd.read_csv(out(cfg, "radiomics_features.csv"), dtype={"sid": str}).set_index("sid")
        mc = DeepRadiomicFusion(CBAMResNet18_3D(base=cfg["cnn"]["base_channels"]), len(sc["features"]),
                                cfg["fusion"]["radiomic_proj_dim"], cfg["fusion"]["se_reduction"], cfg["fusion"]["dropout"], mode)
        mc.load_state_dict(torch.load(c_path, map_location=dev))
        models["C"] = (mc.to(dev), (feats, sc))
    ov_dir = res_dir / "gradcam"
    ov_dir.mkdir(exist_ok=True)
    for i, sid in enumerate(cases):
        x = np.load(pdir / f"{sid}_img.npy").astype(np.float32)
        lesion = (load_mask(pdir, sid) == cfg["preprocessing"]["lesion_label"]).astype(np.uint8)
        if lesion.sum() == 0:
            continue
        xt = torch.from_numpy(x[None, None]).to(dev)
        for k, (mdl, extra) in models.items():
            r = None
            if extra is not None:
                feats, sc = extra
                if sid not in feats.index:
                    continue
                v = ((feats.loc[sid, sc["features"]] - pd.Series(sc["mean"])) / pd.Series(sc["std"])).fillna(0)
                r = torch.tensor(v.values[None], dtype=torch.float32, device=dev)
            cam = gradcam_3d(mdl, xt, 1, r, ev.get("gradcam_layer", "layer3"))
            agree.append({"sid": sid, "flow": k, **lesion_agreement(cam, lesion)})
            if i < ev["n_gradcam_examples"]:
                save_overlay(x, cam, lesion, str(ov_dir / f"{sid}_{k}.png"), f"{sid} · flujo {k}")
    if agree:
        ag = pd.DataFrame(agree)
        ag.to_csv(res_dir / "gradcam_concordancia.csv", index=False)
        print(ag.groupby("flow")[["energy_in_lesion", "pointing_hit"]].mean().round(3))
    log.info("Resultados en %s", res_dir)


if __name__ == "__main__":
    main()
