"""Paso 4: flujo A — 16 rasgos radiómicos + gradient boosting."""
import joblib
import pandas as pd

from _common import out, parse

from lekkas_rep.flow_a import run_flow_a
from lekkas_rep.utils import save_json


def main():
    args, cfg, log = parse(__doc__)
    m = pd.read_csv(out(cfg, "manifest.csv"))
    feats = pd.read_csv(out(cfg, "radiomics_features.csv"))
    feats["sid"] = feats["sid"].astype(str)
    r = run_flow_a(feats, m, cfg)
    r["dev"].to_csv(out(cfg, "preds", "A_dev_oof.csv"), index=False)
    r["external"].to_csv(out(cfg, "preds", "A_external.csv"), index=False)
    save_json({"selected_features": r["selected_features"], "selection_log": r["selection_log"],
               "fold_features": r["fold_features"]}, out(cfg, "flow_a_features.json"))
    joblib.dump(r["model"], out(cfg, "models", "flow_a_gbm.joblib"))
    log.info("16 rasgos seleccionados: %s", r["selected_features"])


if __name__ == "__main__":
    main()
