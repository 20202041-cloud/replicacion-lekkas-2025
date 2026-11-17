"""Paso 3: extracción de rasgos radiómicos con PyRadiomics (OE2)."""
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from _common import ROOT, out, parse

from lekkas_rep.radiomics_features import clean_feature_table, extract_case


def main():
    args, cfg, log = parse(__doc__)
    m = pd.read_csv(out(cfg, "manifest.csv"))
    if args.limit:
        m = m.head(args.limit)
    params = cfg["radiomics"]["params_file"]
    params = params if Path(params).is_absolute() else str(ROOT / params)
    sids = m["PANORAMA_study_id"].astype(str).tolist()
    rows = Parallel(n_jobs=cfg["radiomics"]["n_jobs"], verbose=5)(
        delayed(extract_case)(s, cfg["paths"]["processed_dir"], params) for s in sids)
    raw = pd.DataFrame(rows)
    raw.to_csv(out(cfg, "radiomics_raw.csv"), index=False)
    feats = clean_feature_table(raw.dropna(subset=[c for c in raw.columns if c != "sid"], how="all"))
    feats.to_csv(out(cfg, "radiomics_features.csv"), index=False)
    log.info("Rasgos radiómicos: %d casos × %d rasgos", len(feats), feats.shape[1] - 1)


if __name__ == "__main__":
    main()
