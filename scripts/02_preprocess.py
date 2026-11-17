"""Paso 2: preprocesamiento de todas las TC del manifiesto (OE1)."""
import pandas as pd
from joblib import Parallel, delayed

from _common import out, parse

from lekkas_rep.preprocess import is_processed, preprocess_case


def _run(row, cfg):
    sid = str(row["PANORAMA_study_id"])
    try:
        return preprocess_case(row["image_path"], row["auto_label_path"],
                               row["manual_label_path"] if isinstance(row["manual_label_path"], str) else None,
                               sid, cfg["paths"]["processed_dir"], cfg["preprocessing"])
    except Exception as e:  # se registra y continúa con el resto
        return {"sid": sid, "error": str(e)}


def main():
    args, cfg, log = parse(__doc__)
    m = pd.read_csv(out(cfg, "manifest.csv"))
    pdir = cfg["paths"]["processed_dir"]
    n0 = len(m)
    m = m[m["image_path"].notna()]                                   # solo imágenes presentes (lote actual)
    m = m[[not is_processed(pdir, str(s)) for s in m["PANORAMA_study_id"]]]  # reanudable
    log.info("Pendientes en este lote: %d (de %d en el manifiesto)", len(m), n0)
    if args.limit:
        m = m.head(args.limit)
    res = Parallel(n_jobs=cfg["preprocessing"]["n_jobs"], verbose=5)(delayed(_run)(r, cfg) for _, r in m.iterrows())
    df = pd.DataFrame(res)
    logp = out(cfg, "preprocess_log.csv")
    if logp.exists():
        df = pd.concat([pd.read_csv(logp), df], ignore_index=True)
    df.to_csv(logp, index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    log.info("Preprocesados: %d · con error: %d", len(df) - n_err, n_err)


if __name__ == "__main__":
    main()
