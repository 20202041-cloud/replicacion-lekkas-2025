"""Preprocesamiento por streaming: lee cada TC directamente del zip de Zenodo (peticiones HTTP por rangos),
la preprocesa y borra el archivo temporal. No necesita descargar los ~50 GB de cada lote.

Uso:
  python scripts/stream_preprocess.py --config configs/colab.yaml --batch 1
  python scripts/stream_preprocess.py --config configs/colab.yaml --batch 1 --limit 40   # piloto

Requiere: pip install remotezip
Si el servidor no acepta peticiones por rangos, usar scripts/download_panorama.sh (descarga completa).
"""
import tempfile
from pathlib import Path

import pandas as pd

from _common import out, parse

from lekkas_rep.data import assign_split, find_label, read_clinical_table
from lekkas_rep.preprocess import is_processed, preprocess_case

ZENODO = {1: ("13715870", "batch_1.zip"), 2: ("13742336", "batch_2.zip"),
          3: ("11034011", "batch_3.zip"), 4: ("10999754", "batch_4.zip")}


def main():
    def extra(ap):
        ap.add_argument("--batch", type=int, required=True, choices=[1, 2, 3, 4])
        ap.add_argument("--url", default=None, help="URL alternativa del zip (por defecto, Zenodo)")

    args, cfg, log = parse(__doc__, extra)
    from remotezip import RemoteZip

    d, pdir = cfg["data"], cfg["paths"]["processed_dir"]
    df = read_clinical_table(cfg["paths"]["labels_repo"])
    df = assign_split(df, d["dev_scanners"], d["external_scanners"], d["n_folds"], d["seed"],
                      d.get("exclude_cross_split_patients", False))
    excl = set(map(str, d.get("exclude_studies", [])))
    wanted = set(df.loc[(df["split"] != "excluded") & ~df["PANORAMA_study_id"].astype(str).isin(excl),
                        "PANORAMA_study_id"].astype(str))
    rec, name = ZENODO[args.batch]
    url = args.url or f"https://zenodo.org/records/{rec}/files/{name}?download=1"
    log.info("Abriendo %s", url)
    rows, done = [], 0
    with RemoteZip(url) as z, tempfile.TemporaryDirectory() as tmp:
        members = [m for m in z.namelist() if m.endswith("_0000.nii.gz")]
        todo = []
        for m in members:
            sid = Path(m).name[: -len("_0000.nii.gz")]
            if sid in wanted and not is_processed(pdir, sid):
                todo.append((sid, m))
        log.info("Lote %d: %d volúmenes en el zip; %d pendientes de los que interesan", args.batch, len(members), len(todo))
        if args.limit:
            todo = todo[: args.limit]
        for sid, m in todo:
            try:
                local = z.extract(m, tmp)
                auto, man = find_label(cfg["paths"]["labels_repo"], sid)
                rows.append(preprocess_case(local, auto, man, sid, pdir, cfg["preprocessing"]))
                Path(local).unlink(missing_ok=True)
                done += 1
                if done % 20 == 0:
                    log.info("%d/%d procesados", done, len(todo))
            except Exception as e:
                rows.append({"sid": sid, "error": str(e)})
                log.warning("Error en %s: %s", sid, e)
    logp = out(cfg, "preprocess_log.csv")
    res = pd.DataFrame(rows)
    if logp.exists():
        res = pd.concat([pd.read_csv(logp), res], ignore_index=True)
    res.to_csv(logp, index=False)
    log.info("Lote %d terminado: %d procesados en esta corrida", args.batch, done)


if __name__ == "__main__":
    main()
