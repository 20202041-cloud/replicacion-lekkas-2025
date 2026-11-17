"""Paso 1: manifiesto de casos con la partición por fabricante (desarrollo vs. prueba externa)."""
from _common import out, parse

from lekkas_rep.data import build_manifest, split_summary
from lekkas_rep.preprocess import is_processed


def main():
    def extra(ap):
        ap.add_argument("--allow-missing-images", action="store_true",
                        help="incluir casos cuya imagen no está en disco (procesamiento por lotes)")
        ap.add_argument("--use-processed", action="store_true",
                        help="manifiesto final: incluir solo casos ya preprocesados (las imágenes crudas pueden haberse borrado)")

    args, cfg, log = parse(__doc__, extra)
    df = build_manifest(cfg, require_images=not (args.allow_missing_images or args.use_processed))
    if args.use_processed:
        df = df[[is_processed(cfg["paths"]["processed_dir"], str(s)) for s in df["PANORAMA_study_id"]]]
    df = df[df["split"] != "excluded"].reset_index(drop=True)
    if args.limit:  # piloto: submuestra estratificada por partición y etiqueta
        df = (df.groupby(["split", "y"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max(1, args.limit // 4)), random_state=cfg["data"]["seed"])))
        df = df.reset_index(drop=True)
    path = out(cfg, "manifest.csv")
    df.to_csv(path, index=False)
    log.info("Manifiesto: %d casos → %s", len(df), path)
    print(split_summary(df))


if __name__ == "__main__":
    main()
