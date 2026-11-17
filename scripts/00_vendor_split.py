"""Paso 0: partición por fabricante usando solo la tabla clínica de PANORAMA (no requiere imágenes).

Genera data/splits/panorama_vendor_split.csv y un resumen en data/splits/resumen_particion.md.
Fuente: DIAGNijmegen/panorama_labels (Alves et al., 2024), licencia CC BY-NC 4.0.
"""
from pathlib import Path

from _common import ROOT, parse

from lekkas_rep.data import assign_split, read_clinical_table


def main():
    args, cfg, log = parse(__doc__)
    d = cfg["data"]
    df = read_clinical_table(cfg["paths"]["labels_repo"])
    df = assign_split(df, d["dev_scanners"], d["external_scanners"], d["n_folds"], d["seed"],
                      d.get("exclude_cross_split_patients", False))
    excl = set(map(str, d.get("exclude_studies", [])))
    df.loc[df["PANORAMA_study_id"].astype(str).isin(excl), ["split", "fold"]] = ["excluded", -1]
    cols = ["PANORAMA_patient_id", "PANORAMA_study_id", "scanner", "label", "y", "split", "fold", "cross_split_patient"]
    out_dir = ROOT / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out_dir / "panorama_vendor_split.csv", index=False)

    import pandas as pd
    tab = pd.crosstab([df["split"], df["scanner"]], df["label"], margins=True, margins_name="Total")
    folds = pd.crosstab(df.loc[df.split == "dev", "fold"], df.loc[df.split == "dev", "label"], margins=True, margins_name="Total")
    cross = df.loc[df["cross_split_patient"], ["PANORAMA_patient_id", "PANORAMA_study_id", "scanner", "label", "split"]]
    md = ["# Partición por fabricante de PANORAMA", "",
          "Generada con `scripts/00_vendor_split.py` a partir de `clinical_information.xlsx` "
          "(DIAGNijmegen/panorama_labels; Alves et al., 2024; CC BY-NC 4.0).", "",
          "## Estudios por partición, fabricante y etiqueta", "", tab.to_markdown(), "",
          f"## Folds de validación cruzada en desarrollo (agrupados por paciente, semilla {d['seed']})", "",
          folds.to_markdown(), "",
          f"Estudios excluidos por archivo ilegible en Zenodo: {', '.join(sorted(excl)) or 'ninguno'} "
          "(por eso desarrollo queda con 1 256 estudios en lugar de los 1 257 que reporta el artículo).", "",
          "## Pacientes con estudios en desarrollo y en la prueba externa", "",
          "Se mantienen para reproducir las cifras de Lekkas et al. (2025); pueden excluirse con "
          "`data.exclude_cross_split_patients: true`.", "", cross.to_markdown(index=False), ""]
    (out_dir / "resumen_particion.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Partición guardada en %s", out_dir)
    print(tab)


if __name__ == "__main__":
    main()
