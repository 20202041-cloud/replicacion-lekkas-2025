#!/usr/bin/env bash
# Ejecuta el pipeline completo cuando TODAS las imágenes ya están en disco.
# Para Colab (procesamiento por lotes), usar notebooks/replicacion_lekkas_colab.ipynb.
set -euo pipefail
CFG="${1:-configs/default.yaml}"
python scripts/00_vendor_split.py   --config "$CFG"
python scripts/01_build_manifest.py --config "$CFG"
python scripts/02_preprocess.py     --config "$CFG"
python scripts/03_extract_radiomics.py --config "$CFG"
python scripts/04_flow_a.py         --config "$CFG"
python scripts/05_train_cnn.py      --config "$CFG"
python scripts/06_train_fusion.py   --config "$CFG" --mode frozen
python scripts/06_train_fusion.py   --config "$CFG" --mode end_to_end
python scripts/07_evaluate.py       --config "$CFG" --mode frozen
python scripts/07_evaluate.py       --config "$CFG" --mode end_to_end
