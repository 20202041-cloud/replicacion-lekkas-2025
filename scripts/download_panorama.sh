#!/usr/bin/env bash
# Descarga UN lote de imágenes de PANORAMA desde Zenodo, verifica su SHA-256 y lo descomprime.
# Uso:  bash scripts/download_panorama.sh <lote 1-4> <carpeta_destino>
# Ejemplo: bash scripts/download_panorama.sh 1 data/raw/images
#
# Lotes (Alves et al., 2024; CC BY-NC 4.0):
#   1 → https://zenodo.org/records/13715870   2 → https://zenodo.org/records/13742336
#   3 → https://zenodo.org/records/11034011   4 → https://zenodo.org/records/10999754
set -euo pipefail
BATCH="${1:?indica el lote (1-4)}"
DEST="${2:-data/raw/images}"
case "$BATCH" in
  1) REC=13715870; SHA=aff39b6347650d6c7457adf7a04bfb0a651ab6ecd33676ff109bdab17bc41cff ;;
  2) REC=13742336; SHA=db6353a2c1c565c8bf084bd4fe1512fd6020b7675a1c9ab61b9a13d72a9fe76c ;;
  3) REC=11034011; SHA=c1d71b40948edc36f795a7801cc79000082df8d365c48574af50b36516d64cee ;;
  4) REC=10999754; SHA=3b5341af79c2cc8b8a9fa3ab7a6cfa8fedf694538a3d6be97c18e5c82be4d9d8 ;;
  *) echo "Lote inválido: $BATCH"; exit 1 ;;
esac
mkdir -p "$DEST"
ZIP="$DEST/batch_${BATCH}.zip"
URL="https://zenodo.org/records/${REC}/files/batch_${BATCH}.zip?download=1"
echo "Descargando lote ${BATCH} desde ${URL}"
wget -c -O "$ZIP" "$URL"
echo "Verificando SHA-256…"
echo "${SHA}  ${ZIP}" | sha256sum -c -
echo "Descomprimiendo…"
unzip -q -o "$ZIP" -d "$DEST/batch_${BATCH}"
rm -f "$ZIP"   # libera espacio: el zip ya no se necesita
echo "Listo. Imágenes en $DEST/batch_${BATCH} ($(find "$DEST/batch_${BATCH}" -name '*_0000.nii.gz' | wc -l) volúmenes)"
