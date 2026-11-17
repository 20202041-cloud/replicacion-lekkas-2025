# Replicación: Deep-Radiomic Fusion para la detección temprana de PDAC

Replicación del experimento de **Lekkas, Vrochidou y Papakostas (2025)**, *Deep-Radiomic Fusion for Early Detection
of Pancreatic Ductal Adenocarcinoma*, *Applied Sciences*, 15(24), 13024. https://doi.org/10.3390/app152413024

Proyecto del curso **Propuesta de Investigación 2026-2**, Carrera de Ingeniería de Sistemas, Universidad de Lima.
Autor: Umar Saghir Syed Ybazeta (20202041).

## Qué se replica

El artículo compara tres formas de detectar adenocarcinoma ductal pancreático (PDAC) en tomografías (TC), entrenando
con equipos **Siemens y Philips** y evaluando con equipos **Toshiba**, que el modelo nunca vio:

| Flujo | Entrada | Modelo | AUC externo reportado |
|---|---|---|---|
| **A** | 16 rasgos radiómicos estables | Gradient boosting | 0,991 |
| **B** | Volumen de TC 3D | CNN CBAM-ResNet-18 3D | ≈ 0,65 |
| **C** | Rasgos radiómicos + embedding de la CNN | Fusión con compuerta squeeze-and-excitation | 0,987 (sens./esp. > 0,93; Brier < 0,05) |

## Pipeline

```
TC (PANORAMA) → preprocesamiento → ┬─ A: radiómica → 16 rasgos → gradient boosting ─┐
                                   ├─ B: CBAM-ResNet-18 3D ─────────────────────────┼→ evaluación externa (Toshiba)
                                   └─ C: rasgos + embedding → compuerta SE ─────────┘
```

Preprocesamiento: reorientación, remuestreo a 1 mm isotrópico, ventaneo de −100 a 600 HU, normalización z-score y
recorte de tamaño fijo centrado en el páncreas.

## Datos

- **Imágenes:** PANORAMA, 4 lotes en Zenodo
  ([1](https://zenodo.org/records/13715870), [2](https://zenodo.org/records/13742336),
  [3](https://zenodo.org/records/11034011), [4](https://zenodo.org/records/10999754)).
- **Etiquetas y tabla clínica:** [DIAGNijmegen/panorama_labels](https://github.com/DIAGNijmegen/panorama_labels).
  La columna `scanner` da el fabricante de cada TC.
- **Licencia:** CC BY-NC 4.0 (uso académico no comercial; citar a Alves et al., 2024).

La partición ya está generada en [`data/splits/`](data/splits/resumen_particion.md):

| Partición | Fabricante | PDAC | No PDAC | Total |
|---|---|---|---|---|
| Desarrollo | Siemens + Philips | 411 | 845 | 1 256 |
| Prueba externa | Toshiba | 134 | 517 | 651 |

Se excluye el estudio `100051_00001`, cuyo archivo está vacío en Zenodo; por eso desarrollo tiene 1 256 estudios y no
los 1 257 del artículo. Los folds de validación cruzada se agrupan por paciente.

## Cómo ejecutarlo

### Opción recomendada: Google Colab
Abre [`notebooks/replicacion_lekkas_colab.ipynb`](notebooks/replicacion_lekkas_colab.ipynb) en Colab, activa una GPU y
ejecuta las celdas en orden. El notebook lee las TC directamente de los zips de Zenodo por streaming (no hace falta
descargar los ~200 GB) y guarda las salidas en Google Drive. Empieza con `PILOTO = True`.

**Kaggle:** el mismo notebook funciona quitando la celda de Google Drive y usando `WORK = "/kaggle/working"`.

### Opción local o en servidor (con GPU)
```bash
pip install -r requirements.txt remotezip
pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"
git clone --depth 1 https://github.com/DIAGNijmegen/panorama_labels data/raw/panorama_labels
for b in 1 2 3 4; do python scripts/stream_preprocess.py --config configs/default.yaml --batch $b; done
python scripts/01_build_manifest.py --config configs/default.yaml --use-processed
python scripts/03_extract_radiomics.py --config configs/default.yaml
python scripts/04_flow_a.py --config configs/default.yaml
python scripts/05_train_cnn.py --config configs/default.yaml
python scripts/06_train_fusion.py --config configs/default.yaml --mode frozen
python scripts/07_evaluate.py --config configs/default.yaml --mode frozen
```

## Estructura

```
configs/            default.yaml (replicación), pyradiomics.yaml, smoke_test.yaml (pruebas)
data/splits/        partición por fabricante generada desde la tabla clínica de PANORAMA
docs/decisiones.md  qué viene del artículo y qué es un supuesto de esta implementación
notebooks/          notebook para Google Colab
scripts/            00–07: un script por etapa; stream_preprocess.py y download_panorama.sh para los datos
src/lekkas_rep/     código: datos, preprocesamiento, radiómica, flujo A, modelos, entrenamiento, Grad-CAM, evaluación
tests/              pruebas unitarias y de punta a punta con datos sintéticos
results/            tabla final original frente a replicación
```

## Evaluación

- Métricas en la prueba externa: AUC, sensibilidad, especificidad, exactitud, F1 y Brier.
- Umbral de decisión elegido en desarrollo (Youden), nunca en la prueba externa.
- IC 95 % por bootstrap estratificado y prueba de DeLong entre flujos.
- Grad-CAM 3D y su concordancia con la lesión anotada.
- **Criterio de éxito:** el valor replicado cae dentro del IC 95 % del original o difiere en ≤ 0,03 de AUC.

## Pruebas

```bash
python -m pytest -q            # unitarias + pipeline completo con datos sintéticos (≈ 1 min en CPU)
```
Los datos sintéticos solo verifican que el código corre de punta a punta; no reemplazan a PANORAMA.

## Limitaciones de esta implementación

Se construyó a partir del **resumen** del artículo. Los hiperparámetros, el tamaño del recorte y la lista exacta de los
16 rasgos son supuestos documentados en [`docs/decisiones.md`](docs/decisiones.md). Con el texto completo o el código de
los autores, basta con ajustar `configs/default.yaml`.

## Referencias

- Lekkas, G., Vrochidou, E., & Papakostas, G. A. (2025). Deep-radiomic fusion for early detection of pancreatic ductal
  adenocarcinoma. *Applied Sciences, 15*(24), 13024. https://doi.org/10.3390/app152413024
- Alves, N., Schuurmans, M., Yakar, D., Vendittelli, P., Litjens, G., Hermans, J., & Huisman, H. (2024). *The PANORAMA
  Challenge: Public training and development dataset* (v3) [Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.13715870
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR*.
- van Griethuysen, J. J. M., et al. (2017). Computational radiomics system to decode the radiographic phenotype.
  *Cancer Research, 77*(21), e104–e107.
