# Partición por fabricante de PANORAMA

Generada con `scripts/00_vendor_split.py` a partir de `clinical_information.xlsx` (DIAGNijmegen/panorama_labels; Alves et al., 2024; CC BY-NC 4.0).

## Estudios por partición, fabricante y etiqueta

|                                       |   PDAC |   non-PDAC |   Total |
|:--------------------------------------|-------:|-----------:|--------:|
| ('dev', 'Philips')                    |    241 |         78 |     319 |
| ('dev', 'SIEMENS')                    |    170 |        767 |     937 |
| ('excluded', '0')                     |      1 |          0 |       1 |
| ('excluded', 'Canon Medical Systems') |      6 |          1 |       7 |
| ('excluded', 'GE MEDICAL SYSTEMS')    |     25 |         21 |      46 |
| ('excluded', 'NA')                    |     99 |        177 |     276 |
| ('excluded', 'SIEMENS')               |      0 |          1 |       1 |
| ('external', 'TOSHIBA')               |    134 |        517 |     651 |
| ('Total', '')                         |    676 |       1562 |    2238 |

## Folds de validación cruzada en desarrollo (agrupados por paciente, semilla 42)

| fold   |   PDAC |   non-PDAC |   Total |
|:-------|-------:|-----------:|--------:|
| 0      |     82 |        170 |     252 |
| 1      |     82 |        169 |     251 |
| 2      |     83 |        169 |     252 |
| 3      |     82 |        168 |     250 |
| 4      |     82 |        169 |     251 |
| Total  |    411 |        845 |    1256 |

Estudios excluidos por archivo ilegible en Zenodo: 100051_00001 (por eso desarrollo queda con 1 256 estudios en lugar de los 1 257 que reporta el artículo).

## Pacientes con estudios en desarrollo y en la prueba externa

Se mantienen para reproducir las cifras de Lekkas et al. (2025); pueden excluirse con `data.exclude_cross_split_patients: true`.

|   PANORAMA_patient_id |   PANORAMA_study_id | scanner   | label    | split    |
|----------------------:|--------------------:|:----------|:---------|:---------|
|                100047 |        100047_00001 | Philips   | non-PDAC | dev      |
|                100047 |        100047_00002 | SIEMENS   | non-PDAC | dev      |
|                100047 |        100047_00005 | TOSHIBA   | non-PDAC | external |
|                100047 |        100047_00003 | TOSHIBA   | non-PDAC | external |
|                100047 |        100047_00004 | SIEMENS   | non-PDAC | dev      |
|                100265 |        100265_00001 | TOSHIBA   | PDAC     | external |
|                100265 |        100265_00002 | Philips   | PDAC     | dev      |
|                100416 |        100416_00001 | TOSHIBA   | PDAC     | external |
|                100416 |        100416_00002 | SIEMENS   | PDAC     | dev      |
