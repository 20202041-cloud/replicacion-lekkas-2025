# Decisiones de implementación

Esta replicación se construyó a partir del **resumen** de Lekkas et al. (2025) y de la información del dataset
PANORAMA. Esta tabla separa lo que viene del artículo de lo que es un supuesto de esta implementación, para que
se pueda ajustar cuando se disponga del texto completo o del código de los autores.

| Componente | Valor usado | Origen |
|---|---|---|
| Dataset | PANORAMA, TC con contraste en fase venosa portal | Artículo |
| Partición | Desarrollo: Siemens + Philips (1 257); prueba externa: Toshiba (651) | Artículo; verificado con `clinical_information.xlsx` |
| Estudio excluido | `100051_00001` (archivo vacío en Zenodo) → desarrollo con 1 256 | Dataset |
| Reorientación | LPS | Artículo (orientación concreta: supuesto) |
| Remuestreo | 1 mm isotrópico | Artículo |
| Ventaneo | −100 a 600 HU | Artículo |
| Normalización | z-score por volumen recortado | Artículo (detalle del cálculo: supuesto) |
| Recorte | 160 × 128 × 96 vóxeles centrado en el páncreas | Supuesto |
| Máscara para radiómica | Etiquetas 1 + 4 + 5 (lesión, parénquima, conducto) | Supuesto |
| Rasgos radiómicos | PyRadiomics, imagen original + wavelet, binWidth 25 | "Cientos de descriptores": artículo; parámetros: supuesto |
| Selección de rasgos | Varianza → ANOVA → correlación → estabilidad bootstrap → 16 | Artículo (umbrales: supuesto) |
| Clasificador del flujo A | Gradient boosting (scikit-learn; XGBoost/LightGBM opcionales) | Artículo (hiperparámetros: supuesto) |
| CNN del flujo B | CBAM-ResNet-18 3D | Artículo |
| Pérdida y entrenamiento | Entropía cruzada ponderada, precisión mixta, AdamW, coseno | Pérdida y AMP: artículo; optimizador y tasas: supuesto |
| Fusión (flujo C) | Embedding CNN + firma radiómica → compuerta SE → clasificador | Artículo (dimensiones: supuesto) |
| Variantes de fusión | (i) flujos congelados; (ii) extremo a extremo | Artículo |
| Explicabilidad | Grad-CAM 3D sobre `layer3`; concordancia con la lesión anotada | Grad-CAM: artículo; capa y métricas: supuesto |
| Validación interna | 5 folds estratificados y agrupados por paciente | Supuesto |
| Umbral de decisión | Youden sobre predicciones de desarrollo | Supuesto |
| Métricas | AUC, sensibilidad, especificidad, exactitud, F1, Brier | Artículo |
| Comparación estadística | IC 95 % por bootstrap estratificado; prueba de DeLong | Supuesto |
| Criterio de éxito | Valor dentro del IC 95 % del original o diferencia ≤ 0,03 de AUC | Supuesto (definido para esta replicación) |

## Riesgos conocidos

- **Pacientes en ambas particiones:** 3 pacientes (9 estudios) tienen estudios en desarrollo y en la prueba externa.
  Se mantienen para reproducir las cifras del artículo; `exclude_cross_split_patients: true` los excluye.
- **Posible atajo por fabricante:** en desarrollo, el 76 % de los estudios Philips son PDAC frente al 18 % de los
  Siemens. Una CNN podría aprender el "estilo" del equipo en lugar del tejido, lo que ayudaría a explicar su caída
  con Toshiba. Conviene reportar el desempeño por fabricante dentro de desarrollo.
- **Valores originales:** el AUC ≈ 0,65 del flujo B proviene del artículo de revisión del autor y debe verificarse con
  el texto completo de Lekkas et al. (2025).
