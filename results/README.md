# Resultados de la replicación

Al ejecutar `scripts/07_evaluate.py`, los resultados se escriben en `outputs/results/`. Copia aquí los archivos finales
que quieras versionar (son pequeños):

- `tabla_original_vs_replicacion.md`: tabla para el PPT (original frente a replicación, con IC 95 % y criterio de éxito).
- `metricas.json`: todas las métricas, umbrales e intervalos de confianza, y las pruebas de DeLong entre flujos.
- `externa_roc.png` y `externa_calibration.png`: curvas ROC y diagrama de confiabilidad en la prueba externa.
- `gradcam/`: ejemplos de mapas Grad-CAM y `gradcam_concordancia.csv`.

## Tabla para completar

| Indicador | Original | Replicación | IC 95 % | ¿Reproducido? |
|---|---|---|---|---|
| A · AUC externo | 0,991 | — | — | — |
| B · AUC externo | ≈ 0,65 | — | — | — |
| C · AUC externo | 0,987 | — | — | — |
| C · Sensibilidad | > 0,93 | — | — | — |
| C · Especificidad | > 0,93 | — | — | — |
| C · Brier | < 0,05 | — | — | — |
