# Valid Capture Conditions Batch V3 — Evaluation Report

**Fecha:** 2026-05-20 22:54:19
**Lote:** `supermarket_valid_conditions_batch_v3`
**Condiciones:** fondo blanco/gris/beige mate, iluminación controlada
**Modelo:** `outputs/fruits360_quality_cls_v2/best_model.pt` (V2, congelado)
**Entrada:** `data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3/`
(22 imágenes)

---

## Resumen

| Métrica | Valor |
|---|---|
| Imágenes procesadas | 22 |
| Predichas GOOD | 19 |
| Predichas BAD | 3 |
| Tasa de acierto (asumiendo todas GOOD) | 86.4% |
| Confianza media GOOD | 0.7762 |
| Confianza media BAD | 0.5350 |
| Tiempo de inferencia | 2.33 s |

---

## Interpretación técnica

**V2 clasifica correctamente 19/22 peras (86.4%) bajo condiciones válidas de captura.** Los 3 caso(s) BAD deben revisarse visualmente. Si son peras sanas, guardarlos como hard examples GOOD para V3. Si tienen defecto real, son ejemplos BAD válidos.

---

**Imágenes predichas BAD — requieren revisión humana:**

| filename | pred_confidence |
|---|---|
| `1000060792.jpg` | 0.5693 |
| `1000060802.jpg` | 0.5120 |
| `1000060811.jpg` | 0.5238 |


---

## Contexto y motivación

Este lote evalúa V2 específicamente bajo las condiciones recomendadas tras el análisis
del lote V2, donde se detectó que el modelo fallaba principalmente en fondos azul y negro:

| Lote | Total | GOOD | BAD | Tasa error |
|---|---|---|---|---|
| batch_v1 (fondo variado)      | 20 | 19 | 1  | 5%  |
| batch_v2 (fondos mixtos)      | 22 | 15 | 7  | 32% |
| **batch_v3 (fondo válido)**   | **22** | **19** | **3** | **14%** |

---

## Predicciones completas

| image_id | filename | pred_label | pred_confidence | prob_good | prob_bad |
|---|---|---|---|---|---|
| 1 | 1000060790.jpg | GOOD | 0.8700 | 0.8700 | 0.1300 |
| 2 | 1000060791.jpg | GOOD | 0.5629 | 0.5629 | 0.4371 |
| 3 | 1000060792.jpg | BAD | 0.5693 | 0.4307 | 0.5693 |
| 4 | 1000060793.jpg | GOOD | 0.9598 | 0.9598 | 0.0402 |
| 5 | 1000060794.jpg | GOOD | 0.9244 | 0.9244 | 0.0756 |
| 6 | 1000060795.jpg | GOOD | 0.9529 | 0.9529 | 0.0471 |
| 7 | 1000060796.jpg | GOOD | 0.6307 | 0.6307 | 0.3693 |
| 8 | 1000060797.jpg | GOOD | 0.5982 | 0.5982 | 0.4018 |
| 9 | 1000060800.jpg | GOOD | 0.6779 | 0.6779 | 0.3221 |
| 10 | 1000060801.jpg | GOOD | 0.5026 | 0.5026 | 0.4974 |
| 11 | 1000060802.jpg | BAD | 0.5120 | 0.4880 | 0.5120 |
| 12 | 1000060805.jpg | GOOD | 0.7650 | 0.7650 | 0.2350 |
| 13 | 1000060806.jpg | GOOD | 0.7374 | 0.7374 | 0.2626 |
| 14 | 1000060807.jpg | GOOD | 0.6702 | 0.6702 | 0.3298 |
| 15 | 1000060808.jpg | GOOD | 0.9357 | 0.9357 | 0.0643 |
| 16 | 1000060809.jpg | GOOD | 0.8359 | 0.8359 | 0.1641 |
| 17 | 1000060810.jpg | GOOD | 0.9274 | 0.9274 | 0.0726 |
| 18 | 1000060811.jpg | BAD | 0.5238 | 0.4762 | 0.5238 |
| 19 | 1000060812.jpg | GOOD | 0.7412 | 0.7412 | 0.2588 |
| 20 | 1000060813.jpg | GOOD | 0.7382 | 0.7382 | 0.2618 |
| 21 | 1000060814.jpg | GOOD | 0.8723 | 0.8723 | 0.1277 |
| 22 | 1000060815.jpg | GOOD | 0.8456 | 0.8456 | 0.1544 |

---

## Archivos generados

- `outputs/supermarket_valid_conditions_batch_v3_eval/predictions.csv`
- `outputs/supermarket_valid_conditions_batch_v3_eval/human_review_template.csv`
- `outputs/supermarket_valid_conditions_batch_v3_eval/contact_sheet_all.jpg`
- `outputs/supermarket_valid_conditions_batch_v3_eval/contact_sheet_pred_good.jpg`
- `outputs/supermarket_valid_conditions_batch_v3_eval/contact_sheet_pred_bad.jpg`
- `outputs/supermarket_valid_conditions_batch_v3_eval/summary.txt`

---

## Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.
- **NO** se modificó ningún dataset anterior.

---

## Siguiente paso

José debe abrir `outputs/supermarket_valid_conditions_batch_v3_eval/contact_sheet_all.jpg`
y confirmar si todas las imágenes son GOOD o si hay falsos BAD.
