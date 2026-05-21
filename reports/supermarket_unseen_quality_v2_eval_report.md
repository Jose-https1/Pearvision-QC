# Supermarket Unseen Quality V2 — Evaluation Report

**Fecha:** 2026-05-20 21:59:31
**Modelo:** `outputs/fruits360_quality_cls_v2/best_model.pt` (MobileNetV3-small, V2 congelado)
**Datos entrada:** `data/unseen_quality_eval_input/` (20 imágenes de supermercado, NO vistas durante entrenamiento)

---

## Resumen

| Métrica | Valor |
|---|---|
| Imágenes procesadas | 20 |
| Predichas GOOD | 19 |
| Predichas BAD | 1 |
| Confianza media GOOD | 0.8481 |
| Confianza media BAD | 0.5275 |
| Tiempo de inferencia | 2.23 s |

---

## Predicciones por imagen

| image_id | filename | pred_label | pred_confidence | prob_good | prob_bad |
|---|---|---|---|---|---|
| 1 | 1000060736.jpg | GOOD | 0.7507 | 0.7507 | 0.2493 |
| 2 | 1000060737.jpg | GOOD | 0.8817 | 0.8817 | 0.1183 |
| 3 | 1000060738.jpg | GOOD | 0.9010 | 0.9010 | 0.0990 |
| 4 | 1000060739.jpg | GOOD | 0.8743 | 0.8743 | 0.1257 |
| 5 | 1000060740.jpg | GOOD | 0.6555 | 0.6555 | 0.3445 |
| 6 | 1000060741.jpg | GOOD | 0.9209 | 0.9209 | 0.0791 |
| 7 | 1000060743.jpg | GOOD | 0.8122 | 0.8122 | 0.1878 |
| 8 | 1000060744.jpg | GOOD | 0.8560 | 0.8560 | 0.1440 |
| 9 | 1000060745.jpg | GOOD | 0.9417 | 0.9417 | 0.0583 |
| 10 | 1000060746.jpg | GOOD | 0.8337 | 0.8337 | 0.1663 |
| 11 | 1000060747.jpg | BAD | 0.5275 | 0.4725 | 0.5275 |
| 12 | 1000060748.jpg | GOOD | 0.7555 | 0.7555 | 0.2445 |
| 13 | 1000060749.jpg | GOOD | 0.8494 | 0.8494 | 0.1506 |
| 14 | 1000060750.jpg | GOOD | 0.7852 | 0.7852 | 0.2148 |
| 15 | 1000060751.jpg | GOOD | 0.8107 | 0.8107 | 0.1893 |
| 16 | 1000060752.jpg | GOOD | 0.8262 | 0.8262 | 0.1738 |
| 17 | 1000060753.jpg | GOOD | 0.8070 | 0.8070 | 0.1930 |
| 18 | 1000060754.jpg | GOOD | 0.9310 | 0.9310 | 0.0690 |
| 19 | 1000060755.jpg | GOOD | 0.9537 | 0.9537 | 0.0463 |
| 20 | 1000060756.jpg | GOOD | 0.9669 | 0.9669 | 0.0331 |

---

## Contexto de las imágenes

Las peras evaluadas son peras comerciales compradas en supermercado español.
Son peras sanas, con partes verdes y manchas marrones/russeting naturales.
Esta evaluación comprueba si el modelo V2 confunde características naturales con defectos.

---

## Archivos generados

- `outputs/supermarket_unseen_quality_v2_eval/predictions.csv`
- `outputs/supermarket_unseen_quality_v2_eval/human_review_template.csv`
- `outputs/supermarket_unseen_quality_v2_eval/contact_sheet_all.jpg`
- `outputs/supermarket_unseen_quality_v2_eval/contact_sheet_pred_good.jpg`
- `outputs/supermarket_unseen_quality_v2_eval/contact_sheet_pred_bad.jpg`
- `outputs/supermarket_unseen_quality_v2_eval/summary.txt`

---

## Confirmaciones

- NO se entrenó ningún modelo.
- NO se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- NO se modificó `best_model.pt`.
- NO se modificó `analyze_quality.py`.
- NO se modificó `quality_rules.yaml`.

---

## Siguiente paso

José debe abrir `outputs/supermarket_unseen_quality_v2_eval/contact_sheet_all.jpg`
y rellenar `human_review_template.csv` con `human_label` (GOOD / BAD / REVIEW / INVALID)
para identificar falsos positivos y falsos negativos del modelo V2 ante peras comerciales sanas.
