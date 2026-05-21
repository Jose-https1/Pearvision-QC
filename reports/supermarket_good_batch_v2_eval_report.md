# Supermarket Good Batch V2 — Evaluation Report

**Fecha:** 2026-05-20 22:31:53
**Lote:** `supermarket_good_batch_v2`
**Modelo:** `outputs/fruits360_quality_cls_v2/best_model.pt` (V2, congelado)
**Entrada:** `data/unseen_quality_eval_input/supermarket_good_batch_v2/` (22 imágenes)

---

## Resumen

| Métrica | Valor |
|---|---|
| Imágenes procesadas | 22 |
| Predichas GOOD | 15 |
| Predichas BAD | 7 |
| Confianza media GOOD | 0.7616 |
| Confianza media BAD | 0.6879 |
| Tiempo de inferencia | 2.43 s |

---

## Predicciones por imagen

| image_id | filename | pred_label | pred_confidence | prob_good | prob_bad |
|---|---|---|---|---|---|
| 1 | 1000060759.jpg | GOOD | 0.6852 | 0.6852 | 0.3148 |
| 2 | 1000060760.jpg | GOOD | 0.7793 | 0.7793 | 0.2207 |
| 3 | 1000060761.jpg | GOOD | 0.9492 | 0.9492 | 0.0508 |
| 4 | 1000060762.jpg | GOOD | 0.7539 | 0.7539 | 0.2461 |
| 5 | 1000060766.jpg | GOOD | 0.7806 | 0.7806 | 0.2194 |
| 6 | 1000060767.jpg | GOOD | 0.5960 | 0.5960 | 0.4040 |
| 7 | 1000060768.jpg | GOOD | 0.8819 | 0.8819 | 0.1181 |
| 8 | 1000060769.jpg | GOOD | 0.8117 | 0.8117 | 0.1883 |
| 9 | 1000060770.jpg | BAD | 0.5585 | 0.4415 | 0.5585 |
| 10 | 1000060771.jpg | BAD | 0.9224 | 0.0776 | 0.9224 |
| 11 | 1000060772.jpg | GOOD | 0.6399 | 0.6399 | 0.3601 |
| 12 | 1000060773.jpg | BAD | 0.8448 | 0.1552 | 0.8448 |
| 13 | 1000060774.jpg | BAD | 0.6763 | 0.3237 | 0.6763 |
| 14 | 1000060775.jpg | BAD | 0.6120 | 0.3880 | 0.6120 |
| 15 | 1000060776.jpg | GOOD | 0.6793 | 0.6793 | 0.3207 |
| 16 | 1000060777.jpg | GOOD | 0.8716 | 0.8716 | 0.1284 |
| 17 | 1000060779.jpg | BAD | 0.6287 | 0.3713 | 0.6287 |
| 18 | 1000060780.jpg | GOOD | 0.8886 | 0.8886 | 0.1114 |
| 19 | 1000060781.jpg | BAD | 0.5727 | 0.4273 | 0.5727 |
| 20 | 1000060782.jpg | GOOD | 0.7128 | 0.7128 | 0.2872 |
| 21 | 1000060783.jpg | GOOD | 0.7596 | 0.7596 | 0.2404 |
| 22 | 1000060784.jpg | GOOD | 0.6341 | 0.6341 | 0.3659 |

---

## Imágenes predichas BAD — requieren revisión humana

| filename | pred_confidence | prob_bad |
|---|---|---|
| `1000060770.jpg` | 0.5585 | prob_bad=0.5585 |
| `1000060771.jpg` | 0.9224 | prob_bad=0.9224 |
| `1000060773.jpg` | 0.8448 | prob_bad=0.8448 |
| `1000060774.jpg` | 0.6763 | prob_bad=0.6763 |
| `1000060775.jpg` | 0.6120 | prob_bad=0.6120 |
| `1000060779.jpg` | 0.6287 | prob_bad=0.6287 |
| `1000060781.jpg` | 0.5727 | prob_bad=0.5727 |

**Nota:** Estas imágenes deben abrirse manualmente para confirmar si son false BAD
(peras sanas con russeting/coloración irregular) o verdaderos defectos.

---

## Contexto

Segundo lote de peras comerciales de supermercado español.
Peras sanas con manchas marrones/russeting, partes verdes y variaciones de fondo/luz.
Las etiquetas humanas definitivas aún NO están registradas — este es solo el paso de evaluación.

---

## Archivos generados

- `outputs/supermarket_good_batch_v2_eval/predictions.csv`
- `outputs/supermarket_good_batch_v2_eval/human_review_template.csv`
- `outputs/supermarket_good_batch_v2_eval/contact_sheet_all.jpg`
- `outputs/supermarket_good_batch_v2_eval/contact_sheet_pred_good.jpg`
- `outputs/supermarket_good_batch_v2_eval/contact_sheet_pred_bad.jpg`
- `outputs/supermarket_good_batch_v2_eval/summary.txt`

---

## Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.

---

## Siguiente paso

José debe abrir `outputs/supermarket_good_batch_v2_eval/contact_sheet_all.jpg`
y confirmar para cada imagen si la predicción del modelo es correcta o no.
Etiquetar como GOOD, BAD, REVIEW o INVALID en `human_review_template.csv`.
