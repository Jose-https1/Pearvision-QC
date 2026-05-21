# Auditoría de Dataset — Bad Regression Validation

**Fecha:** 2026-05-21

---

## CSV usado

`data/quality_fruits360_human_v1/metadata/quality_fruits360_human_v1_master.csv`

**Motivo:** Es el CSV que coincide exactamente con los números de referencia del PROMPT (49 GOOD + 220 BAD = 269). Versión v2 tiene 48 GOOD + 219 BAD = 267.

## Columnas encontradas

`review_id, filename, human_label, class, split, original_split, original_class, source_path, image_copied`

## Distribución de etiquetas

| Etiqueta | N |
|---|---|
| GOOD | 49 |
| BAD | 220 |
| REVIEW | 0 (excluidas en este CSV — están en `excluded_review.csv`) |
| **Total** | **269** |

## REVIEW excluidas

31 imágenes están en `data/quality_fruits360_human_v1/metadata/excluded_review.csv` con `human_label=REVIEW`. No se usan en esta evaluación.

## Estructura de imágenes

| Split | Good | Bad | Total |
|---|---|---|---|
| train | 34 | 154 | 188 |
| val | 7 | 33 | 40 |
| test | 8 | 33 | 41 |
| **Total** | **49** | **220** | **269** |

Rutas: `data/quality_fruits360_human_v1/{split}/{class}/{filename}`

## Imágenes no localizadas

0 — todas las 269 imágenes están presentes en disco.

## Características del dataset

- Imágenes Fruits360: fondo blanco uniforme, pera centrada, ~100×100 px originales (ampliadas a 960×1280 en algunos casos)
- U3 fue entrenado sobre versiones ROI-masked-clean de este mismo dataset
- Contexto de evaluación: test de regresión para verificar que la fusión U3 corregida no crea falsos aceptados (BAD→PASA)
