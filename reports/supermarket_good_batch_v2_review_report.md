# Supermarket Good Batch V2 — Review Report

**Fecha:** 2026-05-20 22:41:00
**Lote:** `supermarket_spain_unseen_batch_v2`
**Modelo evaluado:** `outputs/fruits360_quality_cls_v2/best_model.pt` (V2, congelado)

---

## 1. Descripción del lote

- **22 fotos reales** de peras compradas en supermercado en España.
- Peras **comercialmente sanas**, aptas para venta.
- Características naturales presentes: manchas marrones, russeting, partes verdes,
  coloración irregular.
- **Variaciones de fondo en este lote:**
  - Fondo blanco/claro (8 imágenes)
  - Fondo azul / carpeta azul (6 imágenes)
  - Fondo negro/texturizado (8 imágenes)
- Verdad humana para todas: **GOOD**.

---

## 2. Resultado bruto del modelo V2

| Predicción | Cantidad | Confianza media |
|---|---|---|
| GOOD | 15 | 0.7616 |
| BAD  | 7  | 0.6879  |

- **Tasa de error bruta en este lote: 7/22 = 31%**
- Comparación con lote V1 (fondo más uniforme): 1/20 = 5%

---

## 3. Falsos BAD confirmados (7)

| filename | pred_confidence | background_group | gravedad del error |
|---|---|---|---|
| `1000060770.jpg` | 0.5585 | blue | Baja — zona de incertidumbre |
| `1000060771.jpg` | 0.9224 | blue | Alta — problema de dominio |
| `1000060773.jpg` | 0.8448 | blue | Alta — problema de dominio |
| `1000060774.jpg` | 0.6763 | blue | Baja — zona de incertidumbre |
| `1000060775.jpg` | 0.6120 | blue | Baja — zona de incertidumbre |
| `1000060779.jpg` | 0.6287 | black_textured | Baja — zona de incertidumbre |
| `1000060781.jpg` | 0.5727 | black_textured | Baja — zona de incertidumbre |

---

## 4. Diagnóstico por grupo de fondo

| background_group | total | pred_good | pred_bad | false_bad | false_bad_rate | conf_media_good | conf_media_bad |
|---|---|---|---|---|---|---|---|
| white_light | 8 | 8 | 0 | 0 | 0.0% | 0.7797 | N/A |
| blue | 6 | 1 | 5 | 5 | 83.3% | 0.6399 | 0.7228 |
| black_textured | 8 | 6 | 2 | 2 | 25.0% | 0.7577 | 0.6007 |


**Conclusión del diagnóstico:**

- **Fondo blanco/claro:** 0 falsos BAD. El modelo V2 es estable en este dominio.
- **Fondo azul:** 5/6 imágenes mal clasificadas (83%). El fondo azul interfiere
  con las características aprendidas por V2 (entrenado principalmente con Fruits-360,
  que usa fondo blanco uniforme).
- **Fondo negro/texturizado:** 2/8 imágenes mal clasificadas (25%). Mejor que azul
  pero aún problemático.
- **El problema es cambio de dominio visual (domain shift), no defectos reales.**

---

## 5. Predicciones completas

| id | filename | pred | conf | fondo | false_bad |
|---|---|---|---|---|---|
| 1 | 1000060759.jpg | GOOD | 0.6852 | white_light |  |
| 2 | 1000060760.jpg | GOOD | 0.7793 | white_light |  |
| 3 | 1000060761.jpg | GOOD | 0.9492 | white_light |  |
| 4 | 1000060762.jpg | GOOD | 0.7539 | white_light |  |
| 5 | 1000060766.jpg | GOOD | 0.7806 | white_light |  |
| 6 | 1000060767.jpg | GOOD | 0.5960 | white_light |  |
| 7 | 1000060768.jpg | GOOD | 0.8819 | white_light |  |
| 8 | 1000060769.jpg | GOOD | 0.8117 | white_light |  |
| 9 | 1000060770.jpg | BAD | 0.5585 | blue | TRUE |
| 10 | 1000060771.jpg | BAD | 0.9224 | blue | TRUE |
| 11 | 1000060772.jpg | GOOD | 0.6399 | blue |  |
| 12 | 1000060773.jpg | BAD | 0.8448 | blue | TRUE |
| 13 | 1000060774.jpg | BAD | 0.6763 | blue | TRUE |
| 14 | 1000060775.jpg | BAD | 0.6120 | blue | TRUE |
| 15 | 1000060776.jpg | GOOD | 0.6793 | black_textured |  |
| 16 | 1000060777.jpg | GOOD | 0.8716 | black_textured |  |
| 17 | 1000060779.jpg | BAD | 0.6287 | black_textured | TRUE |
| 18 | 1000060780.jpg | GOOD | 0.8886 | black_textured |  |
| 19 | 1000060781.jpg | BAD | 0.5727 | black_textured | TRUE |
| 20 | 1000060782.jpg | GOOD | 0.7128 | black_textured |  |
| 21 | 1000060783.jpg | GOOD | 0.7596 | black_textured |  |
| 22 | 1000060784.jpg | GOOD | 0.6341 | black_textured |  |

---

## 6. Decisiones tomadas

1. **Registrar las 22 imágenes como hard examples GOOD** para futura V3:
   - `data/supermarket_good_hard_examples_v2/` (22 imágenes)
   - Junto con `data/supermarket_good_hard_examples_v1/` (20 imágenes) = 42 hard examples GOOD acumulados.
2. **No entrenar todavía.** 42 hard examples no son suficientes para reentrenar con garantías.
3. **Usar fondo blanco/gris/beige como condición operativa recomendada** para la demo actual.
4. **Aplicar regla operativa de umbral:** BAD con conf < 0.70 → REVIEW (no rechazo automático).
5. **Evaluar V3** cuando se acumulen al menos 100-150 hard examples GOOD con fondos variados,
   o cuando se implemente una máscara de segmentación de pera previa al clasificador.

---

## 7. Archivos generados

| Archivo | Descripción |
|---|---|
| `outputs/supermarket_good_batch_v2_eval/human_review_completed.csv` | Revisión humana con todas las 22 anotadas como GOOD |
| `outputs/supermarket_good_batch_v2_eval/human_error_review.csv` | Análisis de errores con grupo de fondo |
| `outputs/supermarket_good_batch_v2_eval/background_error_analysis.csv` | Estadísticas por grupo de fondo |
| `outputs/supermarket_good_batch_v2_eval/capture_conditions_recommendation.txt` | Guía de condiciones de captura |
| `data/supermarket_good_hard_examples_v2/labels.csv` | Etiquetas hard examples |
| `data/supermarket_good_hard_examples_v2/images/` | 22 imágenes copiadas |

---

## 8. Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.

---

## 9. Siguiente paso recomendado

Hacer una evaluación comparativa usando solo condiciones válidas de captura
(fondo blanco/gris/beige mate). Después decidir si entrenar V3 con los
hard examples GOOD acumulados en los dos lotes (42 imágenes en total).
