# Supermarket Good Batch V1 — Review Report

**Fecha:** 2026-05-20 22:08:42
**Lote:** `supermarket_spain_unseen_batch_v1`
**Modelo evaluado:** `outputs/fruits360_quality_cls_v2/best_model.pt` (V2, congelado)

---

## 1. Descripción del lote

- **20 fotos reales** de peras compradas en supermercado en España.
- Peras **comercialmente sanas**, aptas para venta.
- Características naturales presentes: manchas marrones, russeting, zonas verdes, coloración irregular.
- **Ninguna** de las 20 peras tiene defectos graves (golpes, podredumbre, marcas de ramita).
- Verdad humana para todas: **GOOD**.

---

## 2. Resultado bruto del modelo V2

| Predicción | Cantidad | Confianza media |
|---|---|---|
| GOOD | 19 | 0.8481 |
| BAD  | 1  | 0.5275 |

- Precisión bruta del modelo sobre este lote: **19/20 = 95%**

---

## 3. Error detectado

| Imagen | Predicción modelo | Confianza | Verdad humana | Tipo de error |
|---|---|---|---|---|
| `1000060747.jpg` | BAD | 0.5275 | GOOD | FALSE_BAD_LOW_CONFIDENCE |

- El modelo predijo BAD con confianza muy baja (**0.5275**, apenas por encima de 0.50).
- La imagen corresponde a una pera sana con russeting/manchas marrones naturales.
- Este tipo de fallo es esperado en V2: el modelo aún no tiene suficiente exposición a peras
  sanas con coloración irregular natural de supermercado español.

---

## 4. Interpretación

**V2 se comporta bien frente a russeting natural:**
- 19 de 20 peras sanas clasificadas correctamente como GOOD.
- Confianza media en GOOD: alta (≥ 0.75 en todos los casos correctos).

**El único fallo no debe causar rechazo automático:**
- La predicción BAD tiene confianza de 0.5275, casi aleatoria.
- Una regla operativa simple (BAD con conf < 0.70 → REVIEW) evita el rechazo indebido.
- Con esta regla, el resultado operativo para este lote sería: 19 GOOD, 1 REVIEW, 0 BAD.

---

## 5. Análisis de umbral operativo

Regla propuesta: `BAD conf < 0.7 → REVIEW`

| Decisión operativa | Cantidad | Interpretación |
|---|---|---|
| GOOD   | 19  | Aceptadas directamente |
| REVIEW | 1 | Enviadas a revisión humana (no rechazadas) |
| BAD    | 0   | Rechazadas automáticamente |

Con esta regla, **0 peras sanas serían rechazadas automáticamente** en este lote.

---

## 6. Recomendaciones

1. **Conservar V2 como baseline.** Su rendimiento en peras sanas de supermercado es aceptable.
2. **Aplicar zona de incertidumbre operativa:** BAD con confianza < 0.70 → REVIEW (no rechazo).
3. **Usar estas 20 imágenes como hard examples GOOD para V3:**
   - Ruta: `data/supermarket_good_hard_examples_v1/`
   - Especialmente `1000060747.jpg` como ejemplo de falso BAD con russeting natural.
4. **Acumular más lotes reales** antes de entrenar V3. Un solo lote de 20 imágenes
   no es suficiente para reentrenar — acumular al menos 3-5 lotes similares.

---

## 7. Archivos generados

| Archivo | Descripción |
|---|---|
| `outputs/supermarket_unseen_quality_v2_eval/human_review_completed.csv` | Revisión humana completa |
| `outputs/supermarket_unseen_quality_v2_eval/human_error_review.csv` | Análisis de errores del modelo |
| `outputs/supermarket_unseen_quality_v2_eval/operational_threshold_analysis.csv` | Análisis umbral operativo |
| `data/supermarket_good_hard_examples_v1/labels.csv` | Etiquetas hard examples |
| `data/supermarket_good_hard_examples_v1/images/` | 20 imágenes copiadas |

---

## 8. Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.

---

## 9. Siguiente paso

Hacer una segunda prueba con otro lote real de peras sanas y, si aparecen más falsos BAD,
acumularlos junto con este lote para entrenar V3.
