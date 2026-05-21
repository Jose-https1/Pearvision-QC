# Reporte: Validación de Regresión BAD — Pipeline U3 Fusion v2

**Fecha:** 2026-05-21
**Dataset:** quality_fruits360_human_v1 — 269 imágenes etiquetadas por humano

---

## Resultados

| Categoría | N | % |
|---|---|---|
| GOOD → PASA (OK) | 45 | 91.8% de GOOD |
| GOOD → REVISAR (conservador) | 4 | 8.2% de GOOD |
| GOOD → RECHAZA (**CRITICO**) | 0 | 0.0% de GOOD |
| BAD → RECHAZA (OK) | 0 | 0.0% de BAD |
| BAD → REVISAR (conservador) | 214 | 97.3% de BAD |
| BAD → PASA (**CRITICO**) | 6 | 2.7% de BAD |

## Métricas de negocio

| Métrica | Valor |
|---|---|
| false_reject_rate (GOOD→RECHAZA/GOOD) | 0.0% |
| false_accept_rate (BAD→PASA/BAD) | 2.7% |
| automatic_accept_rate (PASA/total) | 19.0% |
| manual_review_rate (REVISAR/total) | 81.0% |
| reject_rate (RECHAZA/total) | 0.0% |

---

## Falsos aceptados — BAD → PASA (6)

| Imagen | p_good | p_bad | U3_raw | Razón |
|---|---|---|---|---|
| F360_0192.jpg | 0.956 | 0.044 | U3_GOOD | U3_GOOD_STRONG_NO_DEFECT (p=0.956) |
| F360_0204.jpg | 0.944 | 0.056 | U3_GOOD | U3_GOOD_STRONG_NO_DEFECT (p=0.944) |
| F360_0205.jpg | 0.871 | 0.130 | U3_GOOD | U3_GOOD_STRONG_NO_DEFECT (p=0.870) |
| F360_0224.jpg | 0.928 | 0.071 | U3_GOOD | U3_GOOD_STRONG_NO_DEFECT (p=0.928) |
| F360_0226.jpg | 0.883 | 0.117 | U3_GOOD | U3_GOOD_STRONG_NO_DEFECT (p=0.883) |
| F360_0278.jpg | 0.948 | 0.052 | U3_GOOD | U3_GOOD_STRONG_NO_DEFECT (p=0.948) |

## Falsos rechazos — GOOD → RECHAZA (0)

| Imagen | p_good | p_bad | Razón |
|---|---|---|---|
_ninguno_

---

## Interpretación (TAREA 6)

### 1. ¿El fix U3 ha eliminado falsos rechazos de peras sanas?

**Resultado anterior en supermercado:** 86/86 PASA (0 RECHAZA, 0 REVISAR).
**Resultado en dataset humano GOOD (49 imgs):** 45 PASA + 4 REVISAR + 0 RECHAZA.
FRR = 0.0%

→ Sí, los falsos rechazos de GOOD son mínimos o nulos.

### 2. ¿Aparecen falsos aceptados BAD→PASA?

→ Sí: 6 BAD peras son incorrectamente aceptadas como PASA (2.7%).

### 3. ¿Está demasiado permisivo el pipeline?

→ La permisividad está dentro de rango tolerable para un prototipo académico.

### 4. ¿Está demasiado conservador?

→ Sí. La tasa de revisión manual es 81.0%, lo que significa que 218 imágenes quedan pendientes de revisión humana.

### 5. ¿Se puede integrar U3 como versión final provisional?

**SI** — ACEPTAR CON MONITOREO (FAR <= 5%, FRR = 0%)

**Observación clave:** El modo `U3_BAD + no strong_defect → REVISAR` (en lugar de RECHAZA directo) hace que la mayoría de peras BAD correctamente identificadas por U3 queden en REVISAR en vez de RECHAZA. Esto es conservador pero correcto para un prototipo sin detector de defectos confirmado. Para producción real se necesitaría activar el YOLO de defectos (`--use-defect-model`) o reducir el umbral de `strong_defect_evidence`.

---

## Archivos generados

- `outputs/final_u3_bad_regression_eval/results_final_u3_bad_regression.csv`
- `outputs/final_u3_bad_regression_eval/contact_sheet_all.jpg`
- `outputs/final_u3_bad_regression_eval/contact_sheet_bad_pass_critical.jpg`
- `outputs/final_u3_bad_regression_eval/contact_sheet_good_reject_critical.jpg`
- `outputs/final_u3_bad_regression_eval/contact_sheet_review_cases.jpg`
- `outputs/final_u3_bad_regression_eval/summary.txt`
