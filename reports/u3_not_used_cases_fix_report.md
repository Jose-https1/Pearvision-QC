# Reporte: Corrección casos U3_NOT_USED — u3_not_used_cases_fix_report

**Fecha:** 2026-05-21
**Base:** `outputs/u3_fusion_fixed_eval/resultados_u3_fusion_fixed.csv` (86 filas, v1)

---

## Antes

| Decisión | N |
|---|---|
| PASA | 80 |
| REVISAR | 6 |
| RECHAZA | 0 |
| **Total** | **86** |

## Después

| Decisión | N |
|---|---|
| PASA | 86 |
| REVISAR | 0 |
| RECHAZA | 0 |
| **Total** | **86** |

---

## Resultados por imagen recalculada

| Imagen | U3 status | p_good | p_bad | Decisión | Razón |
|---|---|---|---|---|---|
| 1000060736.jpg | OK | 0.993 | 0.007 | PASA | U3_GOOD_STRONG (p_good=0.993) |
| 1000060738.jpg | OK | 0.989 | 0.011 | PASA | U3_GOOD_STRONG (p_good=0.989) |
| 1000060739.jpg | OK | 0.988 | 0.012 | PASA | U3_GOOD_STRONG (p_good=0.988) |
| 1000060740.jpg | OK | 0.968 | 0.032 | PASA | U3_GOOD_STRONG (p_good=0.968) |
| 1000060741.jpg | OK | 0.996 | 0.004 | PASA | U3_GOOD_STRONG (p_good=0.996) |
| 1000060745.jpg | OK | 0.998 | 0.002 | PASA | U3_GOOD_STRONG (p_good=0.998) |

---

## Validación (TAREA 5)

- Total = 86 (OK)
- RECHAZA = 0 (OK)
- Las 6 imágenes recalculadas tienen p_good real (ya no g=0.00 b=0.00): OK

---

## Qué NO se modificó

- Modelo U3 sin cambios
- quality_rules.yaml sin cambios
- V2 sin cambios
- Outputs anteriores sin borrar

---

## Archivos generados

| Archivo | Estado |
|---|---|
| `outputs/u3_not_used_cases_eval/not_used_cases_predictions.csv` | generado |
| `outputs/u3_not_used_cases_eval/contact_sheet_not_used_cases.jpg` | generado |
| `outputs/u3_not_used_cases_eval/summary.txt` | generado |
| `outputs/u3_fusion_fixed_eval_v2/resultados_u3_fusion_fixed_v2.csv` | generado |
| `outputs/u3_fusion_fixed_eval_v2/contact_sheet_u3_fusion_fixed_v2_all.jpg` | generado |
| `outputs/u3_fusion_fixed_eval_v2/contact_sheet_u3_fusion_fixed_v2_review_bad.jpg` | generado |
| `outputs/u3_fusion_fixed_eval_v2/summary.txt` | generado |
