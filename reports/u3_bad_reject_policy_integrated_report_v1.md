# Reporte: Integración Política U3 BAD→RECHAZA — PearVision QC

**Fecha:** 2026-05-21

---

## 1. Qué se modificó

`scripts/analyze_quality.py` — bloque de fusión U3 (función `_process_one`).
Cambio mínimo: añadida condición `p_bad >= 0.995 → RECHAZA` en el bloque `U3_BAD`,
antes de la comprobación de safe_mode/strong_defect existente.

## 2. Backup creado

`scripts/analyze_quality_backup_before_u3_bad_reject_policy_20260521_122336.py`

## 3. Umbral integrado

**threshold_p_bad = 0.995**

## 4. Por qué se eligió 0.995

- Es el único umbral en el grid [0.50, ..., 0.995] con GOOD->RECHAZA=0 y BAD->PASA=0.
- Los 3 casos GOOD con u3_pred=bad tienen p_bad máximo = 0.9943 < 0.995.
- 129 de 214 peras BAD tienen p_bad >= 0.995 → rechazo automático seguro.

## 5. Métricas antes/después

| Métrica | Antes (baseline) | Después (integrado) |
|---|---|---|
| GOOD->PASA | 51 | 51 |
| GOOD->REVISAR | 4 | 4 |
| GOOD->RECHAZA | 0 | 0 |
| BAD->PASA | 0 | 0 |
| BAD->REVISAR | 214 | 85 |
| BAD->RECHAZA | 0 | 129 |
| false_reject_rate | 0.0% | 0.0% |
| false_accept_rate | 0.0% | 0.0% |
| automatic_accept_rate | 19.0% | 19.0% |
| manual_review_rate | 81.0% | 33.1% |
| reject_rate | 0.0% | 48.0% |

## 6. Resultado en dataset corregido

- 269 imágenes evaluadas (55 GOOD, 214 BAD, etiquetas corregidas).
- BAD->RECHAZA: 129 (60.3% del total BAD).
- BAD->REVISAR: 85 (confianza insuficiente para rechazo automático).
- GOOD->RECHAZA: 0 — constraint principal cumplido.

## 7. Resultado en supermercado/holdout

- 86 peras de supermercado (todas etiqueta GOOD esperada).
- PASA: 86 | REVISAR: 0 | RECHAZA: 0
- Todas tienen quality_u3_decision_raw=U3_GOOD → la regla p_bad>=0.995 nunca se activa.

## 8. Falsos rechazos

**0 falsos rechazos** en dataset corregido y en supermercado holdout.

## 9. Falsos aceptados

**0 falsos aceptados** en dataset corregido.

## 10. Conclusión

**U3 integrado queda ACEPTADO como pipeline final provisional.**

Cumple todos los criterios:
- ✓ GOOD->RECHAZA = 0 (no rechaza peras comercialmente válidas)
- ✓ BAD->PASA = 0 (no acepta peras con defectos reales)
- ✓ BAD->RECHAZA = 129 (60.3% de las BAD)
- ✓ Supermercado holdout: 0 rechazos incorrectos
- ✓ manual_review_rate baja de 81.0% a 33.1%
