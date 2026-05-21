# Reporte: Calibración Política U3 BAD→RECHAZA — PearVision QC

**Fecha:** 2026-05-21

---

## 1. BAD que pasan de REVISAR a RECHAZA

Con threshold_p_bad = 0.995:

- BAD→RECHAZA: **129** (antes: 0)
- BAD→REVISAR: **85** (antes: 214)
- Reducción de REVISAR: 129 casos resueltos automáticamente.

## 2. GOOD→RECHAZA (falsos rechazos)

**GOOD→RECHAZA = 0.** No aparece ningún falso rechazo de pera buena.

## 3. BAD→PASA (falsos aceptados)

**BAD→PASA = 0.** No aparece ningún falso aceptado.

## 4. Umbral seleccionado

**threshold_p_bad = 0.995**

Este umbral se seleccionó porque es el más agresivo que cumple:
- GOOD→RECHAZA = 0 (prioridad máxima)
- BAD→PASA = 0
- BAD→RECHAZA maximizado

El valor de 0.995 está justificado por la distribución de los casos GOOD→REVISAR:
los 3 casos con u3_pred=bad tienen p_bad máximo de 0.9943, por lo que cualquier umbral ≤ 0.994 causaría falsos rechazos.

## 5. Casos restantes en REVISAR

Con threshold=0.995, quedan **89 casos** en REVISAR:

- GOOD→REVISAR: 4 — peras buenas con U3 ambiguo (u3_pred=bad pero p_bad < 0.995, o p_good < 0.85)
- BAD→REVISAR: 85 — peras malas con p_bad < 0.995 (confianza insuficiente para rechazo automático)

Estos casos requieren revisión humana — es el comportamiento correcto del sistema conservador.

## 6. Recomendación de integración

**La política PUEDE integrarse provisionalmente en el pipeline.**

Cumple todos los criterios de aceptación:
- ✓ GOOD→RECHAZA = 0
- ✓ BAD→PASA = 0
- ✓ BAD→RECHAZA = 129 (aumenta claramente desde 0)
- ✓ BAD→REVISAR = 85 (baja claramente desde 214)
## 7. Revisión visual recomendada

Los siguientes 3 casos GOOD con u3_pred=bad requieren revisión visual prioritaria:

| Imagen | p_bad | Nota |
|---|---|---|
| F360_0018.jpg | 0.8445 | U3 dice BAD con confianza media-alta; p_bad < 0.995 → REVISAR |
| F360_0048.jpg | 0.9754 | U3 dice BAD con confianza muy alta; posible ruido de etiqueta |
| F360_0060.jpg | 0.9943 | U3 dice BAD con confianza máxima; posible ruido de etiqueta |

F360_0048 y F360_0060 tienen p_bad > 0.97 con etiqueta GOOD: son candidatos a revisión/corrección de etiqueta.

## 8. Comparativa de métricas

| Métrica | Antes (baseline corregido) | Después (policy candidata) |
|---|---|---|
| GOOD→PASA | 51 | 51 |
| GOOD→REVISAR | 4 | 4 |
| GOOD→RECHAZA | 0 | 0 |
| BAD→PASA | 0 | 0 |
| BAD→REVISAR | 214 | 85 |
| BAD→RECHAZA | 0 | 129 |
| false_reject_rate | 0.0% | 0.0% |
| false_accept_rate | 0.0% | 0.0% |
| automatic_accept_rate | 19.0% | 19.0% |
| manual_review_rate | 81.0% | 33.1% |
| reject_rate | 0.0% | 48.0% |

## 9. Conclusión

Con threshold_p_bad = 0.995:

- 129 peras BAD se rechazan automáticamente (antes ninguna).
- Ninguna pera GOOD se rechaza incorrectamente.
- La tasa de revisión manual baja de 81.0% a 33.1%.
- La política es segura y puede integrarse como siguiente paso en analyze_quality.py.
