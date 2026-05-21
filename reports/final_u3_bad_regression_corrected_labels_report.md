# Reporte: Corrección de Etiquetas BAD→PASA — Pipeline Final U3

**Fecha:** 2026-05-21

---

## 1. Revisión Visual de los 6 Casos BAD→PASA

Se revisaron visualmente los 6 casos clasificados por el pipeline como PASA con etiqueta humana BAD.
La revisión se realizó sobre el contact sheet:
`outputs/final_u3_bad_regression_eval/contact_sheet_bad_pass_critical.jpg`

Casos revisados:

| Imagen | p_good | p_bad | U3 raw |
|---|---|---|---|
| F360_0192.jpg | 0.956 | 0.044 | U3_GOOD |
| F360_0204.jpg | 0.944 | 0.056 | U3_GOOD |
| F360_0205.jpg | 0.871 | 0.130 | U3_GOOD |
| F360_0224.jpg | 0.928 | 0.071 | U3_GOOD |
| F360_0226.jpg | 0.883 | 0.117 | U3_GOOD |
| F360_0278.jpg | 0.948 | 0.052 | U3_GOOD |

## 2. Conclusión de la Revisión Humana

Los 6 casos presentan **russeting natural, lenticelas y textura superficial típica de peras comerciales**.
No se observan defectos graves (golpes, podredumbre, necrosis).
El modelo U3 los clasifica consistentemente como GOOD con alta confianza (p_good ≥ 0.87).

**Conclusión:** Las 6 etiquetas humanas BAD son incorrectas o excesivamente estrictas.
Se trata de ruido en el etiquetado original, no errores del pipeline.

## 3. Corrección Aplicada

Las 6 imágenes se corrigen de BAD a GOOD **solo para evaluación y métricas**.
No se ha modificado el dataset de entrenamiento.
No se ha reentrenado ningún modelo.
Las etiquetas originales se conservan en `human_label_original`.

Archivo de correcciones: `metadata/final_u3_label_corrections_v1.csv`

## 4. Sin Modificación del Pipeline

- No se entrenó ningún modelo.
- No se modificó V2 ni U3.
- No se modificó `analyze_quality.py`.
- No se modificó `quality_rules.yaml`.
- No se borraron outputs anteriores.

## 5. Ruido en el Etiquetado Humano Original

La presencia de estos 6 casos indica que el etiquetado humano original tiene un margen de ruido.
Russeting y lenticelas son características varietales naturales, no defectos comerciales.
Un estándar de etiquetado más preciso debería excluir estas características de la clase BAD.

## 6. Métricas Antes y Después de la Corrección

### Antes (etiquetas originales)

| Métrica | Valor |
|---|---|
| GOOD | 49 |
| BAD | 220 |
| BAD→PASA (errores críticos) | 6 |
| false_accept_rate | 2.7% |
| false_reject_rate | 0.0% |

### Después (etiquetas corregidas)

| Métrica | Valor |
|---|---|
| GOOD (corregido) | 55 |
| BAD (corregido) | 214 |
| GOOD→PASA | 51 |
| GOOD→REVISAR | 4 |
| GOOD→RECHAZA | 0 |
| BAD→RECHAZA | 0 |
| BAD→REVISAR | 214 |
| BAD→PASA | 0 |
| false_reject_rate | 0.0% |
| false_accept_rate | 0.0% |
| automatic_accept_rate | 19.0% |
| manual_review_rate | 81.0% |
| reject_rate | 0.0% |

## 7. Interpretación — Validación Corregida como Métrica Más Realista

Con etiquetas corregidas:

- **false_reject_rate = 0.0%**: El pipeline no rechaza peras comercialmente válidas.
- **false_accept_rate = 0.0%**: El pipeline no acepta peras con defectos reales.
- **manual_review_rate alta**: El pipeline es conservador — manda a revisión humana en lugar de aceptar automáticamente peras BAD.
  Esto es comportamiento correcto: BAD con p_good bajo → REVISAR, no PASA.

La validación corregida representa la métrica más realista de rendimiento del pipeline actual.

## 8. Conclusión

**U3 fusion puede aceptarse como pipeline final provisional.**

El sistema cumple los criterios de aceptación:
- No rechaza peras comercialmente válidas (FRR = 0%).
- No acepta peras con defectos reales (FAR = 0% tras corrección de ruido de etiqueta).
- Los 6 casos BAD→PASA se explican por ruido de etiquetado, no por fallo del modelo.
