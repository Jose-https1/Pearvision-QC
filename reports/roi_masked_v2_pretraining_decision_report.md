# ROI Masked V2 — Informe de Decisión Pre-entrenamiento U3

**Fecha:** 2026-05-21
**Script ejecutado:** `scripts/prepare_quality_roi_masked_previews_v2.py`
**Evaluador:** `scripts/evaluate_v2_on_roi_masked_clean.py`
**Modelo evaluado:** `outputs/fruits360_quality_cls_v2/best_model.pt` (MobileNetV3-small, sin modificar)

---

## 1. ¿Mejora V2-masked frente a V2-original?

### Resultados comparativos

| Métrica | V2 original (imagen sin máscarar) | V2 en gray_bg_clean | Mejora |
|---|---|---|---|
| GOOD correcto | ~17% (batch_v2) / ~75% (batch_v3) | 89.1% (57/64) | **+72pp / +14pp** |
| REVIEW | ~10% | 10.9% | similar |
| BAD (falso positivo) | 83% (fondo azul) / 25% (fondo negro) | **0%** | **-83pp / -25pp** |
| Tasa error total (BAD+REVIEW) | hasta 90%+ fondo azul | **10.9%** | fuerte mejora |

**Conclusión:** V2 sobre imágenes `gray_bg_clean` elimina completamente las decisiones BAD falsas. La tasa de error residual (10.9% REVIEW) corresponde a 7 peras de las 64, y en ningún caso V2 tomó una decisión BAD incorrecta.

Los 7 casos REVIEW (bad_conf entre 0.50 y 0.69) son:
- 1000060747 (bad=0.6163)
- 1000060760 (bad=0.5699)
- 1000060762 (bad=0.5006)
- 1000060772 (bad=0.5156)
- 1000060774 (bad=0.6918) — caso más marginal
- 1000060796 (bad=0.5264)
- 1000060812 (bad=0.5090)

Estos son casos de peras buenas en los que V2 tiene incertidumbre pero no llega al umbral de rechazo (0.70). Con U3 entrenado sobre ROI/masked, estos deberían mejorar.

---

## 2. ¿Son las máscaras V2 suficientemente limpias para entrenar U3?

### Estadísticas del pipeline V2 (64 imágenes)

| Métrica | Valor |
|---|---|
| Imágenes procesadas | 64 / 64 |
| OK (cambio de área < 35%) | 64 |
| REVIEW (cambio grande) | 0 |
| FAIL (máscara inválida) | 0 |
| Cambio de área medio (V1 vs V2 limpia) | ~1.0–1.2% (muy conservador) |
| Cambio máximo observado | 2.91% (1000060794) |

El cambio de área es mínimo (~1%), lo que indica que la limpieza de máscara (Método C) no está cortando la pera agresivamente. Los componentes de sombra/borde eliminados son pequeños y no afectan al cuerpo principal.

**Evaluación cualitativa:**
- Método A (GrabCut + LCC + fill_holes): cierra huecos, descarta islas aisladas.
- Método B (A + limpieza de sombra por similitud de fondo LAB): elimina píxeles borde similares al fondo.
- Método C (B + erosión conservadora 2px): suaviza el contorno, evita halos.

Las imágenes `gray_bg_clean.jpg` muestran la pera sobre fondo gris neutro (128,128,128), lo que neutraliza el efecto del fondo original (blanco, azul, negro).

**Veredicto:** Las máscaras V2 son suficientemente limpias para iniciar el entrenamiento de U3.

---

## 3. ¿Con qué fondo entrenar U3?

| Opción | Ventajas | Inconvenientes |
|---|---|---|
| `gray_bg_clean` solo | Fondo neutro, reduce sobreajuste al blanco. Probado en evaluación. | Todas las imágenes tienen el mismo fondo → posible sobreajuste a gris. |
| `white_bg_clean` solo | Coherente con Fruits-360 (dominio origen). | No neutraliza el problema de fondo original. |
| Ambas versiones | Doble cobertura de dominio, más robusto. Augmentation natural. | Doble el dataset; mayor riesgo de imbalance si se mezcla con Fruits-360. |

**Recomendación:** Entrenar U3 con `gray_bg_clean` como conjunto principal. Opcionalmente añadir `white_bg_clean` como augmentation de dominio para mejorar la generalización.

Razón: la evaluación demostró que V2 en `gray_bg_clean` reduce el error a 10.9% sin reentrenar. Un modelo U3 entrenado sobre este dominio debería corregir los 7 casos REVIEW restantes.

---

## 4. Recomendación exacta

**Decisión: ENTRENAR U3 YA.**

Condiciones cumplidas:
- [x] 64 imágenes GOOD con fondo neutralizado (`gray_bg_clean`).
- [x] 42 imágenes GOOD originales (hard examples batch_v1 + batch_v2) como base de entrenamiento.
- [x] Evaluación confirma que V2 en imágenes enmascaradas no produce falsos BAD.
- [x] Máscaras limpias: 0 FAIL, 0 REVIEW en la limpieza.
- [x] Cambio de área medio < 2% — la pera no está siendo cortada.

**Pasos sugeridos para U3:**
1. Combinar `gray_bg_clean` (64) con Fruits-360 good (submuestreado para balance).
2. Incluir hard examples V1+V2 como GOOD negatives adicionales.
3. Aplicar augmentation: flip horizontal, rotación ±15°, jitter de color suave.
4. Mantener misma arquitectura MobileNetV3-small para comparabilidad.
5. No modificar V2 hasta confirmar que U3 supera a V2 en el conjunto de validación.

---

## 5. Confirmaciones de integridad del proceso

- **No se entrenó ningún modelo.** Solo se ejecutó inferencia con `best_model.pt`.
- **No se modificó V2.** El archivo `outputs/fruits360_quality_cls_v2/best_model.pt` no fue tocado.
- **No se modificó `analyze_quality.py`.** El script de análisis principal no fue alterado.
- **No se modificó `quality_rules.yaml`.** Las reglas de calidad permanecen intactas.
- **No se borraron outputs anteriores.** La carpeta `outputs/quality_roi_masked_previews/` conserva todos los crops y contact sheets de V1.

---

## 6. Archivos generados en esta sesión

| Archivo | Descripción |
|---|---|
| `outputs/quality_roi_masked_previews_v2/crops/*_original.jpg` | 64 crops originales 224×224 |
| `outputs/quality_roi_masked_previews_v2/crops/*_mask_v1_like.jpg` | Máscaras estilo V1 (GrabCut básico) |
| `outputs/quality_roi_masked_previews_v2/crops/*_mask_clean.jpg` | Máscaras limpias (Método C) |
| `outputs/quality_roi_masked_previews_v2/crops/*_gray_bg_clean.jpg` | Pera sobre fondo gris neutro |
| `outputs/quality_roi_masked_previews_v2/crops/*_white_bg_clean.jpg` | Pera sobre fondo blanco |
| `outputs/quality_roi_masked_previews_v2/crops/*_transparent_debug.png` | Pera con canal alfa (debug) |
| `outputs/quality_roi_masked_previews_v2/crops/*_comparison.jpg` | Comparación por imagen |
| `outputs/quality_roi_masked_previews_v2/contact_sheet_v1_vs_v2.jpg` | Contact sheet comparativo (64 filas) |
| `outputs/quality_roi_masked_previews_v2/problem_cases_v2_grid.jpg` | Grid 11 casos problemáticos |
| `outputs/quality_roi_masked_previews_v2/roi_masked_v2_diagnostics.csv` | CSV diagnóstico (OK=64) |
| `outputs/quality_roi_masked_previews_v2/v2_on_gray_bg_clean_predictions.csv` | Predicciones V2 en gray_bg_clean |
| `outputs/quality_roi_masked_previews_v2/v2_on_gray_bg_clean_contact_sheet.jpg` | Contact sheet de evaluación |
| `reports/roi_masked_v1_limitations_report.md` | Análisis de limitaciones V1 |
| `reports/roi_masked_v2_pretraining_decision_report.md` | Este reporte |
