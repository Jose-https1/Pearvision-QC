# PearVision QC — Checklist Final de Validación v1

**Fecha:** 2026-05-21
**Pipeline:** U3 ROI masked clean + política BAD→RECHAZA integrada

---

## Estado general

| # | Item | Estado |
|---|---|---|
| 1 | Modelo entrenado (U3 ROI masked clean) | ✅ COMPLETADO |
| 2 | Pipeline integrado en analyze_quality.py | ✅ COMPLETADO |
| 3 | Backup creado antes de integración | ✅ COMPLETADO |
| 4 | Test dataset corregido (269 imágenes) pasado | ✅ COMPLETADO |
| 5 | Test supermercado/holdout (86 peras) pasado | ✅ COMPLETADO |
| 6 | Contact sheets generados | ✅ COMPLETADO |
| 7 | Falsos rechazos GOOD = 0 | ✅ VERIFICADO |
| 8 | Falsos aceptados BAD = 0 (tras corrección humana) | ✅ VERIFICADO |
| 9 | Limitaciones documentadas | ✅ DOCUMENTADO |
| 10 | Próximo test recomendado definido | ✅ DEFINIDO |

---

## Detalle por ítem

### 1. Modelo entrenado
- **Modelo:** U3 — clasificador binario GOOD/BAD sobre ROI con fondo gris neutro (gray_bg_clean).
- **Arquitectura:** EfficientNet-B0 fine-tuned.
- **Accuracy test:** 91.84%.
- **Holdout 22 imágenes:** 22/22 correctas (100%).

### 2. Pipeline integrado
- **Script:** `scripts/analyze_quality.py`
- **Cambio:** bloque U3 fusion actualizado con regla `p_bad >= 0.995 → RECHAZA`.
- **Integración mínima:** sin cambios en segmentación, reglas base ni YOLO.

### 3. Backup creado
- **Archivo:** `scripts/analyze_quality_backup_before_u3_bad_reject_policy_20260521_122336.py`
- **Fecha:** 2026-05-21

### 4. Test dataset corregido
- **Total evaluado:** 269 imágenes (55 GOOD + 214 BAD, etiquetas corregidas por humano).
- **Resultados:**
  - GOOD→PASA: 51 | GOOD→REVISAR: 4 | GOOD→RECHAZA: 0
  - BAD→PASA: 0 | BAD→REVISAR: 85 | BAD→RECHAZA: 129
- **Artefacto:** `outputs/u3_bad_reject_policy_integrated_eval_v1/summary.txt`

### 5. Test supermercado/holdout
- **Total:** 86 peras reales de supermercado.
- **Resultados:** PASA: 86 | REVISAR: 0 | RECHAZA: 0
- **Artefacto:** `outputs/u3_bad_reject_policy_integrated_eval_v1/supermarket_holdout/summary_supermarket.txt`

### 6. Contact sheets generados
| Archivo | Estado |
|---|---|
| `contact_sheet_all_integrated.jpg` | ✅ Existe |
| `contact_sheet_reject_integrated.jpg` | ✅ Existe |
| `contact_sheet_review_integrated.jpg` | ✅ Existe |
| `contact_sheet_false_reject_good.jpg` | ✅ Existe |
| `contact_sheet_false_accept_bad.jpg` | ✅ Existe |
| `supermarket_holdout/contact_sheet_supermarket_all_integrated.jpg` | ✅ Existe |
| `supermarket_holdout/contact_sheet_supermarket_review_reject_integrated.jpg` | ✅ Existe |

### 7. Falsos rechazos GOOD = 0
- **FRR:** 0.0% en dataset corregido.
- **FRR:** 0.0% en supermercado holdout.
- **Verificado:** ninguna pera GOOD enviada a RECHAZA en ningún conjunto evaluado.

### 8. Falsos aceptados BAD = 0
- **FAR:** 0.0% tras corrección humana de etiquetas.
- **Nota:** 6 casos inicialmente etiquetados como BAD resultaron ser peras de calidad supermercado; al corregir las etiquetas, FAR = 0.0%.

### 9. Limitaciones documentadas
- Dataset pequeño (269 + 86).
- Peras BAD provienen mayormente de Fruits-360, no de defectos reales de campo.
- No se han probado peras malas reales de supermercado.
- El umbral 0.995 es específico del modelo U3 actual; requiere recalibración si se reentrena.
- Sistema no validado industrialmente.
- Ver documento completo: `reports/pearvision_qc_final_pipeline_summary_v1.md`

### 10. Próximo test recomendado
- Probar una carpeta nueva de imágenes reales no vistas sin reentrenar nada.
- Idealmente incluir peras defectuosas reales de supermercado o mercado local.
- Registrar resultados y documentar casos dudosos para futura curación.

---

## Decisión final

```
PIPELINE FINAL PROVISIONAL: ACEPTADO
Clasificador: U3 ROI masked clean
Umbral GOOD: p_good > 0.85 → PASA
Umbral BAD:  p_bad >= 0.995 → RECHAZA
Duda/error:  → REVISAR
FRR = 0.0% | FAR = 0.0% | MRR = 33.1% | RR = 48.0%
```

---

*Checklist de validación final PearVision QC v1 — 2026-05-21*
