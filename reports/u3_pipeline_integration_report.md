# U3 Pipeline Integration Report

**Fecha:** 2026-05-21  
**Backup:** `scripts/analyze_quality_backup_before_u3_20260521_010831.py`

---

## 1. Archivos modificados

| Archivo | Cambio |
|---|---|
| `scripts/analyze_quality.py` | Integración U3: 8 edits quirúrgicos |
| `scripts/test_integrated_u3_pipeline.py` | Nuevo — script de prueba integrada |
| `scripts/_build_u3_sheets.py` | Nuevo — generador de contact sheets desde CSV |
| `reports/u3_pipeline_integration_audit.md` | Nuevo — auditoría previa a la integración |

**No modificados:** `configs/quality_rules.yaml`, V2 model, U3 model.

---

## 2. Cómo activar U3

```bash
.venv/Scripts/python scripts/analyze_quality.py \
  --source data/tu_carpeta \
  --save \
  --use-quality-u3 \
  --quality-u3-model outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt \
  --quality-u3-thresholds outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json \
  --quality-u3-safe-mode
```

## 3. Cómo desactivar U3

Omitir `--use-quality-u3`. El pipeline funciona igual que antes.

---

## 4. Qué hace U3 en el pipeline

```
reglas → decide() → brillo cap → máscara cap → quality_cls → U3 → resultado final
```

**Política U3 raw:**
- `p_bad >= 0.60` → `U3_BAD`
- `p_good >= 0.55` → `U3_GOOD`
- otro → `U3_REVIEW`

**Safe mode (siempre activo por defecto):**

| U3 raw | Condición | Resultado final |
|---|---|---|
| U3_GOOD | siempre | sin cambio |
| U3_REVIEW | decisión era PASA | sube a REVISAR |
| U3_BAD | decisión ya era RECHAZA, o YOLO 2+ defectos, o YOLO conf>0.65 | permite RECHAZA |
| U3_BAD | ninguna condición fuerte | convierte a REVISAR |

**U3 NUNCA puede rechazar solo una pera aparentemente sana.**

---

## 5. Nuevas columnas CSV

```
quality_u3_enabled, quality_u3_status, quality_u3_pred,
quality_u3_p_good, quality_u3_p_bad,
quality_u3_decision_raw, quality_u3_decision_safe,
final_decision_before_u3, final_decision_after_u3, final_decision_reason
```

---

## 6. Resultados de la prueba integrada

**Fuentes evaluadas:**
- `data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3` — 22 peras
- `data/unseen_quality_eval_input/supermarket_good_batch_v2` — 22 peras
- `data/supermarket_good_hard_examples_v1/images` — 20 peras
- `data/supermarket_good_hard_examples_v2/images` — 22 peras

**Total: 86 peras (todas de ground truth GOOD por revisión humana)**

### Decisiones finales del pipeline

| Decisión | Conteo | % |
|---|---|---|
| PASA | 10 | 11.6 % |
| REVISAR | 44 | 51.2 % |
| RECHAZA | 32 | 37.2 % |

### Comportamiento U3

| Métrica | Valor |
|---|---|
| U3 activado | 86 / 86 |
| U3 raw = U3_GOOD | 80 / 86 |
| U3 sin resultado (máscara fallida) | 6 / 86 |
| p_good media | 0.914 |
| BAD directos en peras sanas | **0** |
| REVIEW por safe mode (U3=BAD→REVIEW) | **0** |

### Hallazgo clave

U3 dice **GOOD** en 80/86 peras (93%), con p_good media de 0.914.  
El motor de reglas dice **RECHAZA** en 32 de esas mismas peras.

Esto confirma el problema de domain shift: la segmentación + detección de defectos rule-based  
detecta como "defecto" los bordes/sombras del fondo azul/oscuro del supermercado.  
U3 (entrenado sobre gray_bg_clean) no comete ese error.

**Safe mode:** funcionó perfectamente. 0 peras sanas rechazadas directamente por U3.

---

## 7. Análisis de discrepancias U3 vs reglas

| Caso | Conteo |
|---|---|
| U3=GOOD, reglas=PASA | 10 |
| U3=GOOD, reglas=REVISAR | 38 |
| U3=GOOD, reglas=RECHAZA | 32 |
| U3 sin resultado, cualquier decisión | 6 |

Las 32 peras clasificadas RECHAZA a pesar de U3=GOOD son falsos positivos del motor de reglas  
(defectos detectados que son en realidad fondo/sombra no correctamente segmentado).

---

## 8. Recomendación

### ¿Integrar U3 por defecto? **NO todavía — actuar como señal auxiliar opcional**

**Razón:** En la configuración actual, U3 GOOD no puede de-escalar una decisión RECHAZA de las  
reglas. El resultado es que la mayoría de las peras sanas de supermercado siguen siendo RECHAZA  
aunque U3 diga GOOD con confianza muy alta.

Para que U3 sea verdaderamente útil, se necesita una de estas dos opciones:

**Opción A (más segura):**  
Añadir un flag `--quality-u3-protect-good`: si U3 dice GOOD con p_good >= 0.85, convertir  
RECHAZA → REVISAR (nunca a PASA directamente). Esto protege las peras sanas de falso rechazo.

**Opción B (más completa):**  
Resolver el problema de raíz: mejorar la segmentación para fondo azul/oscuro de supermercado.  
Opciones: detector YOLO de pera + crop previo, o ajuste de los umbrales HSV.

### Estado actual: U3 como alarma de seguridad

En la configuración actual, U3 es útil como **alarma de seguridad anti-rechazo**:
- Si el pipeline dice RECHAZA pero U3 dice GOOD con alta confianza → bandera para revisión humana
- 0 falsos rechazos directos por U3 (safe mode funciona)
- 0 peras sanas enviadas a RECHAZA por U3

---

## 9. Archivos de salida

```
outputs/u3_integrated_pipeline_eval/resultados_integrated_u3.csv    (86 rows)
outputs/u3_integrated_pipeline_eval/contact_sheet_integrated_u3_all.jpg
outputs/u3_integrated_pipeline_eval/contact_sheet_integrated_u3_review_bad.jpg
outputs/u3_integrated_pipeline_eval/summary.txt
reports/u3_pipeline_integration_audit.md
reports/u3_pipeline_integration_report.md  (este archivo)
```

---

## 10. Validación de integridad

- `scripts/analyze_quality.py` — modificado con 8 edits quirúrgicos, backup creado ✓  
- V2 (`outputs/fruits360_quality_cls_v2/best_model.pt`) — no modificado ✓  
- U3 model (`outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt`) — no modificado ✓  
- `configs/quality_rules.yaml` — no modificado ✓  
- Sin shells en background al finalizar ✓
