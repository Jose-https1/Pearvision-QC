# Reporte: Corrección de Fusión U3 — u3_fusion_fix_report

**Fecha:** 2026-05-21  
**Versión:** V1 (sin reentrenamiento de ningún modelo)  
**Modelo U3:** `outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt` (sin modificar)

---

## Resumen de resultados

### Antes (pipeline con bug de fusión)

| Decisión | N | % |
|---|---|---|
| PASA | 10 | 11.6% |
| REVISAR | 44 | 51.2% |
| RECHAZA | 32 | 37.2% |
| **Total** | **86** | |

### Después (fusión corregida)

| Decisión | N | % |
|---|---|---|
| PASA | 80 | 93.0% |
| REVISAR | 6 | 7.0% |
| RECHAZA | 0 | 0.0% |
| **Total** | **86** | |

---

## Correcciones aplicadas

| Transición | Cantidad |
|---|---|
| RECHAZA → PASA | 32 |
| RECHAZA → REVISAR | 0 |
| RECHAZA mantenidas | 0 |
| REVISAR → PASA | 38 |
| PASA → PASA (sin cambio) | 10 |

---

## RECHAZA mantenidas: 0 — análisis

Ninguna de las 32 RECHAZA se mantuvo porque en todas ellas:
- `yolo_defect_count = 0` (sin detector YOLO de defectos activo)
- `yolo_defect_max_conf = 0.0`
- No hay necrosis extrema real (`dark_rot_pct <= 50%` o `body_l_mean >= 45`)
- `strong_defect_evidence = False` para todas

Por tanto, las 32 aplicaron la regla `U3_GOOD_STRONG_NO_STRONG_DEFECT` → PASA.

---

## REVISAR restantes: 6 — análisis

Los 6 REVISAR restantes son capturas no válidas (`capture_valid = False`):

| Imagen | Razón |
|---|---|
| 1000060736.jpg | Pera muy pequeña (10.9% < 12%) |
| 1000060738.jpg | Pera muy pequeña (7.0% < 12%) |
| 1000060739.jpg | Pera muy pequeña (11.7% < 12%) |
| 1000060740.jpg | Pera muy pequeña (10.3% < 12%) |
| 1000060741.jpg | Pera muy pequeña (11.5% < 12%) |
| 1000060745.jpg | Máscara rectangular (fill=0.84) |

U3 no se ejecutó en estas imágenes. El REVISAR es correcto: la toma no es aprovechable.

---

## ¿U3 está protegiendo correctamente?

**Sí.** La protección funciona en dos niveles:

1. **Nivel fuerte** (`p_good >= 0.85`, sin defecto YOLO): → PASA  
   Activa para todas las peras sanas con p_good desde 0.85 hasta 0.9997.

2. **Nivel débil** (`0.55 <= p_good < 0.85`, RECHAZA previo, sin defecto): → REVISAR  
   No activado en este lote (todos tenían p_good >= 0.85).

La regla `U3_BAD` también está corregida: ahora solo produce RECHAZA si hay `strong_defect_evidence = True`. Sin detector de defectos, ninguna pera puede ser RECHAZA solo por U3.

---

## Cambios al código

### `scripts/analyze_quality.py` (con backup)

**Backup:** `scripts/analyze_quality_backup_before_fusion_fix_20260521_105824.py`

**Cambio 1:** Eliminado `decision_before_u3 == "RECHAZA"` de `strong_defect`:
```python
# ANTES (bug):
strong_defect = (
    decision_before_u3 == "RECHAZA"  # ← falso positivo
    or yolo_defect_metrics.get("yolo_defect_count", 0) >= 2
    ...
)

# DESPUÉS (correcto):
strong_defect = (
    yolo_defect_metrics.get("yolo_defect_count", 0) >= 2
    or yolo_defect_metrics.get("yolo_defect_max_conf", 0.0) > 0.65
    or (metrics.get("rot_pct", 0.0) > 50.0 and metrics.get("body_l_mean", 128.0) < 45.0)
)
```

**Cambio 2:** Reemplazado el comentario muerto `# U3_GOOD: no modifica la decision existente` por la lógica activa:
```python
elif u3_safe == "GOOD":
    if p_good >= 0.85 and not strong_defect:
        decision = "PASA"  # ← protección activa
        reason = "U3_GOOD_STRONG_NO_STRONG_DEFECT"
    elif p_good >= 0.85 and strong_defect:
        decision = "REVISAR"  # ← duda cuando hay conflicto
    elif p_good >= 0.55 and decision == "RECHAZA" and not strong_defect:
        decision = "REVISAR"  # ← protección débil
```

### `scripts/reevaluate_u3_fusion_fixed.py` (nuevo)

Script offline que aplica la fusión corregida al CSV existente sin reejecutar inferencia.

---

## Archivos generados

| Archivo | Estado |
|---|---|
| `outputs/u3_fusion_fixed_eval/resultados_u3_fusion_fixed.csv` | ✓ generado |
| `outputs/u3_fusion_fixed_eval/contact_sheet_u3_fusion_fixed_all.jpg` | ✓ generado |
| `outputs/u3_fusion_fixed_eval/contact_sheet_u3_fusion_fixed_review_bad.jpg` | ✓ generado (6 peras) |
| `outputs/u3_fusion_fixed_eval/summary.txt` | ✓ generado |
| `reports/u3_fusion_false_reject_diagnosis.md` | ✓ generado |
| `scripts/analyze_quality_backup_before_fusion_fix_20260521_105824.py` | ✓ backup creado |
| `scripts/analyze_quality.py` | ✓ modificado (fusión corregida) |
| `scripts/reevaluate_u3_fusion_fixed.py` | ✓ creado |

---

## Qué NO se modificó

- `outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt` — sin cambios
- `configs/quality_rules.yaml` — sin cambios
- Cualquier output anterior — sin cambios
- Modelos V2 — sin cambios

---

## Evaluación del objetivo

| Objetivo | Meta | Resultado |
|---|---|---|
| RECHAZA = 0 | 0 | **0 ✓** |
| PASA >= 60/86 | >=60 | **80 ✓** |
| REVISAR solo ambiguos | Solo capturas inválidas | **6 ✓** |

**El objetivo se cumple con holgura.** La corrección es efectiva.

---

## Recomendación

**Aceptar esta fusión.** Los resultados son correctos para el lote de evaluación de peras sanas:
- 0 falsos rechazos
- 80/86 correctamente clasificadas como PASA
- Los 6 REVISAR son capturas que legítimamente necesitan retomarse

**Siguiente paso:** José debe revisar visualmente `contact_sheet_u3_fusion_fixed_all.jpg` y `contact_sheet_u3_fusion_fixed_review_bad.jpg` para confirmar que las 80 PASA son visualmente correctas y que los 6 REVISAR son efectivamente capturas de mala calidad.

Si alguna pera sana sana aparece todavía con color marrón intenso y José considera que debe bajar el umbral `good_accept_threshold` de 0.85 a un valor menor, ese ajuste puede hacerse en `selected_thresholds.json` sin reentrenar.
