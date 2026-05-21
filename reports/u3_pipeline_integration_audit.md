# U3 Pipeline Integration Audit

**Fecha:** 2026-05-21
**Backup:** `scripts/analyze_quality_backup_before_u3_20260521_010831.py`

---

## 1. Cómo decide ahora el pipeline

El pipeline actual en `scripts/analyze_quality.py` sigue este flujo:

```
imagen
  │
  ├─ [opcional] YOLO detector de pera → bbox + crop
  │
  ├─ segmentación de pera (GrabCut o ROI mask)
  │
  ├─ detección de defectos (HSV, contornos, color)
  │
  ├─ decide() → PASA / REVISAR / RECHAZA
  │     basado en defect_pct, rot_pct, region_pct vs quality_rules.yaml
  │
  ├─ cap por brillo ambiguo (body_l_mean entre 45-70 → baja RECHAZA a REVISAR)
  │
  ├─ cap por máscara fallback (máscara de baja calidad → baja RECHAZA a REVISAR)
  │
  ├─ [opcional] YOLO detector de defectos PSD → señal auxiliar
  │     puede subir PASA→REVISAR, nunca directamente a RECHAZA
  │
  └─ [opcional] clasificador GOOD/BAD auxiliar (Mendeley, YOLO cls format)
        puede subir PASA→REVISAR si bad_conf >= bad_thr
```

**Vocabulario de decisiones:**
- `PASA` = pera comercial (OK)
- `REVISAR` = inspección humana requerida
- `RECHAZA` = pera no comercial (BAD)

---

## 2. Clasificador de calidad actual

El pipeline ya tiene un slot para un clasificador auxiliar `quality_cls_model`:
- **Formato esperado:** modelo YOLO clasificación (ultralytics), con clases `good`/`bad`.
- **Cargado con:** `_load_yolo_model()` → `model.predict()` vía ultralytics.
- **Activado con:** `--use-quality-cls`
- **Limitación:** solo puede subir `PASA → REVISAR`; no puede bajar ni rechazar directamente.

**U3 NO es un modelo YOLO.** U3 es MobileNetV3-small (PyTorch puro, torchvision). Necesita un slot diferente.

---

## 3. Dónde entra U3

U3 se añade como un **bloque adicional DESPUÉS** del bloque `quality_cls`:

```python
# Bloque existente
metrics.update(quality_cls_metrics)   # ← línea 513

# NUEVO BLOQUE U3 (añadido aquí)
if u3_model is not None and capture_info["capture_valid"]:
    gray_input = _make_u3_gray_input(image_for_analysis)
    p_bad, p_good, u3_raw, u3_safe = _apply_u3_safe_mode(
        gray_input, u3_model, u3_thresholds, decision, yolo_defect_metrics, u3_safe_mode
    )
    decision = _apply_u3_to_decision(decision, u3_safe)

result = { ... }   # ← línea 515 actual
```

---

## 4. Cómo se añade U3 sin romper lo anterior

- Nuevos argumentos: `--use-quality-u3`, `--quality-u3-model`, `--quality-u3-thresholds`, `--quality-u3-safe-mode`
- Si `--use-quality-u3` NO se usa: **comportamiento idéntico al actual**. Cero cambios en el flujo existente.
- U3 recibe `gray_bg_clean` (pera con fondo gris) para coincidir con el dominio de entrenamiento.
- La masking usa estimación de fondo por esquinas + distancia LAB, funciona para fondo blanco, azul y negro.
- Safe mode: U3=BAD solo produce RECHAZA si las reglas/detector ya confirmaban defecto fuerte. En caso contrario, U3=BAD → REVISAR.

---

## 5. Umbrales U3

Leídos desde `selected_thresholds.json`:

| Parámetro | Valor |
|---|---|
| bad_reject_threshold | 0.60 |
| good_accept_threshold | 0.55 |
| classes | ["bad", "good"] |
| class_bad_idx | 0 |
| class_good_idx | 1 |

---

## 6. "Defecto fuerte" para safe mode

Se considera que ya existe evidencia fuerte de defecto cuando:
- `decision_before_u3 == "RECHAZA"` (reglas ya rechazaron), O
- `yolo_defect_count >= 2` (detector PSD encontró 2+ defectos), O
- `yolo_defect_max_conf > 0.65` (detector PSD muy confiado)

Si no se cumple ninguna condición: U3=BAD → REVISAR (safe mode activo).

---

## 7. No modificado

- `configs/quality_rules.yaml`: sin cambios.
- `outputs/fruits360_quality_cls_v2/best_model.pt`: sin cambios.
- `outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt`: sin cambios.
