# Diagnóstico: Casos U3_NOT_USED_KEEP_ORIGINAL

**Fecha:** 2026-05-21  
**Imágenes afectadas:** 1000060736, 1000060738, 1000060739, 1000060740, 1000060741, 1000060745

---

## 1. Por qué U3 muestra g=0.00 b=0.00

El bloque U3 en `analyze_quality.py` está protegido por la condición:

```python
if u3_model is not None and capture_info["capture_valid"] and u3_thresholds is not None:
```

Las 6 imágenes tienen `capture_valid=False`, por lo tanto **U3 nunca se ejecutó**. Los campos `quality_u3_p_good` y `quality_u3_p_bad` quedan en 0.0 y `quality_u3_status = "not_used"`.

La función `apply_new_fusion()` en `reevaluate_u3_fusion_fixed.py` detecta `u3_status = "not_used"` y ejecuta `return original_decision, "U3_NOT_USED_KEEP_ORIGINAL"`, manteniendo la decisión REVISAR original sin modificarla.

---

## 2. ¿Faltaba la imagen original?

No. Las 6 imágenes existen en `data/supermarket_good_hard_examples_v1/images/`. No es un problema de imagen faltante.

---

## 3. ¿Faltaba ROI/masked clean?

No. `_make_u3_gray_input()` genera la máscara interna mediante distancia LAB desde los píxeles de las esquinas. No depende del pipeline de segmentación ni del ROI externo. Puede ejecutarse sobre cualquier imagen BGR directamente.

---

## 4. ¿Falló la máscara?

Para 5 imágenes la razón fue `capture_reason = "Pera muy pequeña (X% < 12%)"`. El pipeline considera capturas inválidas cuando `pear_visible_pct < 12%`. La pera está presente pero ocupa menos del 12% del frame.

Para 1000060745.jpg la razón fue `capture_reason = "Máscara rectangular: posible fondo (fill=0.84)"`. El detector de calidad de máscara detectó que la máscara tenía forma demasiado rectangular (bbox_fill_ratio > 0.84), indicando posible fondo incluido erróneamente.

En ambos casos, el fallo es del **pipeline rule-based**, no del modelo U3.

---

## 5. ¿Se reutilizó un CSV antiguo sin predicción U3?

No. El CSV fue generado por `test_integrated_u3_pipeline.py` que sí cargó U3. El problema es que la condición `capture_valid=False` impidió que U3 se ejecutara durante la evaluación.

---

## 6. Condición que causó U3_NOT_USED_KEEP_ORIGINAL

Cadena de decisión:

```
imagen → validate_capture() → capture_valid=False
                                    ↓
                         U3 bloqueado por condición
                                    ↓
                         quality_u3_status = "not_used"
                                    ↓
                  reevaluate_u3_fusion_fixed.py detecta "not_used"
                                    ↓
                  return original_decision = REVISAR  (U3_NOT_USED_KEEP_ORIGINAL)
```

**Las razones de `capture_valid=False`:**

| Imagen | Razón | pear_visible_pct |
|---|---|---|
| 1000060736.jpg | Pera muy pequeña | 10.9% |
| 1000060738.jpg | Pera muy pequeña | 7.0% |
| 1000060739.jpg | Pera muy pequeña | 11.7% |
| 1000060740.jpg | Pera muy pequeña | 10.3% |
| 1000060741.jpg | Pera muy pequeña | 11.5% |
| 1000060745.jpg | Máscara rectangular (fill=0.84) | 38.8% |

---

## Solución

Ejecutar U3 directamente sobre las imágenes originales sin pasar por la condición `capture_valid`. La función `_make_u3_gray_input()` solo necesita la imagen BGR — genera su propia máscara interna independiente del pipeline. Si U3 dice GOOD con p_good >= 0.85, la decisión puede cambiar de REVISAR a PASA.
