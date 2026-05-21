# Diagnóstico: Falsos Rechazos en Pipeline Integrado U3

**Fecha:** 2026-05-21  
**Archivo base:** `outputs/u3_integrated_pipeline_eval/resultados_integrated_u3.csv`  
**Resultado previo:** PASA=10, REVISAR=44, RECHAZA=32

---

## 1. Por qué hay 32 RECHAZA

Las 32 imágenes RECHAZA son peras sanas de supermercado (lote de evaluación "good pears").  
La causa raíz es la **métrica rule-based de defectos por color/textura** en `detect_defects()`:

- El sistema detecta como "defecto" la coloración marrón uniforme del **russeting natural** de la pera conferencia.
- Detecta como "rot" las zonas oscuras que son simplemente **sombras, pedúnculo o cáliz**.
- Detecta como "defecto localizado" manchas que son **variación natural de piel rugosa**.

Cuando `defect_pct` supera el umbral de "NO COMERCIAL" en `quality_rules.yaml`, la regla devuelve RECHAZA.  
No hay detector YOLO de defectos activo en esta evaluación (`yolo_defect_count=0` en todos los casos).

**Valores típicos en los 32 RECHAZA:**
- `defect_pct`: 18–65% (causado por russeting/textura, no por daño real)
- `dark_rot_pct`: 0–34% (sombras y russeting oscuro, no podredumbre)
- `body_l_mean`: 81–147 (rango normal; no indica necrosis extrema)
- `yolo_defect_count`: 0 en todos los casos
- `yolo_defect_max_conf`: 0.0 en todos los casos

---

## 2. Qué columnas/reglas causan los 32 RECHAZA

La columna `estimated_category = "NO COMERCIAL"` y `display_label = "RECHAZA - NO COMERCIAL"` indica que la función `decide()` en `src/quality_analysis.py` disparó la regla:

```
si defect_pct > umbral_no_comercial → RECHAZA / NO COMERCIAL
```

Esta regla usa exclusivamente métricas HSV (hue, saturación, valor) y no distingue entre:
- Russeting marrón natural → detectado erróneamente como defecto
- Daño mecánico real con cambio de color

---

## 3. ¿U3 estaba diciendo GOOD en esas imágenes?

**Sí. En los 32 RECHAZA, U3 decía GOOD con confianza muy alta:**

| Rango p_good | Número de imágenes RECHAZA |
|---|---|
| p_good >= 0.99 | ~20 |
| 0.95 <= p_good < 0.99 | ~8 |
| 0.90 <= p_good < 0.95 | ~3 |
| 0.85 <= p_good < 0.90 | ~1 |

Todas con `quality_u3_decision_raw = U3_GOOD` y `quality_u3_decision_safe = GOOD`.

El campo `final_decision_reason` = `u3_good (p_good=X.XX)` para todas, confirmando que U3 procesó correctamente la imagen y emitió veredicto GOOD.

---

## 4. ¿Las reglas antiguas están dominando sobre U3?

**Sí. Este es el bug central.**

Código original en `analyze_quality.py` (bloque U3, ~línea 651):

```python
# U3_GOOD: no modifica la decision existente
```

Este comentario describe exactamente el problema: cuando `u3_safe == "GOOD"`, el código **no hace nada**. La decisión `RECHAZA` de la regla rule-based permanece inalterada.

Además, había un error secundario en la definición de `strong_defect`:

```python
strong_defect = (
    decision_before_u3 == "RECHAZA"   # ← BUG: esto siempre es True para las 32 RECHAZA
    or yolo_defect_metrics.get("yolo_defect_count", 0) >= 2
    or yolo_defect_metrics.get("yolo_defect_max_conf", 0.0) > 0.65
)
```

Aunque `strong_defect` solo se usaba en el bloque `U3_BAD`, conceptualmente incluir `decision_before_u3 == "RECHAZA"` como evidencia fuerte era incorrecto: eso significaba que cualquier RECHAZA de las reglas se trataba como evidencia de defecto real, perpetuando el ciclo de falsos rechazos.

---

## 5. Por qué el summary decía "BAD directos en peras sanas: 0"

El campo `quality_u3_decision_safe` para las 32 RECHAZA era `GOOD`, no `BAD`.  
La lógica del `test_integrated_u3_pipeline.py` contaba `direct_bad_on_good` solo cuando `safe == "U3_BAD"`:

```python
if decision == "RECHAZA" and safe == "U3_BAD":
    direct_bad_on_good += 1
```

Como U3 nunca dijo BAD en estas imágenes, el contador quedó en 0. Pero las peras llegaron a RECHAZA **por las reglas antiguas**, no por U3. El summary era técnicamente correcto pero engañoso: los RECHAZA no venían de U3 sino de la fusión rota que ignoraba el veredicto GOOD de U3.

---

## 6. Los 6 REVISAR restantes (legítimos)

Después de la corrección, quedan 6 REVISAR. Todos son **capturas no válidas**:

- `1000060736.jpg`: Pera muy pequeña (10.9% < 12%)
- `1000060738.jpg`: Pera muy pequeña (7.0% < 12%)
- `1000060739.jpg`: Pera muy pequeña (11.7% < 12%)
- `1000060740.jpg`: Pera muy pequeña (10.3% < 12%)
- `1000060741.jpg`: Pera muy pequeña (11.5% < 12%)
- `1000060745.jpg`: Máscara rectangular (fill=0.84, posible fondo)

Estas imágenes no pasaron por U3 (`quality_u3_status = not_used`) porque `capture_valid = False`.  
El REVISAR aquí es correcto: la captura necesita repetirse.

---

## Conclusión

El fallo de fusión era un bug de una línea: U3=GOOD no tenía efecto sobre la decisión final.  
La corrección no requiere reentrenamiento ni cambio de umbrales: solo activar la protección U3=GOOD.
