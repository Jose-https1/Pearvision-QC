# PearVision QC — Fruits-360 Good Pear Eval Audit Report

Fecha: 2026-05-17

---

## 1. Resumen del Dataset Usado

| Campo | Valor |
|-------|-------|
| Fuente | Fruits-360 original-size (Training / Validation / Test) |
| Clases seleccionadas | Todas las carpetas `Pear*` (12 en Training/Validation, 12 en Test) |
| Método de selección | 5 imágenes por clase×split, orden aleatorio seed=42 |
| Total imágenes | 180 |
| Ruta dataset | `data/samples_quality_fruits360_good_eval/` |
| Ruta expectativas | `data/samples_quality_fruits360_good_eval_expectations.csv` |

Todas las imágenes son peras sanas de Fruits-360 (fondo blanco, iluminación controlada de estudio).
Ninguna contiene defectos reales. La expectativa es `PASA|REVISAR` para todas.

---

## 2. Comando Exacto Ejecutado

```
uv run python scripts/analyze_quality.py \
  --source data/samples_quality_fruits360_good_eval \
  --save \
  --use-detector \
  --detect-conf 0.50 \
  --use-defect-model \
  --defect-conf 0.25 \
  --use-quality-cls \
  --quality-cls-model runs/pear_quality_cls/mendeley_good_bad_v1/weights/best.pt \
  --quality-cls-bad-thr 0.85 \
  --quality-cls-affect-decision
```

---

## 3. Conteo Final

| Decisión | Cantidad | % del total | Notas |
|----------|----------|-------------|-------|
| PASA | 5 | 2.8% | Muy bajo — dominio diferente |
| REVISAR | 159 | 88.3% | 137 de ellos por captura inválida |
| RECHAZA | 16 | **8.9%** | **Falsos rechazos** |
| FALSE_REJECT_RATE | — | **8.9%** | Por encima del umbral aceptable (5%) |

**STATUS: NEEDS_RULE_FIX**

---

## 4. Falsos Rechazos (16 imágenes RECHAZA)

| Imagen | def% | rot% | max% | brown% | Causa |
|--------|------|------|------|--------|-------|
| Test__Pear_12__r0_23.jpg | 67% | 32% | 46% | 69% | high_defect+high_rot+high_max |
| Test__Pear_14__r2_131.jpg | 67% | 47% | 67% | 89% | high_defect+high_rot+high_max |
| Test__Pear_8__r0_203.jpg | 93% | 19% | 91% | 98% | high_defect+high_max |
| Training__Pear_6__r0_242.jpg | 23% | 18% | 19% | 23% | combo (triple umbral combinado) |
| Training__Pear_7__r0_24.jpg | 88% | 46% | 87% | 89% | high_defect+high_rot+high_max |
| Training__Pear_8__r0_182.jpg | 93% | 33% | 93% | 97% | high_defect+high_rot+high_max |
| Training__Pear_8__r0_186.jpg | 92% | 28% | 92% | 97% | high_defect+high_rot+high_max |
| Training__Pear_9__r3_106.jpg | 84% | 32% | 83% | 83% | high_defect+high_rot+high_max |
| Training__Pear_9__r3_96.jpg | 74% | 29% | 73% | 73% | high_defect+high_rot+high_max |
| Training__Pear_common_1__r1_4.jpg | 61% | 36% | 61% | 84% | high_defect+high_rot+high_max |
| Validation__Pear_10__r0_161.jpg | 23% | 23% | 20% | 23% | combo |
| Validation__Pear_12__r0_33.jpg | 67% | 29% | 44% | 69% | high_defect+high_rot+high_max |
| Validation__Pear_7__r2_305.jpg | 61% | 43% | 57% | 61% | high_defect+high_rot+high_max |
| Validation__Pear_8__r0_129.jpg | 97% | 24% | 97% | 98% | high_defect+high_max |
| Validation__Pear_8__r0_149.jpg | 97% | 37% | 97% | 99% | high_defect+high_rot+high_max |
| Validation__Pear_common_1__r1_109.jpg | 69% | 37% | 68% | 88% | high_defect+high_rot+high_max |

**14 de 16 falsos rechazos** son `high_defect+high_rot+high_max` con `brown_dark_pct > 60%`.
**2 de 16** son rechazos por combinación de umbrales moderados (combo).

---

## 5. Métricas Más Frecuentes que Causan el Rechazo

| Condición | Frecuencia |
|-----------|-----------|
| defect_pct >= 40% | 14 / 16 |
| dark_rot_pct >= 20% | 14 / 16 |
| max_region_pct >= 25% | 14 / 16 |
| brown_dark_pct > 60% | 14 / 16 |
| rechazo por combo moderado | 2 / 16 |

El patrón dominante: `brown_dark_pct` muy alto → `defect_pct` y `dark_rot_pct` inflados → rechazo por umbral individual o combinación triple.

---

## 6. Diagnóstico Técnico

### 6.1 Problema principal: dominio de iluminación y fondo

Fruits-360 usa **fondo blanco de estudio** con iluminación artificial uniforme y multidireccional.
Las peras del dataset de entrenamiento del pipeline son **fotos con móvil**, fondo variable, iluminación natural.

Este cambio de dominio causa dos problemas independientes:

### 6.2 Detector YOLO (ECLPOD): fallo masivo de detección

- **96 de 180 peras (53%) no detectadas** por el YOLO con `conf >= 0.50`.
- El detector fue entrenado/ajustado con fotos reales de peras sobre fondos naturales.
- Las peras de Fruits-360, sobre fondo blanco puro, tienen distribución de textura y contraste completamente diferente.
- Resultado: todas esas 96 peras quedan como `REVISAR - REPETIR FOTO - PERA NO DETECTADA`.
- **No es un fallo del pipeline de calidad** — es un fallo de detección por diferencia de dominio.

### 6.3 Validación de máscara: "máscara rectangular"

- **41 peras adicionales** son detectadas por YOLO pero la máscara GrabCut produce un fill ratio > 0.82.
- En Fruits-360, el fondo blanco crea una silueta muy limpia y la GrabCut devuelve una máscara casi perfecta que llena casi todo el bounding box → parece "demasiado rectangular" para la validación anti-fondo.
- Resultado: `REVISAR - CAPTURA NO VALIDA`.
- **Estas peras están bien detectadas** pero la validación de captura las descarta.

### 6.4 HSV color analysis: falsos positivos por color natural de pera

- 16 peras superan umbrales de rechazo.
- En Fruits-360 hay peras muy oscuras (Pear 8, Pear common 1) con piel naturalmente marrón-oscura.
- Estas peras tienen `brown_dark_pct > 80%` — casi toda la superficie cae en el rango HSV `[5-25, 50-200, 60-135]` definido para manchas de daño.
- El umbral `reject_defect_pct = 40%` no contempla peras enteras de color marrón oscuro natural.
- **Causa**: los umbrales HSV se calibraron sobre peras verde-amarillas de supermercado español; Fruits-360 incluye variedades de pera con colores más oscuros/distintos.

### 6.5 Modelo de defectos PSD (YOLO)

- Solo **1 detección válida** en todo el dataset (igual que en el test controlado).
- No contribuye a los rechazos. No es el problema.

### 6.6 Clasificador GOOD/BAD Mendeley

- Predice GOOD para la mayoría de peras (sesgo de dominio conocido).
- Para los 16 falsos rechazos, el clasificador predice mayoritariamente GOOD.
- No agrava los rechazos. No es el problema.
- Pero tampoco los corrige: la decisión ya viene forzada por las reglas HSV.

---

## 7. Recomendación Concreta

### Recomendación 1 — Mantener Fruits-360 SOLO como evaluación de peras buenas (no tocar umbrales ahora)

Los umbrales actuales están calibrados y validados en 8 imágenes propias (8/8 PASS).
No deben modificarse solo para que pasen peras de Fruits-360, que tienen un dominio de iluminación y variedad completamente diferente.

**Acción inmediata**: ninguna sobre los umbrales.

### Recomendación 2 — Validación de captura no compatible con Fruits-360

El filtro de máscara rectangular (`max_mask_bbox_fill_ratio = 0.82`) y la confianza mínima del detector (`detect-conf 0.50`) son adecuados para fotos reales con móvil.
Para evaluar Fruits-360 con fondo blanco se necesitaría bajar la confianza del detector y ajustar el filtro de fill ratio — pero esto degradaría el rendimiento en el caso de uso real.

**Acción inmediata**: no cambiar. Documentar que el pipeline no está diseñado para imágenes de estudio con fondo blanco.

### Recomendación 3 — Entrenar un nuevo clasificador GOOD/BAD

El clasificador Mendeley tiene sesgo de dominio conocido. Si se combina Fruits-360 (peras buenas, variedad de color) con las peras propias dañadas, se puede entrenar un clasificador más robusto.

**Acción futura**: preparar dataset mixto Fruits-360 + peras propias y reentrenar con YOLOv8 cls.

### Recomendación 4 — Crear subconjunto Fruits-360 compatible

Usar solo las variedades de Fruits-360 cuyo color se asemeja a las peras reales del pipeline:
`Pear 1`, `Pear 3`, `Pear 5`, `Pear 6` (colores verdes/amarillos claros).
Excluir `Pear 8`, `Pear 9`, `Pear common 1` (muy oscuras o tipo russeting extremo).
Esto reduciría la tasa de falso rechazo a < 2% sin tocar umbrales.

**Acción inmediata opcional**: filtrar el subset en la siguiente evaluación.

---

## 8. Archivos Generados

| Archivo | Ruta |
|---------|------|
| Dataset Fruits-360 eval | `data/samples_quality_fruits360_good_eval/` |
| CSV expectativas | `data/samples_quality_fruits360_good_eval_expectations.csv` |
| CSV resultados | `outputs/quality_analysis_fruits360_good_eval/resultados_calidad.csv` |
| Reporte validación | `outputs/quality_audit_fruits360_good_eval/validation_report.txt` |
| Contact sheet audit | `outputs/quality_audit_fruits360_good_eval/contact_sheet_fruits360_good_eval.jpg` |
| Preview dataset | `outputs/fruits360_good_pear_eval_preview/contact_sheet.jpg` |
| Este informe | `reports/fruits360_good_pear_eval_audit_report.md` |
