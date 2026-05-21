# Quality ROI/Masked Pipeline — U3 Training Plan

**Fecha:** 2026-05-20
**Estado:** Preparación pre-entrenamiento (no se ha entrenado aún)

---

## 1. Por qué falla V2

### Causa raíz
V2 recibe la imagen completa (224×224) sin ningún preprocesado previo de localización.
El modelo aprende features globales que incluyen el fondo, sombras e iluminación.

Como el dataset de entrenamiento (Fruits-360) usa fondo blanco uniforme, el modelo
aprendió implícitamente que "pera sana = pera sobre fondo blanco". Cuando encuentra
un fondo azul, negro o texturizado, lo interpreta como señal de anomalía.

### Consecuencias observadas
- 83% de falsos BAD en peras sanas sobre fondo azul.
- 25% de falsos BAD en peras sanas sobre fondo negro/texturizado.
- 14% de falsos BAD con fondo blanco válido (confianza muy baja 0.50–0.57),
  probablemente por sombras locales u orientación.

### Lo que V2 no puede hacer
- Ignorar el fondo por diseño.
- Generalizar a fondos no vistos durante entrenamiento.
- Distinguir entre "textura de fondo" y "defecto en la pera".

---

## 2. Qué debe hacer U3

### Pipeline de entrada propuesto

```
imagen original
  → YOLO detector (best.pt eclpod_v1)  →  bbox de la pera
  → GrabCut inicializado con bbox        →  máscara binaria de la pera
  → morfología (closing + opening)       →  máscara limpia
  → reemplazar fondo por gris neutro     →  imagen pera-only
  → resize 224×224 con padding           →  input al clasificador U3
```

### Requisitos del entrenamiento U3

1. **Imágenes ROI/masked:** todas las imágenes de entrenamiento deben pasar por el
   mismo pipeline YOLO+GrabCut antes de llegar al clasificador.
2. **Fondo neutralizado:** gris neutro (128, 128, 128) o blanco (255, 255, 255).
   Usar gris neutro es preferible para no sesgar el modelo hacia blanco.
3. **Augmentación fuerte de luz/sombra:**
   - `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)`
   - `RandomAffine` ligero para variaciones de orientación
   - Augmentación de fondo sintético: insertar la pera masked sobre fondos de colores
     variados para forzar invariancia al fondo
4. **Hard examples GOOD de supermercado:** incluir obligatoriamente en entrenamiento:
   - `data/supermarket_good_hard_examples_v1/` (20 imágenes)
   - `data/supermarket_good_hard_examples_v2/` (22 imágenes, 7 falsos BAD clave)
   - batch V3 si se registra (22 imágenes)

---

## 3. Dataset para U3

| Dataset | Ruta | Imágenes | Uso en U3 |
|---|---|---|---|
| Fruits-360 V2 (good) | `data/quality_fruits360_human_v2/train/good/` | ~N | Entrenamiento GOOD base |
| Fruits-360 V2 (bad) | `data/quality_fruits360_human_v2/train/bad/` | ~N | Entrenamiento BAD base |
| Hard examples V1 | `data/supermarket_good_hard_examples_v1/` | 20 | GOOD supermercado |
| Hard examples V2 | `data/supermarket_good_hard_examples_v2/` | 22 | GOOD supermercado (fondos variados) |
| Batch V3 (pendiente registro) | a registrar | 22 | GOOD condiciones válidas |

**Nota:** Los hard examples de supermercado deben tener un peso mayor en el sampler
(WeightedRandomSampler) porque son los casos reales más difíciles.

---

## 4. Qué NO hacer

- **No seguir ajustando solo thresholds** operativos. El umbral `BAD < 0.70 → REVIEW`
  es un parche temporal útil, pero no soluciona el problema de raíz.
- **No entrenar U3 con imágenes completas** (sin ROI/mask). Si se hace así, U3
  repetirá los mismos errores de V2 con fondos no vistos.
- **No entrenar todavía** sin comprobar primero visualmente las máscaras GrabCut.
  Si las máscaras recortan mal la pera (incluyen demasiado fondo o pierden partes),
  el entrenamiento aprenderá sobre datos ruidosos.
- **No confiar en fondos azul/negro** hasta que U3 esté validado explícitamente con
  imágenes de esos fondos en el set de test.
- **No eliminar V2** — mantenerlo como baseline para comparación cuantitativa de U3.

---

## 5. Recomendación operativa temporal (hasta U3)

Mientras no esté U3 entrenado y validado:

| Predicción V2 | Confianza | Decisión operativa |
|---|---|---|
| GOOD | cualquiera | Aceptar |
| BAD | < 0.70 | REVIEW (no rechazar automáticamente) |
| BAD | ≥ 0.70 | Rechazar (revisar si hay duda) |

- Usar **fondo blanco mate** o gris claro para las capturas.
- Usar **iluminación difusa** (sin flash directo, sin luz lateral fuerte).
- **Evitar fondos azules, negros, brillantes o con textura fuerte.**

---

## 6. Checklist antes de entrenar U3

- [ ] Revisar `contact_sheet_original_vs_masked.jpg` — ¿las máscaras son correctas?
- [ ] Revisar `problem_cases_grid.jpg` — ¿los casos falsos BAD quedan bien segmentados?
- [ ] Si máscaras OK → lanzar pipeline batch para preprocesar todo el dataset V2 + hard examples
- [ ] Contar imágenes por clase tras el preprocesado
- [ ] Definir arquitectura U3 (MobileNetV3-small como V2, o probar EfficientNet-B0)
- [ ] Definir augmentación fuerte de fondo
- [ ] Entrenar con WeightedRandomSampler (hard examples con peso ×2 o ×3)
- [ ] Evaluar U3 sobre batches V1/V2/V3 con imágenes preprocesadas
- [ ] Comparar U3 vs V2 en tasa de falso BAD

---

## 7. Archivos de referencia

| Archivo | Descripción |
|---|---|
| `reports/quality_v2_input_audit_report.md` | Auditoría del pipeline de entrada V2 |
| `outputs/quality_roi_masked_previews/contact_sheet_original_vs_masked.jpg` | Preview de máscaras |
| `outputs/quality_roi_masked_previews/problem_cases_grid.jpg` | Casos problemáticos con máscara |
| `outputs/quality_roi_masked_previews/roi_masked_diagnostics.csv` | Métricas de detección/máscara |
| `scripts/prepare_quality_roi_masked_previews.py` | Script de preprocesado |

---

## 8. Confirmaciones

- **NO** se ha entrenado ningún modelo en este paso.
- **NO** se ha modificado V2.
- **NO** se ha modificado `analyze_quality.py`.
- **NO** se ha modificado `quality_rules.yaml`.
