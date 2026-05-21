# Quality V2 — Input Pipeline Audit Report

**Fecha:** 2026-05-20
**Auditor:** Claude Code (automático)
**Objetivo:** determinar qué imagen ve realmente el clasificador V2 y por qué eso causa falsos BAD.

---

## 1. ¿Qué imagen entra al clasificador V2?

**Respuesta: imagen completa original, redimensionada a 224×224.**

Pipeline exacto (extraído de `scripts/train_fruits360_quality_v2.py` y todos los scripts `predict_*`):

```
imagen original  →  Resize(224, 224)  →  ToTensor()  →  Normalize(ImageNet)  →  MobileNetV3-small
```

No hay ningún paso de detección, segmentación ni máscara antes del clasificador.

### Transforms de evaluación (eval_tf)

```python
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),   # escala directamente sin preservar ratio
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
```

### Transforms de entrenamiento (train_tf)

```python
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
```

---

## 2. ¿Se usa detector de pera antes del clasificador?

**No.** El clasificador V2 recibe la imagen directamente sin ningún detector previo.

Existe un detector YOLO en:
`runs/detect/runs/pear_detector/eclpod_v1/weights/best.pt`
(task: detect, clase única: `pear`)

Pero este detector **NO está conectado al pipeline de clasificación de calidad V2**.
Solo existe como componente independiente.

---

## 3. ¿Se usa máscara antes del clasificador?

**No.** No existe ningún paso de segmentación, GrabCut, SAM ni máscara de ningún tipo
antes de que la imagen llegue al clasificador V2.

El fondo, sombras e iluminación son completamente visibles para el modelo.

---

## 4. ¿El fondo original sigue visible?

**Sí, completamente.** El clasificador V2 ve:

- La pera completa
- El fondo (blanco, azul, negro, texturizado, etc.)
- Las sombras proyectadas
- Los reflejos de iluminación
- El encuadre y orientación de la foto

---

## 5. ¿Por qué eso explica los falsos BAD por fondo/luz?

### Dataset de entrenamiento V2

V2 fue entrenado con `data/quality_fruits360_human_v2/`, que proviene del dataset
**Fruits-360**. Este dataset tiene imágenes de frutas sobre **fondo blanco uniforme**,
capturadas en condiciones de laboratorio controladas.

El modelo aprendió asociaciones que incluyen:
- Pera + fondo blanco → GOOD
- Pera + algún color/textura de fondo ≈ señal de anomalía

Cuando ve una pera sana sobre fondo azul o negro, hay píxeles no vistos durante
el entrenamiento que activan neuronas que el modelo asoció con "anomalía".

### Evidencia numérica

| Lote | Fondo dominante | Tasa de falso BAD |
|---|---|---|
| batch_v1 | variado, mayormente neutro | 5% (1/20) |
| batch_v2 — fondo blanco/claro | blanco | 0% (0/8) |
| batch_v2 — fondo azul | azul | 83% (5/6) |
| batch_v2 — fondo negro/texturizado | negro | 25% (2/8) |
| batch_v3 — condiciones válidas | blanco | 14% (3/22) |

Incluso con fondo blanco siguen apareciendo 3 falsos BAD en batch_v3 con confianza muy
baja (0.50–0.57). Esto sugiere que también influyen **sombras locales**, **orientación**
y **zonas de la pera** que el modelo interpreta como defecto por cambio de textura o color.

### Diagnóstico técnico

El clasificador V2 **no tiene anclaje espacial a la pera**. Aprende features globales
sobre toda la imagen 224×224. Si el fondo cambia, los features cambian y la predicción
puede verse arrastrada hacia BAD aunque la pera sea sana.

---

## 6. Comparación con la arquitectura óptima

| Componente | V2 actual | U3 propuesto |
|---|---|---|
| Detector previo | ❌ No | ✅ YOLO → bbox pera |
| Segmentación/máscara | ❌ No | ✅ GrabCut / SAM en bbox |
| Fondo neutralizado | ❌ No | ✅ Fondo gris/blanco neutro |
| Input al clasificador | imagen completa | solo región de la pera |
| Augmentación fondo | N/A | Sí (fondos sintéticos variados) |

---

## 7. Conclusión

El clasificador V2 es **sensible al dominio de captura** porque ve la imagen completa.
Para conseguir un clasificador robusto a fondos variados, el siguiente paso es:

1. Detectar la pera con YOLO.
2. Segmentar su contorno con GrabCut (o SAM si disponible).
3. Reemplazar el fondo por un color neutro.
4. Clasificar solo la pera normalizada.
5. Entrenar U3 con este pipeline y hard examples de supermercado.

Ver `reports/quality_roi_masked_u3_plan.md` para el plan completo.
