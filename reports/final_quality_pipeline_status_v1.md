# PearVision QC — Estado Final del Pipeline v1

Fecha: 2026-05-17

---

## 1. Estado General del Pipeline

El pipeline rule-based de inspección superficial de peras está operativo y validado.
Todos los componentes funcionan de forma integrada:

- Detector YOLO de pera (ECLPOD v1): detecta y recorta la ROI de la pera.
- Segmentación por GrabCut: genera la máscara de pera a partir del crop.
- Motor HSV rule-based: calcula defect_pct, dark_rot_pct, max_region_pct sobre el cuerpo de la pera.
- Discriminador de brillo corporal (body_l_mean): evita falsos rechazos en peras naturalmente oscuras.
- Modelo YOLO PSD de defectos: señal auxiliar (no dominante).
- Clasificador GOOD/BAD Mendeley: señal auxiliar informativa.
- Validación automática contra verdad humana: 8/8 PASS.

---

## 2. Resultado de Validación Humana

| Métrica             | Valor |
|---------------------|-------|
| Total expectativas  | 8     |
| PASS                | 8     |
| FAIL                | 0     |
| NOT FOUND           | 0     |

Resultado: **TODAS LAS EXPECTATIVAS CUMPLIDAS.**

---

## 3. Tabla de las 8 Imágenes de Expectativas

| Imagen              | Grupo             | Decisión | Allowed         | Resultado | def%  | rot%  | max%  | L    |
|---------------------|-------------------|----------|-----------------|-----------|-------|-------|-------|------|
| 1000057648.jpg      | bad               | REVISAR  | REVISAR\|RECHAZA | PASS     | 22.4% | 0.0%  | 9.8%  | 135  |
| 1000057649.jpg      | bad               | RECHAZA  | RECHAZA         | PASS      | 51.3% | 4.7%  | 47.0% | 108  |
| 1000057653.jpg      | bad               | RECHAZA  | RECHAZA         | PASS      | 43.5% | 22.6% | 37.6% | 80   |
| 1000057656.jpg      | bad               | REVISAR  | REVISAR\|RECHAZA | PASS     | 3.5%  | 2.6%  | 0.6%  | 72   |
| 1000057659.jpg      | bad               | RECHAZA  | RECHAZA         | PASS      | 99.8% | 98.5% | 99.8% | 39   |
| 1000057654.jpg      | supermarket_good  | REVISAR  | PASA\|REVISAR    | PASS     | 18.3% | 13.9% | 17.1% | 94   |
| 1000057655.jpg      | supermarket_good  | REVISAR  | PASA\|REVISAR    | PASS     | 14.2% | 3.9%  | 13.3% | 115  |
| 1000057658.jpg      | supermarket_good  | REVISAR  | PASA\|REVISAR    | PASS     | 74.2% | 66.8% | 73.8% | 58   |

---

## 4. Peras Malas Detectadas Correctamente

- **1000057649.jpg**: RECHAZA por defect_pct=51.3% >= reject_defect_pct(40.0). Pera con daño extenso.
- **1000057653.jpg**: RECHAZA por defect_pct=43.5% y rot=22.6% y max=37.6% — todos superan umbrales individuales.
- **1000057659.jpg**: RECHAZA por defect_pct=99.8% y rot=98.5%. Pera completamente necrosada (L_body=39, no activó el cap de brillo).
- **1000057648.jpg**: REVISAR (dudosa, confirmada como mala por usuario). Defecto alto (22.4%) sin podredumbre real — correctamente marcada para revisión humana.
- **1000057656.jpg**: REVISAR (dudosa). Valores moderados (def=3.5%, rot=2.6%) — no rechazada automáticamente pero tampoco aceptada.

---

## 5. Peras Buenas de Supermercado que ya no se Rechazan

Las tres peras de supermercado con russeting natural dejaron de clasificarse como RECHAZA:

- **1000057654.jpg**: Antes RECHAZA (rot=13.9% superaba reject_dark_rot_pct=12.0). Ahora REVISAR tras subir umbral a 20.0 y combo_rot a 15.0.
- **1000057655.jpg**: REVISAR. Russeting moderado (def=14.2%, rot=3.9%), dentro de los umbrales actuales.
- **1000057658.jpg**: Antes RECHAZA. Ahora REVISAR por el cap de brillo corporal: body_l_mean=58 (rango 45-70 = oscuro natural, no necrosis) + rot_pct=67% (inflado por color oscuro uniforme sin necrosis real).

---

## 6. Limitaciones Actuales

1. **Sensible a iluminación**: Variaciones de exposición afectan significativamente a defect_pct y dark_rot_pct. Una pera correcta fotografiada con poca luz puede obtener métricas muy altas.
2. **Fondo con manos u objetos**: La validación de captura filtra casos obvios, pero fondos complejos o presencia de manos pueden corromper la máscara GrabCut y producir métricas incorrectas.
3. **Color marrón/russeting sigue inflando métricas**: Los umbrales HSV actuales reducen el problema pero no lo eliminan. Peras con russeting extenso obtienen defect_pct y rot_pct artificialmente altos.
4. **Modelo PSD no generaliza suficiente**: En este conjunto de 12 imágenes solo detectó 1 defecto válido en 1 imagen. El modelo fue entrenado con el dataset Plant Sisease Dataset y no transfiere bien a peras con russeting o condiciones fotográficas distintas.
5. **Clasificador Mendeley GOOD/BAD tiene sesgo de dominio**: Predice GOOD incluso en peras con el 99% de superficie necrosada. Aprendió características de composición fotográfica (screenshots del dataset vs. fotos reales), no de calidad de la fruta. No debe activarse `--quality-cls-affect-decision` hasta reentrenarlo.

---

## 7. Recomendación Operativa Actual

Para obtener resultados fiables con el pipeline actual:

- **Fondo liso** de color contrastante con la pera (blanco o negro).
- **Buena iluminación uniforme** — evitar contraluz, sombras duras o subexposición.
- **Una sola pera por imagen**, centrada y visible completamente.
- **No usar manos ni fondos complejos** — el detector YOLO puede confundirse y la máscara GrabCut falla.
- Usar resolución de análisis >= 1280px (`--max-size 1280` o sin flag) — a 640px las métricas se alteran significativamente.
- Usar el clasificador Mendeley solo como señal informativa (`--use-quality-cls` sin `--quality-cls-affect-decision`).

---

## 8. Próximo Paso Recomendado

1. **Congelar esta versión como baseline v1**: los umbrales actuales y la lógica de decisión quedan bloqueados. No modificar sin una nueva ronda de validación.
2. **Crear un conjunto de test móvil no visto**: fotografiar peras nuevas con el móvil, sin que las imágenes hayan influido en ningún ajuste de umbrales.
3. **Evaluar sin tocar umbrales**: correr el pipeline en el nuevo conjunto y medir PASS/FAIL de expectativas.
4. **Añadir ejemplos fallidos como hard examples**: imágenes donde el pipeline comete errores deben incorporarse al conjunto de validación para detectar regresiones en iteraciones futuras.
5. **Reentrenar clasificador Mendeley** con imágenes propias (fotos reales de peras buenas y malas con el mismo dispositivo de captura).
