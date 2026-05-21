# Inspección Dataset Mendeley Good/Bad Pear

**Fecha:** 2026-05-17

## Conteos

| Split | Total | Válidas | Corruptas |
|-------|-------|---------|-----------|
| good  | 532 | 532 | 0 |
| bad   | 500  | 500  | 0  |
| **Total** | **1032** | **1032** | **0** |

## Resoluciones

| Split | Min (WxH) | Max (WxH) | Media (WxH) |
|-------|-----------|-----------|-------------|
| good | 523x334 | 720x1600 | 574x386 |
| bad  | 720x691  | 720x975  | 720x794  |

## Orientación

| Split | Horizontal | Vertical | Cuadrada |
|-------|-----------|---------|---------|
| good | 529 | 3 | 0 |
| bad  | 0  | 300  | 200  |

## Previews generados

- `outputs/mendeley_good_bad_preview/good_grid.jpg`
- `outputs/mendeley_good_bad_preview/bad_grid.jpg`
- `outputs/mendeley_good_bad_preview/mixed_grid.jpg`

## Utilidad estimada del dataset

| Uso | Valoración |
|-----|-----------|
| Clasificación good/bad | **Alta** — etiquetas binarias directas |
| Detección de defectos  | **Baja** — sin bounding boxes |
| Segmentación           | **Baja** — sin máscaras de instancia |
| Apoyo al sistema actual (pre-filtro / clasificador) | **Media-Alta** — útil como clasificador binario auxiliar |

## Siguiente paso recomendado

Entrenar un clasificador binario ligero (YOLOv8n-cls o MobileNet)  
sobre este dataset para usarlo como pre-filtro que descarte peras  
claramente dañadas antes del pipeline principal de detección de defectos.