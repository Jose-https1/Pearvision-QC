# Fruits-360 Quality Curation v1 — Informe

Generado: 2026-05-17 20:34

## Origen
- Dataset evaluado: `data/samples_quality_fruits360_good_eval/`
- CSV de resultados: `outputs/quality_analysis_fruits360_good_eval/resultados_calidad.csv`
- Total imágenes procesadas: 180

## Distribución del curado

| Grupo | Cantidad | % |
|-------|----------|---|
| good | 5 | 2.8% |
| bad | 16 | 8.9% |
| review | 22 | 12.2% |
| excluded_invalid_capture | 137 | 76.1% |

## Dataset de clasificación (good / bad)

| Split | good | bad | total |
|-------|------|-----|-------|
| train | 3 | 11 | 14 |
| val | 1 | 3 | 4 |
| test | 1 | 2 | 3 |
| **TOTAL** | 5 | 16 | 21 |

## Criterios de curado (automático, basados en pipeline)

- **excluded_invalid_capture**: `capture_valid=False`, máscara inválida, clasificador sin crop, o todas las métricas a cero.
- **good**: `decision=PASA` con captura válida.
- **bad**: `decision=RECHAZA` con captura válida.
- **review**: `decision=REVISAR` con captura válida.

## Advertencia

> El curado es **automático** basado en las métricas del pipeline rule-based.
> No ha sido revisado imagen por imagen por un humano.
> Úsalo como dataset auxiliar para entrenamiento, no como verdad absoluta.
> Antes de entrenar, se recomienda revisar visualmente las muestras `bad` y `review`.

## Recomendación

- Usar `good` y `bad` para entrenamiento del clasificador de calidad.
- No usar `review` en entrenamiento (etiqueta ambigua).
- No usar `excluded_invalid_capture` en entrenamiento.
- Complementar con las peras propias ya etiquetadas en `data/samples_quality_controlled_test`.

## Rutas finales

- Curado: `data/quality_curated_fruits360_v1/`
- Clasificación: `data/quality_curated_fruits360_cls_v1/`
- Previews: `outputs/fruits360_curated_quality_preview/`