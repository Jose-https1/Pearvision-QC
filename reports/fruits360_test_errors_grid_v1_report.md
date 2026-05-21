# Fruits-360 Test Errors Grid v1

Fecha: 2026-05-18 14:18

## Resumen TEST

| Metrica | Valor |
|---------|-------|
| Total imagenes test | 41 |
| Correctas | 33 |
| Errores | 8 |
| Accuracy | 0.8049 |
| FP (bad pred como good) | 6 |
| FN (good pred como bad) | 2 |

## Errores FP — bad clasificada como good

| Filename | Conf | Prob_bad | Prob_good |
|----------|------|----------|-----------|
| F360_0198.jpg | 0.995 | 0.005 | 0.995 |
| F360_0223.jpg | 0.857 | 0.143 | 0.857 |
| F360_0253.jpg | 0.824 | 0.176 | 0.824 |
| F360_0107.jpg | 0.648 | 0.352 | 0.648 |
| F360_0096.jpg | 0.559 | 0.441 | 0.559 |
| F360_0227.jpg | 0.511 | 0.489 | 0.511 |

## Errores FN — good clasificada como bad

| Filename | Conf | Prob_bad | Prob_good |
|----------|------|----------|-----------|
| F360_0060.jpg | 0.957 | 0.957 | 0.043 |
| F360_0052.jpg | 0.665 | 0.665 | 0.335 |

## Archivos generados

- `outputs/fruits360_quality_cls_v1/test_errors_grid.jpg`
- `outputs/fruits360_quality_cls_v1/test_all_grid.jpg`
- `outputs/fruits360_quality_cls_v1/test_predictions_all.csv`
- `outputs/fruits360_quality_cls_v1/test_errors_detail.csv`

## Confirmaciones

- NO se entrenó ningun modelo.
- analyze_quality.py NO fue modificado.
- quality_rules.yaml NO fue modificado.
- Se uso el modelo best_model.pt ya existente.