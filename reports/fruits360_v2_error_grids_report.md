# Fruits-360 Quality V2 — Error Grids

Fecha: 2026-05-18 14:45

## Resumen TEST V2

| Metrica | Valor |
|---------|-------|
| Total imagenes test | 40 |
| Correctas | 37 |
| Errores totales | 3 |
| Accuracy | 0.9250 |
| FP bad->good | 2 |
| FN good->bad | 1 |

## Errores FP (bad predicha como good)

| Filename | Conf | Prob_bad | Prob_good |
|----------|------|----------|-----------|
| F360_0224.jpg | 0.988 | 0.012 | 0.988 |
| F360_0107.jpg | 0.568 | 0.432 | 0.568 |

## Errores FN (good predicha como bad)

| Filename | Conf | Prob_bad | Prob_good |
|----------|------|----------|-----------|
| F360_0060.jpg | 0.996 | 0.996 | 0.004 |

## Archivos generados

- `outputs/fruits360_quality_cls_v2/test_all_grid.jpg`
- `outputs/fruits360_quality_cls_v2/test_errors_grid.jpg`
- `outputs/fruits360_quality_cls_v2/test_errors_detail.csv`
- `outputs/fruits360_quality_cls_v2/bad_as_good_errors_grid.jpg`
- `outputs/fruits360_quality_cls_v2/good_as_bad_errors_grid.jpg`

## Confirmaciones

- NO se entrenó ningun modelo.
- analyze_quality.py NO fue modificado.
- quality_rules.yaml NO fue modificado.
- Se uso el modelo best_model.pt V2 ya existente.