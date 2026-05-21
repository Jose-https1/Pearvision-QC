# Train U3 ROI Masked Clean Report

**Fecha:** 2026-05-21

## Configuracion

- Modelo: MobileNetV3-small pretrained ImageNet
- Epochs: hasta 40 con early stopping patience=8
- Mejor epoch: 18
- LR: 0.0001, WeightDecay: 0.0001
- Device: cpu
- Augmentations: HorizontalFlip, Rotation(12), ColorJitter

## Dataset

- Train: 215 (good=62, bad=153)
- Val: 45
- Test: 49

## Resultados test

| Metrica | Valor |
|---|---|
| Accuracy | 0.9184 (45/49) |
| Confusion matrix bad | TP=33, FN=1 (bad->good) |
| Confusion matrix good | TN=12, FP=3 (good->bad) |
| BAD->GOOD (falso negativo) | 1/34 (2.9%) |
| GOOD->BAD (falso positivo) | 3/15 (20.0%) |
| Val accuracy (best epoch) | 1.000 (45/45) |

Nota: sklearn no disponible en .venv; F1 y metricas derivadas calculadas desde confusion matrix manual.
Confusion matrix val: [[32,0],[0,13]] — perfecta.

## Archivos

- outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt
- outputs/fruits360_quality_cls_u3_roi_masked_clean/last_model.pt
- outputs/fruits360_quality_cls_u3_roi_masked_clean/train_log.csv
- outputs/fruits360_quality_cls_u3_roi_masked_clean/test_results.csv
- outputs/fruits360_quality_cls_u3_roi_masked_clean/confusion_matrix_test.png
- outputs/fruits360_quality_cls_u3_roi_masked_clean/confusion_matrix_val.png
- outputs/fruits360_quality_cls_u3_roi_masked_clean/training_curves.png
- outputs/fruits360_quality_cls_u3_roi_masked_clean/test_errors_grid.jpg
- outputs/fruits360_quality_cls_u3_roi_masked_clean/test_all_grid.jpg

## Notas

- No se modifico V2.
- No se modifico analyze_quality.py ni quality_rules.yaml.
- No se integro U3 en el pipeline final.
