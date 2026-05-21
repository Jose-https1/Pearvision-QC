# Fruits-360 Quality Classifier v1 — Reporte de Entrenamiento

Fecha: 2026-05-18 14:10
Status: **PASS**

## Configuracion

| Parametro | Valor |
|-----------|-------|
| Modelo | MobileNetV3-small |
| Pesos pretrained | ImageNet (IMAGENET1K_V1) |
| Dataset | data/quality_fruits360_human_v1/ |
| Train | 188 imgs (good=34, bad=154) |
| Val | 40 imgs |
| Test | 41 imgs |
| Epochs | 30 |
| Batch size | 32 |
| LR | 0.0001 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR |
| Device | cuda |
| Seed | 42 |
| Desbalance compensado | class_weights en loss + WeightedRandomSampler |

## Mejor checkpoint

- Epoca: 29  (criterio: F1-macro en val)
- Val loss: 0.3532
- Val accuracy: 0.8500
- Val F1-macro: 0.7656

## Resultados en VAL (mejor epoch)

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| bad | 0.935 | 0.879 | 0.906 |
| good | 0.556 | 0.714 | 0.625 |

## Resultados en TEST (evaluacion final)

- Test accuracy : 0.8049
- Test F1-macro : 0.7355

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| bad | 0.931 | 0.818 | 0.871 |
| good | 0.500 | 0.750 | 0.600 |

## Matriz de confusion (TEST)

```
                bad      good
       bad        27         6
      good         2         6
```

## Archivos generados

- `outputs\fruits360_quality_cls_v1/best_model.pt`
- `outputs\fruits360_quality_cls_v1/training_curves.png`
- `outputs\fruits360_quality_cls_v1/confusion_matrix_val.png`
- `outputs\fruits360_quality_cls_v1/confusion_matrix_test.png`
- `outputs\fruits360_quality_cls_v1/train_log.csv`

## Advertencia de desbalance

Ratio good:bad ≈ 1:4.5. Se aplicaron:
- class_weights inversamente proporcionales a la frecuencia de clase
- WeightedRandomSampler para equilibrar batches durante el entrenamiento

Para entrenamiento futuro considerar adicionalmente:
- Augmentacion fuerte solo en la clase `good`
- Mixup / CutMix

## Confirmaciones

- analyze_quality.py NO fue modificado.
- quality_rules.yaml NO fue modificado.
- Imagenes REVIEW NO fueron usadas.
- Test set usado UNA SOLA VEZ al final.
- Tiempo total de entrenamiento: 64.9 s