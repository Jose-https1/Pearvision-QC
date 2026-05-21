# Fruits-360 Quality Classifier V2 — Reporte de Entrenamiento

Fecha: 2026-05-18 14:35
Status: **PASS**

## Configuracion V2

| Parametro | Valor |
|-----------|-------|
| Modelo | MobileNetV3-small |
| Pesos pretrained | ImageNet (IMAGENET1K_V1) |
| Dataset | data/quality_fruits360_human_v2/ |
| Train | 187 imgs (good=34, bad=153) |
| Val   | 40 imgs |
| Test  | 40 imgs |
| Epochs | 40 |
| Batch size | 32 |
| LR | 0.0001 |
| Optimizer | Adam + CosineAnnealingLR |
| Device | cuda |
| Seed | 42 |
| Desbalance | class_weights + WeightedRandomSampler |

## Cambios V1 -> V2

| ID | Cambio |
|----|--------|
| F360_0198 | BAD -> REVIEW (excluida) |
| F360_0052 | GOOD -> REVIEW (excluida) |
| F360_0060 | GOOD -> GOOD (confirmada, sin cambio) |
| Epochs | 30 -> 40 |

## Mejor checkpoint: epoch 37  (val F1-macro=0.8000)

## Resultados TEST

| Metrica | V1 | V2 | Delta |
|---------|----|----|-------|
| Accuracy | 0.8049 | 0.9250 | +0.1201 |
| F1-macro | 0.7355 | 0.8769 | +0.1414 |

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| bad | 0.969 | 0.939 | 0.954 |
| good | 0.750 | 0.857 | 0.800 |

## Matriz de confusion (TEST)

```
                bad      good
       bad        31         2
      good         1         6
```

## Archivos generados

- `outputs/fruits360_quality_cls_v2/best_model.pt`
- `outputs/fruits360_quality_cls_v2/training_curves.png`
- `outputs/fruits360_quality_cls_v2/confusion_matrix_val.png`
- `outputs/fruits360_quality_cls_v2/confusion_matrix_test.png`
- `outputs/fruits360_quality_cls_v2/train_log.csv`
- `outputs/fruits360_quality_cls_v2/test_results.csv`

## Confirmaciones

- analyze_quality.py NO fue modificado.
- quality_rules.yaml NO fue modificado.
- Imagenes REVIEW NO usadas.
- Test set evaluado UNA SOLA VEZ al final.
- Dataset V1 NO fue destruido.
- Tiempo total: 86.2 s