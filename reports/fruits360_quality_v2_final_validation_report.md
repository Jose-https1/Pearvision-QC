# Fruits-360 Quality Classifier V2 — Reporte Final de Validación

Fecha: 2026-05-20  
Estado: **CERRADO — BASELINE VALIDADO**

---

## 1. Resumen del estado de V2

El clasificador binario de calidad de peras V2 ha sido entrenado, evaluado y revisado manualmente.
Tras la última revisión humana de los errores residuales, se confirma que **no hay correcciones
adicionales que aplicar**. V2 queda congelado como baseline actual del proyecto.

---

## 2. Dataset usado

`data/quality_fruits360_human_v2/`

| Split | Imágenes |
|-------|----------|
| train | 187 (good=34, bad=153) |
| val   | 40 |
| test  | 40 |
| **Total** | **267** |

Imágenes excluidas (clase REVIEW, no usadas en entrenamiento):
- F360_0198: movida a REVIEW (era BAD, caso ambiguo)
- F360_0052: movida a REVIEW (era GOOD, caso ambiguo)

---

## 3. Configuración del modelo

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | MobileNetV3-small |
| Pesos pretrained | ImageNet (IMAGENET1K_V1) |
| Epochs | 40 |
| Batch size | 32 |
| LR | 0.0001 |
| Optimizer | Adam + CosineAnnealingLR |
| Desbalance | class_weights + WeightedRandomSampler |
| Mejor checkpoint | epoch 37 (val F1-macro = 0.8000) |
| Modelo guardado | `outputs/fruits360_quality_cls_v2/best_model.pt` |

---

## 4. Métricas de V2 (test set, 40 imágenes)

| Métrica | V1 | V2 | Delta |
|---------|----|----|-------|
| Accuracy | 0.8049 | 0.9250 | +0.1201 |
| F1-macro | 0.7355 | 0.8769 | +0.1414 |

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| bad   | 0.969 | 0.939 | 0.954 |
| good  | 0.750 | 0.857 | 0.800 |

Matriz de confusión (TEST):

```
              bad      good
   bad         31         2
  good          1         6
```

---

## 5. Revisión manual de los 3 errores residuales

Tras revisar manualmente los 3 errores del test set, el usuario confirma:

| ID | Ground Truth | Predicción | Error type | Veredicto humano |
|----|-------------|------------|------------|-----------------|
| F360_0107 | **BAD** | good (conf=0.568) | bad→good | Etiqueta BAD correcta — error del modelo |
| F360_0224 | **BAD** | good (conf=0.988) | bad→good | Etiqueta BAD correcta — error del modelo |
| F360_0060 | **GOOD** | bad (conf=0.996) | good→bad | Etiqueta GOOD correcta — error del modelo |

Fuente de verificación: `outputs/fruits360_quality_cls_v2/test_errors_detail.csv`

### Conclusión

- Los 3 errores son **errores genuinos del clasificador**, no errores de etiquetado.
- El ground truth de los 3 casos es correcto tal y como está.
- **No se aplica ninguna corrección adicional de etiquetas.**
- V2 queda congelado como baseline actual.

---

## 6. Confirmaciones de integridad

| Elemento | Estado |
|---------|--------|
| Dataset V2 (`data/quality_fruits360_human_v2/`) | NO modificado |
| Labels del dataset V2 | NO modificados |
| Modelo V2 (`best_model.pt`) | NO reentrenado |
| `scripts/analyze_quality.py` | NO modificado |
| `configs/quality_rules.yaml` | NO modificado |
| `data/fruits360_human_review/human_labels_template.csv` | NO modificado |
| Dataset V1 | NO destruido |

---

## 7. Próximos pasos

Para seguir iterando sobre el clasificador con datos reales no vistos:

1. Copiar imágenes nuevas en `data/unseen_quality_eval_input/`
2. Ejecutar `scripts/predict_fruits360_quality_v2_on_folder.py`
3. Revisar `outputs/fruits360_quality_unseen_eval_example/human_review_template.csv`
4. Anotar errores encontrados y decidir si se construye V3 con esos datos

Ver flujo completo en: `reports/fruits360_quality_unseen_eval_workflow_report.md`
