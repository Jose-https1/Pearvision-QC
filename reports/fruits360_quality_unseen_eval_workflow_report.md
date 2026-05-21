# Flujo de Evaluación en Imágenes Nuevas — Fruits-360 Quality V2

Fecha: 2026-05-20  
Estado: **LISTO PARA USAR**

---

## 1. Script creado

`scripts/predict_fruits360_quality_v2_on_folder.py`

Permite evaluar imágenes nuevas (no vistas durante el entrenamiento) con el clasificador V2
sin tocar el modelo ni el dataset.

---

## 2. Cómo se usa

### Uso básico (carpetas por defecto)

```powershell
python scripts/predict_fruits360_quality_v2_on_folder.py
```

### Carpetas personalizadas

```powershell
python scripts/predict_fruits360_quality_v2_on_folder.py --input <ruta_entrada> --output <ruta_salida>
```

---

## 3. Carpeta de entrada por defecto

`data/unseen_quality_eval_input/`

Copiar aquí imágenes `.jpg`, `.jpeg` o `.png`.  
Ver instrucciones detalladas en `data/unseen_quality_eval_input/README.md`.

---

## 4. Carpeta de salida por defecto

`outputs/fruits360_quality_unseen_eval_example/`

Se crea automáticamente si no existe.

---

## 5. Archivos que genera

| Archivo | Descripción |
|---------|------------|
| `predictions.csv` | Imagen, predicción, confianza, prob_good, prob_bad |
| `human_review_template.csv` | Plantilla para revisión humana (columnas human_label y notes vacías) |
| `contact_sheet_all.jpg` | Hoja de contacto con todas las imágenes evaluadas |
| `contact_sheet_pred_good.jpg` | Solo imágenes predichas como GOOD |
| `contact_sheet_pred_bad.jpg` | Solo imágenes predichas como BAD |
| `summary.txt` | Resumen de la ejecución (n imágenes, GOOD/BAD, tiempo) |

### Columnas de predictions.csv

| Columna | Descripción |
|---------|------------|
| image | Nombre del archivo |
| image_path | Ruta absoluta |
| pred_label | `good` o `bad` |
| confidence | Probabilidad máxima (0–1) |
| prob_good | Probabilidad de clase GOOD |
| prob_bad | Probabilidad de clase BAD |

### Columnas de human_review_template.csv

| Columna | Descripción |
|---------|------------|
| image | Nombre del archivo |
| image_path | Ruta absoluta |
| pred_label | Predicción del modelo |
| confidence | Confianza del modelo |
| human_label | **Vacío — rellenar a mano** (good / bad) |
| notes | **Vacío — anotaciones opcionales** |

---

## 6. Ejemplo de comando completo

```powershell
cd "C:\Users\joser\Desktop\Sistemas de Percepción y Visión Artificial\computer vision"

# Copiar imágenes nuevas
# (copiar manualmente archivos a data/unseen_quality_eval_input/)

# Ejecutar predicción
python scripts/predict_fruits360_quality_v2_on_folder.py

# Ver resultados
# outputs/fruits360_quality_unseen_eval_example/
```

---

## 7. Comportamiento con carpeta vacía

Si `data/unseen_quality_eval_input/` está vacía, el script:
- No falla.
- Crea `summary.txt` con un aviso de que no había imágenes.
- No genera CSV ni contact sheets.

---

## 8. Modelo utilizado

`outputs/fruits360_quality_cls_v2/best_model.pt`

MobileNetV3-small fine-tuned con ImageNet, 2 clases (bad, good), eval transforms:
`Resize(224) → ToTensor → Normalize(ImageNet mean/std)`.

---

## 9. Confirmaciones de integridad

- NO se reentrenó ningún modelo.
- NO se modificó `outputs/fruits360_quality_cls_v2/best_model.pt`.
- NO se modificó `data/quality_fruits360_human_v2/`.
- NO se modificó `scripts/analyze_quality.py`.
- NO se modificó `configs/quality_rules.yaml`.
- V2 queda intacto como baseline actual.

---

## 10. Flujo de iteración sugerido

```
1. Acumular imágenes nuevas en data/unseen_quality_eval_input/
2. Ejecutar predict_fruits360_quality_v2_on_folder.py
3. Abrir human_review_template.csv → anotar human_label
4. Identificar errores reales del modelo
5. Decidir si construir V3 con esos datos adicionales
```

Ver estado de V2 y errores revisados en:
`reports/fruits360_quality_v2_final_validation_report.md`
