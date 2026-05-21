# Auditoría de Pipeline — PearVision QC Real-Time Camera App V2

**Fecha:** 2026-05-21  
**Propósito:** Documentar el pipeline exacto que se implementa en la app de cámara en tiempo real.

---

## 1. Modelo usado

**Arquitectura:** MobileNetV3-Small (PyTorch / torchvision)  
**Cabeza:** `Linear(in_features, 2)` — 2 clases: `bad` (idx 0) y `good` (idx 1)  
**Pesos base usados en entrenamiento:** ImageNet1K_V1 (fine-tuning)  
**Pesos finales:** `outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt`  
**Thresholds:** `outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json`

---

## 2. Ruta del modelo

```
outputs/
└── fruits360_quality_cls_u3_roi_masked_clean/
    ├── best_model.pt              ← pesos del clasificador U3
    └── selected_thresholds.json  ← calibración original (bad=0.6, good=0.55)
```

El modelo se carga con `weights=None` y luego `load_state_dict` desde `best_model.pt`.

---

## 3. Preprocesado usado — gray_bg_clean

El modelo U3 fue entrenado con imágenes en formato **gray_bg_clean**: pera visible con fondo reemplazado por gris neutro `(128, 128, 128)` RGB.

**Pasos exactos:**

1. `cv2.resize(frame, (224, 224), INTER_LANCZOS4)`
2. Muestrear 4 esquinas de 12×12 px → mediana → color de fondo estimado
3. Convertir imagen a espacio LAB
4. Distancia euclídea LAB entre cada pixel y color de fondo
5. Umbral: `dist > 25` → foreground (pera)
6. Limpieza morfológica:
   - `MORPH_CLOSE(ELLIPSE 5×5, iters=2)`
   - `MORPH_OPEN(ELLIPSE 5×5, iters=1)`
7. Reemplazar pixels donde máscara==0 con `(128, 128, 128)`
8. Devolver como PIL Image RGB 224×224

**Normalización (transforms):**

```python
Resize((224, 224))
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## 4. Umbrales usados en la app de cámara

La app de cámara usa umbrales más conservadores que la calibración original:

| Variable         | Valor   | Procedencia               |
|------------------|---------|---------------------------|
| threshold_good   | 0.85    | Política de cámara V2     |
| threshold_bad    | 0.995   | Política de cámara V2     |
| min_pear_ratio   | 0.04    | Heurístico de detección   |
| max_pear_ratio   | 0.90    | Heurístico de detección   |

*Los thresholds del JSON (bad=0.6, good=0.55) son de la calibración de validación; la app de cámara usa valores más estrictos para reducir falsos rechazos en demo.*

---

## 5. Decisión final — política implementada

```
capture_status == SIN_PERA     → decision = SIN PERA
capture_status == MALA_CAPTURA → decision = REVISAR
u3_pred == ERROR               → decision = REVISAR
p_good > 0.85                  → decision = PASA
p_bad >= 0.995                 → decision = RECHAZA
p_bad < 0.995 (u3=BAD)        → decision = REVISAR
else                           → decision = REVISAR
```

**Estabilización temporal:** ventana de N predicciones (por defecto N=7).  
RECHAZA requiere mayoría fuerte (≥ 3 frames) antes de mostrarse.

---

## 6. Riesgos actuales

| Riesgo | Mitigación |
|--------|------------|
| Fondo complejo (mesa con objetos) | Corner sampling puede fallar → máscara ruidosa → REVISAR |
| Pera muy pegada al borde del frame | ratio > 0.90 → MALA_CAPTURA → REVISAR |
| Iluminación muy oscura o sobreexpuesta | LAB distance < 25 → sin máscara → SIN PERA |
| Cámara desenfocada | Laplacian variance baja; se puede detectar y enviar a REVISAR |
| Objetos con color similar a la pera | Máscara ruidosa; contorno más grande puede no ser pera |
| Fondo muy oscuro (pera amarilla brillante) | Corner sampling estima fondo oscuro bien; funciona OK |

---

## 7. Lo que se implementa en la app de cámara

- Detección de pera en tiempo real con LAB distance masking (a 1/4 resolución para velocidad)
- Creación de gray_bg_clean 224×224 compatible con entrenamiento U3
- Inferencia U3 cada N frames (configurable, default=5)
- Estabilización temporal con deque de N decisiones
- Dashboard profesional con fondo oscuro: cámara, métricas, resultado grande
- Miniaturas: original, máscara, ROI crop, gray_bg_clean
- Guardado de evidencias en `outputs/live_camera_qc_pro_v2/`
- Modo carpeta para test sin cámara

---

## 8. Validación de resultados previos

| Métrica | Resultado validado |
|---------|-------------------|
| FRR (pera buena rechazada) | 0.0% (49 GOOD → 0 rechazadas) |
| FAR (pera mala pasa) | 2.7% (6/220 BAD → PASA) |
| BAD → REVISAR | 214/220 → capturadas como dudosas |
| Test accuracy U3 | 91.84% |
| Holdout 22 peras supermarket | 22/22 PASA |

*Fuente: evaluate_final_u3_pipeline_bad_regression_v1.py ejecutado 2026-05-21*
