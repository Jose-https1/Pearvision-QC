# PearVision QC — Real-Time Camera App Pro V2
## Reporte técnico

**Fecha:** 2026-05-21  
**Etapa del proyecto:** Etapa 5 — Aplicación en tiempo real

---

## 1. Archivos creados

| Archivo | Descripción |
|---------|-------------|
| `scripts/camera_smoke_test_v2.py` | Test básico de cámara (sin modelo) |
| `scripts/pearvision_qc_realtime_camera_pro_v2.py` | Aplicación principal |
| `reports/pearvision_qc_realtime_camera_pipeline_audit_v2.md` | Auditoría del pipeline |
| `reports/pearvision_qc_realtime_camera_pro_v2_report.md` | Este reporte |
| `reports/pearvision_qc_realtime_camera_validation_checklist_v2.md` | Checklist de validación |
| `outputs/live_camera_qc_pro_v2/` | Carpeta de evidencias (creada al ejecutar) |

**No se ha modificado ningún modelo, V2, U3, ni quality_rules.yaml.**

---

## 2. Modelo usado

**Nombre:** U3 ROI/masked clean  
**Arquitectura:** MobileNetV3-Small (PyTorch / torchvision)  
**Clases:** `bad` (índice 0), `good` (índice 1)  
**Ruta:** `outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt`  
**Test accuracy:** 91.84%  

---

## 3. Cómo detecta la pera

La app usa **LAB distance masking** aplicado a baja resolución (1/4 del frame) para velocidad:

1. Reducir frame a 1/4 de resolución.
2. Muestrear 4 esquinas (8% del lado menor) → color de fondo por mediana.
3. Convertir a espacio de color LAB.
4. Distancia euclídea LAB de cada píxel al color de fondo estimado.
5. Umbral: `distancia > 25` → píxel de pera (foreground).
6. Escalar máscara al tamaño original.
7. Encontrar contornos; seleccionar el mayor.
8. Calcular `pear_area_ratio = área_contorno / área_frame`.
9. Si `ratio < 0.04` → SIN PERA. Si `ratio > 0.90` → MALA CAPTURA.

---

## 4. Cómo crea la máscara

La máscara a tamaño original se obtiene del paso 6 anterior. Se usa para:
- Dibujar el contorno sobre el vídeo.
- Mostrar la miniatura "Máscara".
- Guardar como evidencia.

El contorno principal se convierte en bounding box (`cv2.boundingRect`) y centroide (`cv2.moments`).

---

## 5. Cómo crea el ROI/masked clean (entrada a U3)

Función `make_gray_bg_clean(frame_bgr, size=224)`:

1. `cv2.resize(frame, (224, 224), INTER_LANCZOS4)`
2. Muestrear esquinas 12×12 px → color de fondo.
3. Máscara LAB con umbral 25 (igual que en entrenamiento).
4. `MORPH_CLOSE(ELLIPSE 5×5, iter=2)` + `MORPH_OPEN(ELLIPSE 5×5, iter=1)`.
5. Reemplazar fondo (`máscara==0`) con gris neutro `(128, 128, 128)` RGB.
6. Devolver PIL Image RGB 224×224.

Luego transforms de inferencia:
- `ToTensor()` + `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`.

---

## 6. Cómo decide PASA / REVISAR / RECHAZA

```
Sin pera (ratio < 0.04)          → SIN PERA
Captura dudosa (ratio > 0.90)    → REVISAR
Error de inferencia              → REVISAR
U3=GOOD  AND  p_good > 0.85      → PASA
U3=BAD   AND  p_bad >= 0.995     → RECHAZA
U3=BAD   AND  p_bad < 0.995      → REVISAR
Cualquier otro caso              → REVISAR
```

**Estabilización temporal:** ventana de N predicciones (default N=7).  
RECHAZA se muestra solo si tiene mayoría fuerte (≥ max(3, N/2) frames).  
PASA se muestra solo si tiene ≥ max(2, N/3) frames.

---

## 7. Datos técnicos mostrados en pantalla

Panel derecho muestra en tiempo real:

- `capture_status` · `instant_decision` · `stable_decision` · `smoothing_count`
- `u3_pred` · `p_good` · `p_bad` · `thr_good` · `thr_bad`
- `pear_area_ratio` · `bbox` · `mask_area_px` · `roi_size`
- `preprocessing_ms` · `inference_ms` · `total_latency_ms`
- `saved_count` · `last_saved` · `reason`

Barra de progreso p_good (verde) / p_bad (rojo) en la parte inferior del panel.

---

## 8. Teclas

| Tecla | Acción |
|-------|--------|
| Q / ESC | Salir |
| S | Guardar evidencia actual |
| P | Pausar / reanudar vídeo |
| R | Resetear smoothing temporal |
| H | Mostrar/ocultar ayuda de teclado |
| M | Mostrar/ocultar miniaturas técnicas |

---

## 9. Evidencias guardadas (al pulsar S)

Carpeta base: `outputs/live_camera_qc_pro_v2/`

```
frames_original/    ← frame original de la cámara
frames_overlay/     ← canvas completo del dashboard
masks/              ← máscara binaria de la pera
roi_processed/      ← imagen gray_bg_clean 224×224
snapshots/          ← JSON con todos los datos técnicos
live_predictions.csv ← CSV acumulativo de todas las evidencias
```

CSV con 30 columnas: timestamp, frame_id, decisiones, probabilidades, bboxes, latencias, etc.

---

## 10. Limitaciones

| Limitación | Descripción |
|-----------|-------------|
| Fondo complejo | Si el fondo tiene colores similares a la pera, la máscara falla |
| Iluminación extrema | Muy oscuro o sobreexpuesto puede impedir la detección |
| Encuadre | La pera debe cubrir entre 4% y 90% del frame |
| Modelo entrenado en Fruits-360 | Puede no generalizar a peras con formas muy distintas |
| Sin YOLO | No usa detector YOLO; solo LAB masking clásico |
| Cámara RGB única | No mide defectos subcutáneos ni parámetros internos |

---

## 11. Cómo ejecutar

```powershell
# Activar entorno
# (desde PyCharm o con el .venv del proyecto)

# Test de cámara (sin modelo):
python scripts/camera_smoke_test_v2.py --camera 0

# App completa:
python scripts/pearvision_qc_realtime_camera_pro_v2.py --camera 0

# App con inferencia más frecuente:
python scripts/pearvision_qc_realtime_camera_pro_v2.py --camera 0 --infer-every 3

# Modo carpeta (sin cámara):
python scripts/pearvision_qc_realtime_camera_pro_v2.py --image-folder data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3
```

---

## 12. Cómo demostrarlo en clase

1. **Abrir la app** con `--camera 0`.
2. **Sin pera** → mostrar que aparece "SIN PERA" en azul.
3. **Colocar pera sana** de supermercado → mostrar contorno verde, resultado "PASA".
4. **Mostrar el panel técnico**: p_good > 0.85, bbox visible, FPS > 10.
5. **Mostrar las miniaturas**: original / máscara / ROI crop / gray_bg_clean.
6. **Pulsar S** → mostrar que se guardan los archivos en `outputs/live_camera_qc_pro_v2/`.
7. **Modo carpeta**: usar imágenes del batch de validación para mostrar que el pipeline es idéntico.

---

## 13. Pruebas que debe hacer José

- [ ] La cámara abre correctamente al ejecutar la app
- [ ] Sin pera en cámara: aparece "SIN PERA" en azul en grande
- [ ] Con pera centrada: aparece contorno coloreado sobre la pera
- [ ] Resultado "PASA" en verde para pera sana de supermercado
- [ ] p_good y p_bad visibles en el panel técnico
- [ ] FPS visible y razonable (> 5)
- [ ] Las miniaturas (Original, Máscara, ROI crop, gray_bg_clean) son visibles
- [ ] Pulsar S guarda archivos en `outputs/live_camera_qc_pro_v2/`
- [ ] El CSV `live_predictions.csv` se actualiza con cada pulsación de S
- [ ] Pulsar P pausa/reanuda el vídeo
- [ ] Pulsar R resetea el smoothing
- [ ] Pulsar Q cierra la app sin dejar procesos colgados
- [ ] Modo carpeta genera CSV y contact sheet en `outputs/live_camera_qc_pro_v2/folder_test/`
