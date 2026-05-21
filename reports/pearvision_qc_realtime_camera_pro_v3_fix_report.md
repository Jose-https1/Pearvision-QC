# PearVision QC - Real-Time Camera App Pro V3 Fix Report

## Problemas encontrados en V2

| # | Problema | Síntoma observado |
|---|----------|-------------------|
| 1 | Fondo/habitación detectado como pera | `pear_area_ratio: 0.7672`, `bbox: (0,0,1112,720)`, `p_good: 0.8742` con la cámara apuntando a la habitación vacía |
| 2 | Máscara incluye sombras y fondo | El contorno rodeaba toda la escena en lugar de solo la fruta |
| 3 | Bbox ocupa todo el ancho del frame | `bbox_w = 1112` con frame `w = 1280` → ratio 0.87 |
| 4 | Tecla S no guardaba snapshot del dashboard | Solo guardaba `last_canvas` (=overlay), no el panel completo |
| 5 | Caracteres raros en pantalla (`???`) | Textos con tildes/eñes no soportados por OpenCV HERSHEY fonts |
| 6 | Smoothing mantenía PASA al quitar la pera | El buffer no se limpiaba al detectar SIN PERA consecutivos |

---

## Soluciones aplicadas en V3

### TAREA 2 — Texto ASCII puro

Todos los `cv2.putText` usan únicamente caracteres ASCII:

- `Percepción` → `Percepcion`
- `Cámara` → `Camara`
- `Resolución` → `Resolucion`
- `Máscara` → `Mascara`
- `Revisión` → `Revision`
- Guiones largos `─` → `---`
- Símbolos especiales eliminados

### TAREA 3 — Calibración de fondo vacío

Nueva variable `bg_frame` (BGR, resolución original):

- **B** → captura `last_frame` como fondo calibrado.
- **C** → limpia el fondo calibrado.
- El header muestra `[BG:OK]` (verde) o `[BG:---]` (naranja).
- Cuando hay fondo calibrado, `_segment_with_background()` usa diferencia LAB absoluta + eliminación de sombras (baja saturación + más oscuro que fondo).

### TAREA 4 y 8 — Gating estricto

Función `detect_pear_v3()` aplica reglas antes de ejecutar U3:

```
MAX_BBOX_W_RATIO   = 0.75  # bbox_w / frame_w
MAX_BBOX_H_RATIO   = 0.75  # bbox_h / frame_h
MAX_PEAR_AREA      = 0.45  # area_contorno / area_frame
MIN_PEAR_AREA      = 0.04
MAX_RECTANGULARITY = 0.92  # alto => fondo rectangular
```

Reglas (en orden):
1. Sin contornos → SIN_PERA
2. `pear_area_ratio < 0.04` → SIN_PERA (objeto demasiado pequeño)
3. `bbox_w/fw > 0.75` O `bbox_h/fh > 0.75` O `area > 0.45` → MALA_CAPTURA (bloquea U3)
4. Toca >= 3 bordes del frame → MALA_CAPTURA
5. Atraviesa dimensión completa (left+right ó top+bottom) → MALA_CAPTURA
6. `rectangularity > 0.92` y `area > 15%` → MALA_CAPTURA (fondo rectangular)
7. `mask_valid = False` → no se ejecuta U3

El caso de V2 (`bbox: (0,0,1112,720)`, `ratio=0.7672`) queda bloqueado por la regla 3.

### TAREA 5 — Máscara mejorada

**Con fondo calibrado** (`_segment_with_background`):
1. Diferencia absoluta frame vs fondo en espacio LAB
2. Threshold `dist > 18`
3. Eliminación de sombras: píxeles con saturación HSV < 25 y más oscuros que el fondo
4. Morfología: close (k=7, iter=3) + open (k=7, iter=2)

**Sin fondo calibrado** (`_segment_lab_corners`):
- Igual que V2: distancia LAB al color de esquinas (mismo umbral=25)
- Solo cambia el gating posterior

El preprocesado para U3 (`make_gray_bg_clean`) **no cambia** — debe coincidir exactamente con el entrenamiento.

### TAREA 6 — Guardado con S

`EvidenceSaverV3` guarda en `outputs/live_camera_qc_pro_v3/`:

| Subcarpeta | Contenido |
|------------|-----------|
| `frames_original/` | Frame BGR original de la cámara |
| `frames_overlay/` | Frame con contorno + bbox + etiqueta de decisión |
| `masks/` | Máscara de segmentación (BGR) |
| `roi_processed/` | gray_bg_clean 224×224 (entrada a U3) |
| `snapshots/` | **Dashboard completo** (1600×900 canvas) |
| `metadata/` | JSON técnico por guardado |

CSV acumulativo: `live_predictions.csv`

Tras guardar:
- `saved_count` sube en la pantalla
- `last_saved` muestra el nombre base del snapshot
- Se imprime `SAVED evidence: <ruta_snapshot>`
- Mensaje visual en la pantalla durante 3 segundos

Si falla: se muestra `SAVE ERROR: <detalle>` en rojo.

### TAREA 7 — Estado correcto SIN PERA

`sinpera_streak` cuenta ciclos de inferencia consecutivos sin pera válida.

Cuando `sinpera_streak >= 3` (configurable con `SINPERA_RESET_STREAK`):
- `smoother.reset()` — limpia el buffer de smoothing
- `stable = "SIN PERA"`
- `u3_pred = "--"`, `p_good = 0.0`, `p_bad = 0.0`
- `reason = "NO_VALID_PEAR_DETECTED"`
- Panel muestra "N/A" para p_good, p_bad, u3_pred
- Barra de probabilidades se vacía

Cuando vuelve una pera válida: `sinpera_streak = 0`.

---

## Cómo calibrar el fondo (tecla B)

1. Ejecutar la app: `python scripts/pearvision_qc_realtime_camera_pro_v3.py --camera 0 --infer-every 5`
2. Apuntar la cámara al fondo vacío (mesa, bandeja, etc.) **sin ninguna pera**.
3. Pulsar **B** → aparece `[BG:OK]` en verde en el header y mensaje "Fondo calibrado".
4. Colocar la pera → la detección usará diferencia respecto al fondo capturado.
5. Para limpiar: pulsar **C**.

La calibración de fondo mejora significativamente la segmentación porque:
- Evita que el modelo de esquinas falle cuando el fondo no es uniforme
- Elimina sombras automáticamente
- Reduce falsos positivos en escenas con objetos de fondo

---

## Cómo guardar evidencias (tecla S)

Pulsar **S** en cualquier momento guarda:
- Frame original
- Frame con overlay (contorno + decisión)
- Máscara de segmentación
- ROI gray_bg_clean (entrada a U3)
- **Snapshot completo del dashboard** (panel entero 1600×900)
- JSON técnico con todas las métricas
- Fila en CSV acumulativo

Los archivos se guardan en `outputs/live_camera_qc_pro_v3/` con timestamp.

---

## Cómo interpretar las decisiones

| Decisión | Significado | Acción recomendada |
|----------|-------------|-------------------|
| **SIN PERA** | No se detectó objeto válido | Ninguna — esperar pera |
| **PASA** | U3 clasifica GOOD con p_good > 0.85 | Pera aceptada |
| **REVISAR** | Confianza media, captura dudosa o borde cortado | Inspección manual |
| **RECHAZA** | U3 clasifica BAD con p_bad >= 0.995 | Pera rechazada |

### Casos especiales de REVISAR

- `BAD_CAPTURE_OR_INVALID_MASK`: el gating bloqueó la captura (objeto demasiado grande, rectangular, etc.)
- `BORDER_CUT`: la pera está cortada en un borde del frame — no se puede dar PASA
- `U3_INFERENCE_ERROR`: fallo de PyTorch (raro)
- `NO_VALID_PEAR_DETECTED`: sin pera durante varios frames consecutivos

---

## Limitaciones

1. **Sin fondo calibrado**: la segmentación por esquinas puede fallar con fondos no uniformes o de color similar a la pera.
2. **Gating estricto**: peras muy cerca de la cámara (área > 45% del frame) dan REVISAR, no PASA.
3. **Defectos internos**: el sistema solo detecta defectos superficiales visibles.
4. **Umbral MAX_PEAR_AREA = 0.45**: diseñado para evitar falsos positivos; puede requerir ajuste según la distancia de captura habitual.
5. **Texto ASCII**: los nombres de archivos y paths con caracteres Unicode (tildes) se manejan con `cv2.imencode + pathlib.write_bytes` para compatibilidad en Windows.

---

## Comandos de ejecución

```powershell
# App en tiempo real (camara 0, inferencia cada 5 frames)
python scripts/pearvision_qc_realtime_camera_pro_v3.py --camera 0 --infer-every 5

# App con smoothing mas largo (mas estable, mas lento)
python scripts/pearvision_qc_realtime_camera_pro_v3.py --camera 0 --infer-every 3 --smoothing 11

# Test de carpeta (sin camara)
python scripts/pearvision_qc_realtime_camera_pro_v3.py --image-folder data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3

# Solo compilar (smoke test)
python -m py_compile scripts/pearvision_qc_realtime_camera_pro_v3.py
```

---

## Archivos generados

```
outputs/live_camera_qc_pro_v3/
├── frames_original/     # Frames BGR originales
├── frames_overlay/      # Frames con contorno + decision
├── masks/               # Mascaras de segmentacion
├── roi_processed/       # gray_bg_clean 224x224 (entrada U3)
├── snapshots/           # Dashboard completo 1600x900
├── metadata/            # JSON por guardado
└── live_predictions.csv # CSV acumulativo

outputs/live_camera_qc_pro_v3/folder_test/
├── predictions.csv
├── contact_sheet_all.jpg
├── contact_sheet_review_reject.jpg
└── summary.txt
```

---

*Generado automaticamente. No se entreno ni modifico ningun modelo.*
