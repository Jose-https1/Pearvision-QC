# PearVision QC — Checklist de Validación de Aplicación V2

**Fecha:** 2026-05-21  
**App:** `scripts/pearvision_qc_realtime_camera_pro_v2.py`

---

## Cómo usar este checklist

Ejecutar la app y marcar cada ítem como ✅ (correcto), ❌ (fallo), o ⚠️ (parcial).

```powershell
python scripts/pearvision_qc_realtime_camera_pro_v2.py --camera 0
```

---

## Sección 1 — Inicio y cámara

- [ ] La app arranca sin errores de importación
- [ ] La cámara abre correctamente (mensaje `[OK] Cámara 0: 1280×720`)
- [ ] El modelo U3 carga correctamente (mensaje `[OK] U3 cargado: best_model.pt`)
- [ ] La ventana `PearVision QC` aparece en pantalla
- [ ] El vídeo de la cámara es fluido (no congelado)
- [ ] El FPS se muestra en la cabecera y es > 5

---

## Sección 2 — Estado sin pera

- [ ] Sin pera en cámara: `capture_status = SIN_PERA`
- [ ] Sin pera: resultado grande muestra **SIN PERA** en color azul/acero
- [ ] Sin pera: `pear_area_ratio` es cercano a 0
- [ ] Sin pera: no aparece contorno ni bounding box sobre el vídeo

---

## Sección 3 — Detección de pera

- [ ] Al colocar una pera centrada: aparece un contorno coloreado alrededor de la pera
- [ ] El bounding box (rectángulo) rodea aproximadamente la pera
- [ ] Se dibuja el centroide (punto) en el centro de la pera detectada
- [ ] La miniatura "Máscara" muestra claramente la pera en verde/blanco sobre fondo oscuro
- [ ] La miniatura "gray_bg_clean" muestra la pera sobre fondo gris (224×224)
- [ ] La miniatura "ROI crop" muestra el recorte de la pera
- [ ] `pear_area_ratio` es razonable (entre 0.05 y 0.80 para pera bien encuadrada)

---

## Sección 4 — Inferencia y resultados

- [ ] `u3_pred` cambia entre GOOD / BAD según la pera mostrada
- [ ] `p_good` y `p_bad` son valores entre 0.0 y 1.0
- [ ] `p_good + p_bad ≈ 1.0` (suma de softmax)
- [ ] `preprocessing_ms` se muestra y es > 0
- [ ] `inference_ms` se muestra y es > 0
- [ ] El resultado grande (PASA / REVISAR / RECHAZA / SIN PERA) es legible y grande
- [ ] El color del resultado coincide: verde=PASA, naranja=REVISAR, rojo=RECHAZA, azul=SIN PERA
- [ ] La barra inferior p_good/p_bad se actualiza visualmente

---

## Sección 5 — Decisiones de política

- [ ] **Pera sana de supermercado** → resultado **PASA** (p_good > 0.85)
- [ ] **Pera claramente mala / podrida** → resultado **RECHAZA** o **REVISAR**
  - RECHAZA si p_bad >= 0.995
  - REVISAR si U3=BAD pero p_bad < 0.995
- [ ] **Imagen borrosa o sin pera clara** → resultado **REVISAR** o **SIN PERA**
- [ ] La máscara no corta agresivamente la pera (se ve la pera completa en la miniatura)

---

## Sección 6 — Estabilización temporal

- [ ] El resultado no parpadea con cambios rápidos de un solo frame
- [ ] `smoothing_count` se incrementa hasta el tamaño de ventana (default 7)
- [ ] `instant_decision` puede diferir de `stable_decision` (el grande es el estable)
- [ ] Al pulsar **R**: `smoothing_count` vuelve a 0 y el resultado se reinicia

---

## Sección 7 — Controles de teclado

- [ ] **Q** o **ESC**: cierra la app correctamente, sin dejar procesos abiertos
- [ ] **P**: pausa el vídeo (aparece "PAUSADO" en la pantalla)
- [ ] **P** de nuevo: reanuda el vídeo
- [ ] **R**: resetea el smoothing (visible en `smoothing_count = 0`)
- [ ] **H**: oculta/muestra la línea de ayuda de teclado en la cabecera
- [ ] **M**: oculta/muestra la zona de miniaturas técnicas

---

## Sección 8 — Guardado de evidencias (tecla S)

- [ ] Al pulsar **S**: aparece mensaje en consola `[S] Guardado: YYYYMMDD_...`
- [ ] Se crea archivo en `outputs/live_camera_qc_pro_v2/frames_original/`
- [ ] Se crea archivo en `outputs/live_camera_qc_pro_v2/frames_overlay/`
- [ ] Se crea archivo en `outputs/live_camera_qc_pro_v2/masks/`
- [ ] Se crea archivo en `outputs/live_camera_qc_pro_v2/roi_processed/`
- [ ] Se crea archivo JSON en `outputs/live_camera_qc_pro_v2/snapshots/`
- [ ] El CSV `outputs/live_camera_qc_pro_v2/live_predictions.csv` se actualiza
- [ ] `saved_count` se incrementa en el panel técnico
- [ ] Los archivos son legibles y no están corruptos

---

## Sección 9 — Modo carpeta (sin cámara)

```powershell
python scripts/pearvision_qc_realtime_camera_pro_v2.py --image-folder data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3
```

- [ ] La app procesa las imágenes sin abrir ventana de cámara
- [ ] Se genera `outputs/live_camera_qc_pro_v2/folder_test/folder_predictions.csv`
- [ ] Se genera `outputs/live_camera_qc_pro_v2/folder_test/contact_sheet.jpg`
- [ ] Los overlays muestran contorno, bbox y resultado con el color correcto
- [ ] Las decisiones son consistentes con las de `analyze_quality.py`

---

## Sección 10 — Cierre limpio

- [ ] Al pulsar Q: la cámara se libera (`cap.release()`)
- [ ] La ventana OpenCV se cierra (`cv2.destroyAllWindows()`)
- [ ] No quedan procesos Python en background tras cerrar
- [ ] El CSV no queda corrupto tras cerrar (es posible abrirlo en Excel/LibreOffice)

---

## Resultado global

| Sección | Estado |
|---------|--------|
| 1 — Inicio y cámara | ⬜ |
| 2 — Estado sin pera | ⬜ |
| 3 — Detección de pera | ⬜ |
| 4 — Inferencia | ⬜ |
| 5 — Política de decisiones | ⬜ |
| 6 — Estabilización temporal | ⬜ |
| 7 — Controles de teclado | ⬜ |
| 8 — Guardado de evidencias | ⬜ |
| 9 — Modo carpeta | ⬜ |
| 10 — Cierre limpio | ⬜ |

**Decisión:** ⬜ APTO PARA DEMO  /  ⬜ REQUIERE AJUSTES

---

*Checklist completado por: _______________  Fecha: _______________*
