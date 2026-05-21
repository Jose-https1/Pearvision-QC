# PearVision QC - Real-Time Camera App Pro V4 Fix Report

## Problema raiz detectado en V3

En V3, el detector `detect_pear_v3` funcionaba así:

1. Generaba UNA sola mascara (distancia LAB a esquinas)
2. Elegía el contorno MAS GRANDE de esa mascara
3. Aplicaba gating para rechazarlo si era demasiado grande

El problema: en cámara real sobre fondo blanco, el ruido de iluminación, sombras y variaciones hacían que la mascara LAB capturara el FONDO ENTERO como objeto. El contorno más grande era `bbox: (88,0,1192,720)` con `pear_area_ratio: 0.90`. El gating rechazaba correctamente ese contorno → `MALA_CAPTURA` → pero no había fallback para encontrar la pera real.

Resultado: con una pera visible, la app decía `SIN PERA` o `MALA_CAPTURA`.

---

## Solución en V4 — Segmentación multi-estrategia

### Estrategia 1: Mascara por saturacion HSV (primaria)

```python
mask = cv2.inRange(hsv, (0, 30, 35), (180, 255, 242))
```

Lógica: peras tienen COLOR (saturacion > 30). Fondos blancos/grises tienen saturacion baja (S < 20). Esta mascara rechaza papel blanco, mesa gris, sombras oscuras y regiones muy brillantes.

Funciona sin calibrar fondo.

### Estrategia 2: Mascara por rangos de color de pera

Rangos HSV que cubren los colores tipicos de peras:

| Color | H | S | V |
|-------|---|---|---|
| Verde pera | 25-90 | >30 | >50 |
| Amarillo | 15-30 | >40 | >100 |
| Marron/russet | 5-25 | >20 | 40-200 |
| Naranja-amarillo | 10-22 | >50 | >100 |
| Verde palido | 40-100 | 15-100 | >100 |

### Estrategia 3: Diferencia con fondo calibrado (opcional)

Si el usuario pulsa B, se usa diferencia absoluta LAB con eliminacion de sombras. Activa `[BG:OK]` en el header.

### Seleccion de candidatos

Cada mascara genera componentes conectados. Cada componente se valida:

| Criterio | Umbral | Accion |
|----------|--------|--------|
| area_ratio | < 0.01 o > 0.45 | rechazar |
| bbox_w_ratio | > 0.80 | rechazar |
| bbox_h_ratio | > 0.80 | rechazar |
| rectangularity | > 0.93 y area > 12% | rechazar |
| touches_border | >= 3 lados | rechazar |

Los candidatos validos se PUNTUAN (no se elige simplemente el mas grande):

```
score = area_score * 0.25
      + center_score * 0.15    # preferir candidatos centrados
      + shape_score * 0.25     # solidez * (1 - penalizacion rectangularidad)
      + compact_score * 0.10   # compacidad (4*pi*area/perim^2)
      + aspect_score * 0.10    # aspect ratio cercano a 1.0
      + sat_score * 0.15       # saturacion media del ROI (pera tiene color)
      - border_penalty         # penalizar si toca bordes
```

Se elige el candidato con mayor score.

### ROI para U3

V4 recorta el bbox de la pera (+ 25% margen) antes de pasarlo a U3, en lugar de usar el frame completo. Esto centra la pera en el input de 224×224 y mejora la precision de la estimacion de fondo.

---

## Caso que falla en V3 y se resuelve en V4

| Metrica | V3 (problema) | V4 (solucion) |
|---------|--------------|----------------|
| Mascara generada | Fondo completo (LAB corners falla) | Solo region con saturacion/color |
| Candidato elegido | Max area = fondo | Max score = pera |
| area_ratio | 0.90 | 0.05-0.25 (tipico pera) |
| cap_status | MALA_CAPTURA | OK |
| mask_valid | NO | YES |
| U3 ejecutado | NO | SI |
| Resultado | SIN PERA | PASA / REVISAR / RECHAZA |

---

## Controles de teclado

| Tecla | Accion |
|-------|--------|
| B | Calibrar fondo. Si se detecta pera al pulsar, muestra WARNING (no bloquea). |
| C | Limpiar fondo calibrado. Vuelve a modo sin fondo. |
| S | Guardar evidencias completas. |
| R | Resetear smoothing. |
| P | Pausar/reanudar. |
| H | Mostrar/ocultar ayuda. |
| M | Mostrar/ocultar miniaturas. |
| Q/ESC | Salir. |

### Aviso de calibracion incorrecta

Si el usuario pulsa B con la pera visible, V4:
- Igualmente guarda el fondo (el usuario puede querer hacerlo)
- Muestra WARNING en naranja: "WARNING: pera detectada al calibrar fondo?"
- No bloquea la deteccion
- El usuario puede pulsar C y volver a calibrar sin pera

---

## Funcionamiento sin fondo calibrado

V4 funciona sin pulsar B. Las estrategias 1 y 2 (saturacion + color) son activas por defecto.

El header muestra `[BG:OFF]` en gris cuando no hay fondo calibrado.

Recomendacion: calibrar fondo con B para mejor segmentacion en fondos no uniformes.

---

## Como interpretar las decisiones

| Decision | Significado |
|----------|-------------|
| SIN PERA | No se encontro ningun candidato valido en el frame |
| REVISAR | Captura dudosa (MALA_CAPTURA), borde cortado, o U3 con confianza media |
| PASA | U3=GOOD con p_good > 0.85 |
| RECHAZA | U3=BAD con p_bad >= 0.995 |

### Razon de REVISAR

La razon se muestra en el panel tecnico:

- `BAD_CAPTURE_OR_INVALID_MASK`: gating rechazo la mascara
- `BORDER_CUT`: pera cortada en borde (no se da PASA)
- `NO_VALID_PEAR_DETECTED`: varios frames consecutivos sin pera valida
- `U3=BAD p_bad=... < 0.995`: defecto detectado pero no con certeza suficiente para RECHAZAR

---

## Panel tecnico — nuevas metricas en V4

| Campo | Descripcion |
|-------|-------------|
| strategy | mascara usada: sat, color, bg_diff |
| solidity | area / convex_hull. Pera tipica: 0.85-0.95 |
| rectangularity | area / bbox. Alta en fondos rectangulares |
| bbox_w_ratio | bbox_w / frame_w. Bloqueado si > 0.80 |
| bbox_h_ratio | bbox_h / frame_h. Bloqueado si > 0.80 |
| border_cut | YES si pera toca alguno de los bordes |

---

## Evidencias al pulsar S

Carpeta: `outputs/live_camera_qc_pro_v4/`

| Subcarpeta | Contenido |
|------------|-----------|
| frames_original/ | Frame BGR original |
| frames_overlay/ | Frame con contorno + bbox + etiqueta |
| masks/ | Mascara binaria del mejor candidato |
| roi_processed/ | gray_bg_clean 224x224 enviado a U3 |
| snapshots/ | Dashboard completo 1600x900 |
| metadata/ | JSON con todas las metricas |

CSV acumulativo: `live_predictions.csv`

---

## Limitaciones

1. **Fondos con color similar a la pera** (madera amarilla, tela verde): pueden generar falsos candidatos. El scoring por centralidad y forma los penaliza, pero puede confundir en casos extremos. Usar calibracion de fondo (B) para estos casos.

2. **Peras muy palidas o muy oscuras**: si la saturacion es < 30, pueden escapar a la mascara por saturacion. La mascara por color de pera las captura parcialmente.

3. **Umbral MAX_AREA_RATIO = 0.45**: una pera muy cerca de la camara puede superar este umbral. Alejar ligeramente la camara o ajustar el umbral.

4. **GrabCut no implementado**: se decidio no usar GrabCut para evitar latencia y comportamiento impredecible. Las mascaras morfologicas son suficientes para la mayoria de casos.

---

## Comandos de ejecucion

```powershell
# App en tiempo real
python scripts/pearvision_qc_realtime_camera_pro_v4.py --camera 0 --infer-every 5

# Smoothing mas suave
python scripts/pearvision_qc_realtime_camera_pro_v4.py --camera 0 --infer-every 3 --smoothing 11

# Validar sintaxis
.\.venv\Scripts\python.exe -m py_compile scripts\pearvision_qc_realtime_camera_pro_v4.py

# Test carpeta
python scripts/pearvision_qc_realtime_camera_pro_v4.py --image-folder data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3
```

---

## Checklist de prueba recomendada para Jose

1. Ejecutar la app.
2. Apuntar la camara a la escena SIN pera.
3. Confirmar que muestra SIN PERA.
4. Pulsar B para calibrar fondo (opcional pero recomendado).
5. Colocar la pera en el centro de la camara.
6. Confirmar que el contorno verde rodea SOLO la pera.
7. Confirmar que mask_valid = YES y capture_status = OK.
8. Confirmar que U3 se ejecuta (u3_pred = GOOD o BAD).
9. Confirmar decision PASA / REVISAR / RECHAZA.
10. Quitar la pera. Confirmar que vuelve a SIN PERA en ~3 ciclos.
11. Pulsar S. Confirmar que saved_count sube y snapshot se guarda.
12. Revisar outputs/live_camera_qc_pro_v4/.

---

*V4 creado sin entrenar ni modificar ningun modelo.*
*No se modifico V2, V3, U3, ni quality_rules.yaml.*
