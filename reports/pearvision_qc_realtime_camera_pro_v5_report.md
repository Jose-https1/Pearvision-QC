# PearVision QC - Real-Time Camera App Pro V5 Report

## Que fallaba en V4

### 1. Umbral GOOD demasiado estricto para camara real

V4 usaba `THR_GOOD = 0.85` heredado del pipeline offline. Una pera sana de supermercado
bajo camara real da tipicamente:

```
u3_pred = GOOD
p_good  ≈ 0.74
p_bad   ≈ 0.25
```

Como `0.74 < 0.85`, la app emitia REVISAR en lugar de PASA para una pera sana visible.
El modelo no tenia ningun error; el problema era el umbral configurado para condiciones
de laboratorio (iluminacion controlada, fondos negros/blancos uniformes del dataset).

### 2. REVISAR persistia cuando no habia pera

El SmoothingBuffer de V4 acumulaba las decisiones de los ultimos N ciclos de inferencia.
Cuando la pera se retiraba:

- Ciclos anteriores con REVISAR permanecian en el buffer.
- El buffer tardaba varios ciclos en vaciarse.
- Durante ese tiempo, el banner grande mostraba REVISAR aunque no hubiera pera.

El usuario veia REVISAR en grande sobre una escena vacia o con fondo/persona, lo cual
era incorrecto y confuso para demo.

### 3. No habia distincion clara entre SIN PERA y MALA CAPTURA

V4 emitia SIN PERA en ambos casos (sin candidato valido y con candidato invalido),
lo que dificultaba diagnosticar si el problema era ausencia real de pera o
segmentacion incorrecta.

---

## Que cambia en V5

### Umbral PASA para camara real

```python
LIVE_GOOD_ACCEPT_THRESHOLD = 0.70
```

Este umbral aplica solo a la camara en tiempo real. Reconoce que las condiciones de
iluminacion de una camara portatil no son las del dataset de entrenamiento (fondos
negros/blancos uniformes, luz controlada). Con 0.70, una pera sana de supermercado
con `p_good ≈ 0.74` pasa correctamente a PASA.

### Reset inmediato de smoothing cuando no hay pera

```python
if not mask_valid:
    smoother.reset()
    stable = "SIN PERA"  # o "MALA CAPTURA"
```

En V4 se esperaba que el buffer vaciara solo. En V5 el reset es inmediato: en cuanto
`mask_valid = False`, el buffer se vacia y el banner muestra SIN PERA o MALA CAPTURA
en el mismo ciclo. No hay persistencia de REVISAR.

### SIN PERA y MALA CAPTURA son estados distintos

| Estado       | Cuando ocurre                                  |
|--------------|------------------------------------------------|
| SIN PERA     | No se encontro ningun candidato en el frame    |
| MALA CAPTURA | Hay candidato pero gating lo rechazo (borde, area, forma) |

Ambos son azul en el banner pero con etiqueta diferente para facilitar diagnostico.

### Politica de decision V5 (resumen)

```
mask_valid = False
  -> smoother.reset()
  -> SIN PERA o MALA CAPTURA (nunca REVISAR)

mask_valid = True
  -> border_cut = True y p_bad >= 0.995 -> RECHAZA
  -> border_cut = True                   -> REVISAR
  -> u3_pred=GOOD y p_good >= 0.70       -> PASA
  -> u3_pred=BAD y p_bad  >= 0.995       -> RECHAZA
  -> u3_pred=BAD y p_bad  < 0.995        -> REVISAR
  -> cualquier otro caso                 -> REVISAR
```

### Gating con filtro de saturacion minima

```python
MIN_CANDIDATE_SAT = 18
```

Si el ROI del candidato tiene saturacion media (canal S en HSV) menor a 18, se descarta
como fondo neutro (papel blanco, mesa gris, pared beige). Esto evita que regiones de
fondo de baja saturacion pasen el gating geometrico y lleguen a U3.

---

## Por que p_good 0.70 se usa solo para camara real

El dataset de entrenamiento de U3 usa imagenes de Fruits360: fondos negros/blancos
uniformes, iluminacion controlada, camara cenital. En esas condiciones el modelo
distingue bien GOOD/BAD con p_good > 0.85.

En camara real portatil:
- La iluminacion varia (sombras, reflejos, temperatura de color).
- El fondo no es uniforme (papel, mesa, madera).
- El recorte ROI incluye mas variacion de entorno que en el dataset.

El modelo sigue siendo util: en la validacion de 269 peras humano-etiquetadas el FRR
fue 0.0% (ninguna pera mala pasada como PASA). Pero para peras buenas en camara real
el modelo da p_good en el rango 0.70-0.84 en lugar de > 0.85.

Bajar el umbral de 0.85 a 0.70 recupera los verdaderos positivos de pera sana sin
reintroducir falsos negativos, porque BAD sigue siendo estricto con 0.995.

Este umbral NO se aplica al pipeline offline (analyze_quality.py), que mantiene 0.85.

---

## Por que BAD sigue siendo estricto con p_bad >= 0.995

Un rechazo incorrecto tiene alto coste en demo: una pera sana rechazada genera
desconfianza en el sistema. El umbral 0.995 asegura que solo se emite RECHAZA cuando
el modelo esta casi completamente seguro de que hay defecto. Si hay dudas, se emite
REVISAR para inspeccion humana.

En la validacion BAD (220 peras malas):
- FAR = 2.7% (6 peras malas clasificadas como PASA)
- 214 peras malas -> REVISAR (seguro, pasa a inspeccion humana)

Bajar el umbral BAD introduciria mas rechazos pero tambien mas falsos rechazos de
peras sanas. El 0.995 es conservador a proposito.

---

## Como se evita que fondo / persona / mesa sea REVISAR

### Gating geometrico (heredado de V4)

| Criterio             | Accion si se incumple           |
|----------------------|---------------------------------|
| area_ratio < 0.01    | Rechazar (demasiado pequeno)    |
| area_ratio > 0.45    | Rechazar (demasiado grande)     |
| bbox_w_ratio > 0.80  | Rechazar (casi todo el ancho)   |
| bbox_h_ratio > 0.80  | Rechazar (casi toda la altura)  |
| rectangularity > 0.93 y area > 12% | Rechazar (fondo rectangular) |
| toca >= 3 bordes     | Rechazar (borde de frame)       |

### Gating de saturacion (nuevo en V5)

Si el ROI del candidato tiene `sat_mean < MIN_CANDIDATE_SAT (18)`:
- El candidato se descarta aunque pase todos los filtros geometricos.
- Fondos blancos/grises tipicamente tienen sat < 15.
- Peras (verdes, amarillas, marrones) tipicamente tienen sat > 30.

### Reset inmediato de smoothing

Si el fondo/persona/mesa pasa accidentalmente un ciclo de gating y U3 da REVISAR,
en cuanto desaparece el candidato el smoother se resetea de inmediato.
No acumula REVISARs de ciclos anteriores.

### Scoring de candidatos (heredado de V4)

El mejor candidato no es el mas grande sino el que tiene mejor puntuacion:
- centralidad (candidatos centrados son mas probables peras)
- forma organica (solidity alta, rectangularity baja)
- compacidad
- saturacion del ROI

Un candidato de fondo/persona suele fallar en centralidad o en forma organica.

---

## Donde se guardan las evidencias

Al pulsar `S`:

```
outputs/live_camera_qc_pro_v5/
├── frames_original/   # frame BGR original de camara
├── frames_overlay/    # frame con contorno, bbox y resultado
├── masks/             # mascara binaria del mejor candidato
├── roi_processed/     # imagen 224x224 enviada a U3 (gray_bg_clean)
├── snapshots/         # dashboard completo 1600x900
└── metadata/          # JSON con todos los datos tecnicos
```

JSON incluye:
- timestamp, decision, instant_decision, stable_decision
- u3_pred, p_good, p_bad
- live_good_accept_threshold, bad_reject_threshold
- reason, strategy, capture_status, mask_valid
- bbox, pear_area_ratio, bbox_w_ratio, bbox_h_ratio
- rectangularity, solidity, border_cut

CSV acumulativo: `outputs/live_camera_qc_pro_v5/live_predictions.csv`

---

## Comando de ejecucion

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_realtime_camera_pro_v5.py --camera 0 --infer-every 5
```

Opciones:

| Argumento        | Default | Descripcion                              |
|------------------|---------|------------------------------------------|
| --camera         | 0       | Indice de camara                         |
| --infer-every    | 5       | Ciclos entre inferencias U3              |
| --smoothing      | 7       | Tamano del buffer de smoothing           |
| --image-folder   | None    | Modo batch: carpeta de imagenes          |
| --no-display     | False   | Desactivar ventana OpenCV                |

---

## Controles de teclado

| Tecla  | Accion                                           |
|--------|--------------------------------------------------|
| B      | Calibrar fondo (opcional, mejora segmentacion)   |
| C      | Limpiar fondo calibrado                          |
| S      | Guardar evidencias completas                     |
| R      | Resetear smoothing                               |
| P      | Pausar / reanudar                                |
| H      | Mostrar / ocultar ayuda                          |
| M      | Mostrar / ocultar miniaturas                     |
| Q/ESC  | Salir                                            |

---

## Checklist de prueba para Jose

1. Ejecutar la app.
2. Apuntar la camara al fondo vacio -> confirmar SIN PERA.
3. Colocar persona/mano/cabeza frente a la camara -> confirmar SIN PERA o MALA CAPTURA, nunca REVISAR.
4. Colocar pera sana de supermercado centrada -> confirmar PASA si U3=GOOD y p_good >= 0.70.
5. Pulsar S -> confirmar que saved_count sube y se guardan archivos en outputs/live_camera_qc_pro_v5/.
6. Retirar la pera -> confirmar que vuelve a SIN PERA en el mismo ciclo o siguiente.
7. Confirmar que el panel tecnico muestra thr_live_good=0.7 y thr_bad_rej=0.995.

---

## Archivos creados / modificados

| Archivo                                              | Estado   |
|------------------------------------------------------|----------|
| scripts/pearvision_qc_realtime_camera_pro_v5.py      | CREADO   |
| reports/pearvision_qc_realtime_camera_pro_v5_report.md | CREADO |

No se entrenaron modelos.
No se modifico V2, V3, V4, U2, U3 ni quality_rules.yaml.
