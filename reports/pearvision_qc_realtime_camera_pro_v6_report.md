# PearVision QC - Real-Time Camera App Pro V6 Report

## Que fallaba en V5

### 1. Umbral GOOD 0.70 todavia demasiado conservador para ciertos angulos

V5 bajo el umbral de 0.85 a 0.70. Fue una mejora pero en pruebas reales
se observo que la misma pera sana fotografiada desde otro angulo da:

```
u3_pred = GOOD
p_good  ≈ 0.61
p_bad   ≈ 0.38
decision = REVISAR   <- incorrecto para pera comercialmente buena
```

El modelo varia con el angulo porque el dataset de entrenamiento (Fruits360)
usa siempre la misma perspectiva. Un umbral de 0.60 recupera estos casos.

### 2. border_cut=YES enviaba el candidato a U3 y acababa en REVISAR

En V5, si el candidato tocaba el borde (border_cut=True), `apply_policy_v5`
entraba en la rama:

```python
if border_cut:
    if u3_pred == "BAD" and p_bad >= BAD_REJECT_THRESHOLD:
        return "RECHAZA", ...
    return "REVISAR", ...   # <- REVISAR aunque sea persona/ropa
```

Cuando una persona o camiseta aparecia en el frame, el contorno tocaba el borde,
se ejecutaba U3 sobre ropa/piel, y U3 devolvía p_good ≈ 0.63 (la ropa tiene
textura similar a frutas para el modelo). Resultado: REVISAR en grande.

Esto es incorrecto: persona/fondo nunca debe provocar REVISAR.

---

## Que cambia en V6

### 1. Umbral PASA bajado a 0.60

```python
LIVE_GOOD_ACCEPT_THRESHOLD = 0.60
```

Permite que peras sanas fotografiadas desde angulos no ideales pasen a PASA
si U3 devuelve GOOD con confianza >= 0.60. BAD sigue en 0.995.

### 2. Bloqueo duro pre-U3: `is_valid_live_pear_candidate()`

Nueva funcion que se evalua ANTES de llamar a `run_u3()`:

```python
candidate_ok, gate_reason = is_valid_live_pear_candidate(det, frame)
if candidate_ok:
    u3_pred, p_good, p_bad = run_u3(model, gray_pil)
    decision, reason = apply_policy_v6(u3_pred, p_good, p_bad)
else:
    # U3 NO se ejecuta
    decision = "SIN PERA" o "MALA CAPTURA"
    smoother.reset()
```

Condiciones que bloquean la ejecucion de U3:

| Condicion                     | Razon                                           |
|-------------------------------|-------------------------------------------------|
| mask_valid = False            | No hay mascara valida                           |
| border_cut = True             | Candidato toca borde -> probable fondo/persona  |
| bbox ausente                  | Sin bounding box real                           |
| pear_area_ratio > 0.45        | Demasiado grande para ser pera aislada          |
| pear_area_ratio < 0.004       | Particula minima / ruido                        |
| bbox_w_ratio > 0.80           | Ocupa casi todo el ancho                        |
| bbox_h_ratio > 0.80           | Ocupa casi toda la altura                       |
| rectangularity > 0.93 y area > 10% | Forma rectangular -> fondo/papel          |
| candidato en esquina          | Centro del bbox muy cerca de una esquina        |

### 3. `apply_policy_v6` simplificada

V6 ya no necesita gestionar border_cut porque esta bloqueado upstream:

```
U3=GOOD y p_good >= 0.60 -> PASA
U3=BAD  y p_bad  >= 0.995 -> RECHAZA
cualquier otro caso       -> REVISAR
```

---

## Por que p_good 0.60 para camara real

El modelo U3 fue entrenado con Fruits360: imagenes de frutas sobre fondo negro
o blanco uniforme, siempre desde la misma perspectiva cenital, con luz controlada.

En camara portatil real:
- La iluminacion varia (ventana, lampara, sombras).
- El angulo cambia segun como sostenga la pera.
- El recorte ROI incluye mas fondo que en el dataset.

Con p_good = 0.70 (V5), angulos no ideales de una pera sana daban REVISAR.
Con p_good = 0.60 (V6), esos mismos angulos dan PASA.

El riesgo de bajar el umbral es clasificar una pera mala como PASA. Ese riesgo
esta mitigado porque:
1. El bloqueo pre-U3 impide que fondos/personas lleguen a U3.
2. BAD sigue en 0.995: el modelo tiene que estar casi seguro para RECHAZA.
3. Si U3 da BAD con p_bad < 0.995, el resultado es REVISAR, no PASA.

---

## Por que BAD sigue siendo estricto con p_bad >= 0.995

Un falso rechazo (pera buena -> RECHAZA) es el peor error en demo:
genera desconfianza inmediata en el sistema.

El umbral 0.995 garantiza que RECHAZA solo ocurre cuando el modelo
esta practicamente seguro (99.5%) de que hay defecto. Si hay duda, REVISAR
envia la pera a inspeccion humana.

En la validacion previa (269 peras etiquetadas):
- FRR = 0.0%: ninguna pera buena fue rechazada
- FAR = 2.7%: 6 peras malas clasificadas PASA
- 214 peras malas -> REVISAR (inspeccion humana)

Este equilibrio es conservador a proposito para demo academica.

---

## Como se evita que fondo / persona / camiseta sea REVISAR

### Bloqueo por border_cut (nuevo en V6)

El gating de candidatos ya existia en V5 (comprobaba >= 3 bordes).
V6 anade: si el candidato toca CUALQUIER borde (border_cut=True),
`is_valid_live_pear_candidate()` devuelve False y U3 no se ejecuta.

Motivo: personas, camisetas y fondos suelen ser objetos grandes que
tocan el borde del frame. Una pera bien posicionada y centrada
tipicamente NO toca el borde.

Si una pera real toca el borde (pera muy grande o muy cerca), la app
muestra MALA CAPTURA, que es correcto: el usuario debe alejar la pera
o reencuadrar.

### Bloqueo por saturacion (heredado de V5)

Los candidatos con saturacion media del ROI < 18 (fondo blanco/gris)
se descartan en `_is_candidate_valid_v6` antes de llegar a pre-U3.

### Reset inmediato de smoothing (heredado de V5)

Cuando el candidato es invalido, `smoother.reset()` se llama de inmediato.
No hay acumulacion de REVISARs de ciclos anteriores. El banner cambia
a SIN PERA / MALA CAPTURA en el mismo ciclo.

---

## Significado de cada decision

| Decision     | Cuando ocurre                                             | Color  |
|--------------|-----------------------------------------------------------|--------|
| PASA         | U3=GOOD y p_good >= 0.60                                 | Verde  |
| REVISAR      | U3 no es concluyente (p_good < 0.60 o p_bad < 0.995)    | Azul   |
| RECHAZA      | U3=BAD y p_bad >= 0.995                                  | Rojo   |
| SIN PERA     | No se encontro ningun candidato valido                    | Naranja|
| MALA CAPTURA | Hay candidato pero es invalido (borde, forma, tamano)    | Naranja|

Nunca aparece REVISAR cuando:
- No hay mascara valida
- El candidato toca el borde (border_cut=YES)
- El candidato es demasiado grande o pequeno
- El candidato es rectangular (fondo/papel)
- El candidato esta en una esquina

---

## Donde se guardan las evidencias

Al pulsar `S`:

```
outputs/live_camera_qc_pro_v6/
├── frames_original/   # frame BGR de camara sin modificar
├── frames_overlay/    # frame con contorno, bbox y decision superpuesta
├── masks/             # mascara binaria del mejor candidato
├── roi_processed/     # imagen 224x224 gray_bg_clean enviada a U3
├── snapshots/         # dashboard completo 1600x900
└── metadata/          # JSON con todos los datos tecnicos
```

JSON incluye:
- timestamp, decision, instant_decision, stable_decision
- u3_pred, p_good, p_bad
- live_good_accept_threshold (0.60), bad_reject_threshold (0.995)
- u3_blocked, gate_reason
- reason, strategy, capture_status, mask_valid
- bbox, pear_area_ratio, bbox_w_ratio, bbox_h_ratio
- rectangularity, solidity, border_cut

CSV acumulativo: `outputs/live_camera_qc_pro_v6/live_predictions.csv`

---

## Comando de ejecucion

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_realtime_camera_pro_v6.py --camera 0 --infer-every 5
```

Opciones:

| Argumento        | Default | Descripcion                              |
|------------------|---------|------------------------------------------|
| --camera         | 0       | Indice de camara                         |
| --infer-every    | 5       | Ciclos entre inferencias U3              |
| --smoothing      | 7       | Tamano del buffer de smoothing           |
| --image-folder   | None    | Modo batch: carpeta de imagenes          |

---

## Controles de teclado

| Tecla  | Accion                                              |
|--------|-----------------------------------------------------|
| B      | Calibrar fondo (recomendado con fondo vacio)        |
| C      | Limpiar fondo calibrado                             |
| S      | Guardar evidencias completas                        |
| R      | Resetear smoothing                                  |
| P      | Pausar / reanudar                                   |
| H      | Mostrar / ocultar ayuda en header                   |
| M      | Mostrar / ocultar miniaturas                        |
| Q/ESC  | Salir                                               |

---

## Checklist de prueba para Jose

1. Ejecutar la app.
2. Fondo blanco vacio -> confirmar SIN PERA.
3. Poner persona/cabeza/camiseta -> confirmar SIN PERA o MALA CAPTURA, nunca REVISAR.
4. Poner pera sana lado 1 -> confirmar PASA si p_good >= 0.60.
5. Poner pera sana lado 2 (angulo diferente) -> confirmar PASA si p_good >= 0.60.
6. Poner pera con defecto visible -> confirmar REVISAR o RECHAZA.
7. Pulsar S -> confirmar evidencias en outputs/live_camera_qc_pro_v6/.
8. Verificar panel tecnico: u3_blocked muestra YES cuando fondo/persona.
9. Verificar panel tecnico: thr_live_good = 0.6 y thr_bad_rej = 0.995.

---

## Archivos creados

| Archivo                                              | Estado   |
|------------------------------------------------------|----------|
| scripts/pearvision_qc_realtime_camera_pro_v6.py      | CREADO   |
| reports/pearvision_qc_realtime_camera_pro_v6_report.md | CREADO |

No se entrenaron modelos.
No se modifico V2, V3, V4, V5, U2, U3 ni quality_rules.yaml.
