
# PearVision QC

Sistema de inspección visual de calidad de peras desarrollado como proyecto integrador de visión artificial.

PearVision QC permite analizar peras mediante visión artificial para apoyar una decisión de control de calidad superficial:

- `PASA`
- `REVISAR`
- `RECHAZA`
- `SIN PERA`
- `MALA CAPTURA`

El sistema combina técnicas clásicas de visión artificial, segmentación, reglas de decisión, validación de captura y un clasificador neuronal ligero basado en MobileNetV3-small.

El objetivo final del proyecto es que la cámara del portátil se abra en tiempo real, detecte correctamente si hay una pera válida en la escena, segmente la fruta, analice su estado superficial y muestre una interfaz visual clara, profesional y tecnológica con la decisión final.

---

## Alcance del proyecto

Este proyecto evalúa únicamente defectos superficiales visibles en peras.

El sistema NO mide:

- sabor,
- dulzor,
- índice Brix,
- firmeza,
- maduración interna,
- textura interna,
- calidad nutricional,
- defectos internos no visibles,
- contaminación química,
- seguridad alimentaria completa.

El prototipo trabaja con imagen RGB convencional, por lo que su alcance queda limitado a defectos externos observables con cámara.

---

## Motivación

En el sector hortofrutícola, una parte importante de la clasificación comercial depende del aspecto visual de la fruta.

Muchas frutas se descartan o se separan porque no cumplen criterios estéticos o comerciales, aunque algunas podrían seguir siendo válidas para venta secundaria, transformación, zumos, mermeladas, alimentación animal u otros usos.

En el caso concreto de las peras, defectos como podredumbre visible, golpes, manchas oscuras severas, necrosis o daños superficiales pueden justificar una separación de la línea comercial principal.

Sin embargo, también existen casos ambiguos:

- russeting natural,
- lenticelas,
- variaciones normales de color,
- sombras,
- marcas leves,
- reflejos,
- iluminación irregular,
- formas varietales distintas.

PearVision QC intenta resolver este problema como prototipo académico de bajo coste:

- usando una cámara RGB normal,
- ejecutando todo en local,
- sin depender de servicios cloud,
- mostrando resultados explicables,
- generando evidencias visuales,
- y separando claramente los casos aceptados, rechazados y dudosos.

---

## Estado final del proyecto

| Elemento | Estado |
|---|---|
| Aplicación de cámara en tiempo real | Funcional |
| Interfaz visual profesional | Funcional |
| Segmentación de pera | Funcional en condiciones controladas |
| Clasificación `PASA / REVISAR / RECHAZA` | Funcional |
| Detección de fondo sin pera | Funcional |
| Bloqueo de personas/manos/fondos incorrectos | Funcional como `SIN PERA` o `MALA CAPTURA` |
| Guardado de evidencias | Funcional |
| API/interfaz web local | Funcional |
| Validación con dataset corregido | Realizada |
| Validación con peras de supermercado | Realizada |
| Estado final académico | Aceptado como prototipo funcional |

---

## Resultado principal

El sistema final V6 funciona correctamente bajo condiciones de captura controladas:

- fondo claro,
- buena iluminación,
- una sola pera visible,
- pera completa dentro del encuadre,
- sin manos ni objetos adicionales,
- cámara estable.

En esas condiciones, el sistema:

1. detecta si hay una pera válida,
2. genera máscara de la pera,
3. calcula métricas técnicas,
4. ejecuta el clasificador U3,
5. aplica reglas de seguridad,
6. muestra resultado en grande,
7. guarda evidencias cuando se pulsa `S`.

---

## Tabla de contenidos

1. [Fases del proyecto](#fases-del-proyecto)
2. [Resumen técnico](#resumen-técnico)
3. [Pipeline final](#pipeline-final)
4. [Decisiones del sistema](#decisiones-del-sistema)
5. [Resultados validados](#resultados-validados)
6. [Aplicación V6 de cámara en tiempo real](#aplicación-v6-de-cámara-en-tiempo-real)
7. [API e interfaz web local](#api-e-interfaz-web-local)
8. [Análisis por lotes](#análisis-por-lotes)
9. [Instalación](#instalación)
10. [Dependencias](#dependencias)
11. [Estructura del repositorio](#estructura-del-repositorio)
12. [Modelos y artefactos necesarios](#modelos-y-artefactos-necesarios)
13. [Evidencias generadas](#evidencias-generadas)
14. [Validación funcional](#validación-funcional)
15. [Limitaciones](#limitaciones)
16. [Uso recomendado para evaluación](#uso-recomendado-para-evaluación)
17. [Conclusión](#conclusión)

---

# Fases del proyecto

## Fase 1 — Búsqueda y preparación de imágenes

La primera fase consistió en reunir imágenes de peras y preparar una base de trabajo para entrenar y validar el sistema.

Se utilizaron distintas fuentes:

- imágenes propias de peras compradas en supermercado,
- imágenes de datasets externos,
- imágenes de peras sanas,
- imágenes de peras con defectos,
- capturas de prueba en condiciones reales,
- casos difíciles o hard examples.

El objetivo no era solamente reunir muchas imágenes, sino cubrir situaciones relevantes:

- pera sana con buena iluminación,
- pera sana con russeting natural,
- pera con manchas oscuras,
- pera con podredumbre visible,
- pera con daño fuerte,
- pera pequeña,
- pera parcialmente dudosa,
- fondo vacío,
- persona en cámara,
- mano u objeto no pera.

---

## Fase 2 — Tratamiento y transformación de imágenes

La segunda fase se centró en transformar las imágenes para que el modelo aprendiera la fruta y no el fondo.

Se aplicaron técnicas como:

- recorte de ROI,
- segmentación de la pera,
- eliminación o neutralización del fondo,
- sustitución del fondo por gris neutro,
- normalización de tamaño,
- preparación de tensores de entrada,
- generación de datasets train/val/test,
- revisión manual de etiquetas,
- corrección de ruido de etiqueta.

El preprocesado más importante para U3 fue `gray_bg_clean`.

La idea principal fue:

```text
imagen original
    -> detección/segmentación de la pera
    -> eliminación del fondo
    -> fondo gris neutro
    -> entrada 224x224
    -> clasificador U3
````

Esto reduce el riesgo de que el modelo aprenda información irrelevante del fondo.

---

## Fase 3 — Entrenamiento del modelo

La tercera fase consistió en entrenar un clasificador ligero de calidad visual.

El modelo final utilizado fue:

```text
MobileNetV3-small
```

Nombre interno:

```text
U3
```

Clases:

```text
GOOD
BAD
```

El modelo recibe una imagen de la pera ya preprocesada y devuelve:

```text
p_good
p_bad
```

Estas probabilidades no se usan de forma aislada, sino dentro de una lógica de decisión conservadora.

---

## Fase 4 — Inferencia

La cuarta fase consistió en integrar el modelo en un pipeline de inferencia.

La inferencia no es simplemente “clasificar una imagen”.

El pipeline comprueba:

* si hay una pera,
* si la máscara es válida,
* si el objeto toca los bordes,
* si el objeto parece demasiado rectangular,
* si la pera es demasiado grande o pequeña,
* si la captura es usable,
* si el modelo tiene confianza suficiente,
* si debe aceptar, revisar o rechazar.

La salida final se muestra como:

```text
PASA
REVISAR
RECHAZA
SIN PERA
MALA CAPTURA
```

---

## Fase 5 — Aplicación

La quinta fase consistió en construir una aplicación real de cámara.

La aplicación final V6 permite:

* abrir la cámara del portátil,
* analizar la escena en tiempo real,
* mostrar la máscara de segmentación,
* mostrar datos técnicos,
* mostrar una decisión grande y clara,
* guardar evidencias,
* funcionar como demo profesional.

También se añadió una opción de API/interfaz web local para consultar el sistema desde otro dispositivo en la misma red.

---

# Resumen técnico

## Componentes principales

| Componente           | Función                                                 |
| -------------------- | ------------------------------------------------------- |
| OpenCV               | Captura de cámara, procesado de imagen, interfaz visual |
| Segmentación clásica | Separar la pera del fondo                               |
| HSV/Lab              | Detección de color, fondo y regiones sospechosas        |
| MobileNetV3-small    | Clasificador neuronal GOOD/BAD                          |
| PyTorch              | Inferencia del modelo U3                                |
| FastAPI              | API/interfaz web local                                  |
| Uvicorn              | Servidor local                                          |
| CSV/JSON             | Guardado de resultados y evidencias                     |
| Contact sheets       | Validación visual rápida                                |

---

## Filosofía de diseño

El sistema no busca forzar siempre una decisión automática.

La política final es conservadora:

* si la pera está claramente bien, `PASA`;
* si la pera está claramente mal, `RECHAZA`;
* si hay duda, `REVISAR`;
* si no hay pera, `SIN PERA`;
* si la captura no es válida, `MALA CAPTURA`.

Esto es importante porque en un sistema de calidad real es preferible revisar manualmente un caso dudoso antes que aceptar una fruta mala o rechazar una fruta buena.

---

# Pipeline final

```text
Entrada de cámara o imagen
        |
        v
Captura RGB
        |
        v
Validación básica de escena
        |
        |-- No hay candidato válido --> SIN PERA
        |-- Escena no fiable ---------> MALA CAPTURA
        |
        v
Segmentación de candidato
        |
        v
Validación de máscara
        |
        |-- Máscara inválida ---------> REVISAR / MALA CAPTURA
        |
        v
Recorte ROI
        |
        v
Preprocesado gray_bg_clean
        |
        v
Clasificador U3
        |
        v
Obtención de p_good y p_bad
        |
        v
Reglas de decisión
        |
        |-- p_good alto -------------> PASA
        |-- p_bad muy alto ----------> RECHAZA
        |-- caso dudoso -------------> REVISAR
        |
        v
Interfaz visual + guardado de evidencia
```

---

## Pipeline explicado paso a paso

### 1. Captura de imagen

La aplicación captura frames en tiempo real desde la cámara.

Por defecto se usa:

```text
camera = 0
resolution = 1280x720
```

---

### 2. Validación de escena

Antes de ejecutar el modelo, el sistema intenta determinar si la imagen tiene sentido.

Se filtran casos como:

* fondo vacío,
* persona en pantalla,
* mano en pantalla,
* objeto demasiado grande,
* objeto tocando bordes,
* máscara rectangular,
* captura inestable,
* mala iluminación,
* pera fuera del encuadre.

---

### 3. Segmentación de pera

El sistema intenta crear una máscara binaria de la pera.

La máscara permite:

* aislar la fruta,
* eliminar fondo,
* calcular área,
* generar ROI,
* mostrar overlay,
* preparar entrada para U3.

---

### 4. Preprocesado del ROI

El ROI se procesa para que el clasificador vea principalmente la pera.

El fondo se neutraliza para reducir ruido.

Entrada final del clasificador:

```text
224 x 224 x 3
```

---

### 5. Clasificador U3

U3 estima:

```text
p_good
p_bad
```

Ejemplo:

```text
p_good = 0.8953
p_bad  = 0.1047
```

---

### 6. Política de decisión

La política final V6 en cámara usa:

```text
LIVE_GOOD_ACCEPT_THRESHOLD = 0.60
BAD_REJECT_THRESHOLD = 0.995
```

Esto significa:

```text
si p_good >= 0.60 -> PASA
si p_bad >= 0.995 -> RECHAZA
si no -> REVISAR
```

Siempre que la captura y la máscara sean válidas.

---

### 7. Interfaz visual

La interfaz muestra:

* frame principal,
* contorno de la pera,
* máscara,
* ROI,
* entrada procesada para U3,
* datos técnicos,
* probabilidades,
* umbrales,
* latencia,
* decisión grande.

---

### 8. Guardado de evidencia

Al pulsar `S`, se guardan:

* imagen original,
* overlay,
* máscara,
* ROI procesado,
* snapshot completo,
* JSON de metadatos.

---

# Decisiones del sistema

## PASA

`PASA` significa que el sistema considera que la pera es visualmente aceptable bajo las condiciones del prototipo.

No significa que la fruta sea perfecta.

Significa que no se han encontrado evidencias suficientes para enviarla a revisión o rechazo.

---

## REVISAR

`REVISAR` significa que el sistema no tiene suficiente seguridad.

Puede aparecer por:

* marca pequeña,
* iluminación dudosa,
* forma extraña,
* confianza intermedia,
* posible daño,
* máscara imperfecta,
* pera parcialmente compleja,
* incertidumbre del modelo.

`REVISAR` no es un fallo.

Es una salida segura.

---

## RECHAZA

`RECHAZA` significa que el sistema detecta una señal muy fuerte de defecto.

En la política final, el rechazo automático es exigente porque se evita rechazar peras buenas.

---

## SIN PERA

`SIN PERA` significa que no hay una pera válida en la escena.

Puede aparecer con:

* fondo vacío,
* persona en cámara,
* objeto no pera,
* zona blanca sin fruta.

---

## MALA CAPTURA

`MALA CAPTURA` significa que hay algo en la imagen, pero la escena no cumple condiciones mínimas.

Puede ocurrir si:

* la pera está cortada,
* toca el borde,
* la cámara está mal colocada,
* aparece una persona,
* aparece una mano,
* el objeto no parece una pera,
* la segmentación es incorrecta.

---

# Resultados validados

## Validación Fase 1

Se validó el pipeline rule-based inicial contra expectativas humanas.

| Métrica            | Valor |
| ------------------ | ----: |
| Total expectativas |     8 |
| PASS               |     8 |
| FAIL               |     0 |
| NOT FOUND          |     0 |

Resultado:

```text
8/8 expectativas humanas cumplidas
```

---

## Dataset U3

El clasificador U3 se entrenó con un conjunto preparado de imágenes GOOD/BAD.

| Split                | GOOD | BAD | Total |
| -------------------- | ---: | --: | ----: |
| Train                |   62 | 153 |   215 |
| Val                  |   13 |  32 |    45 |
| Test                 |   15 |  34 |    49 |
| Holdout supermercado |   22 |   0 |    22 |

---

## Resultados del clasificador U3

| Métrica              |      Valor |
| -------------------- | ---------: |
| Exactitud en test    |     91.84% |
| Holdout supermercado | 22/22 PASA |
| FRR en test          |       0.0% |
| Recall BAD en test   |     97.06% |

---

## Corrección de etiquetas BAD→PASA

Durante la validación se detectaron 6 casos inicialmente etiquetados como BAD que visualmente eran peras comercialmente aceptables.

Se corrigieron como GOOD.

Motivo:

* eran peras con russeting,
* lenticelas,
* textura natural,
* coloración normal,
* no defectos reales graves.

Después de corregir ese ruido de etiqueta, el sistema quedó con:

```text
false_accept_rate = 0.0%
false_reject_rate = 0.0%
```

---

## Validación final corregida

Dataset corregido:

| Categoría      | Resultado |
| -------------- | --------: |
| GOOD → PASA    |        51 |
| GOOD → REVISAR |         4 |
| GOOD → RECHAZA |         0 |
| BAD → PASA     |         0 |
| BAD → REVISAR  |        85 |
| BAD → RECHAZA  |       129 |

---

## Métricas finales de negocio

| Métrica               | Valor |
| --------------------- | ----: |
| False Reject Rate     |  0.0% |
| False Accept Rate     |  0.0% |
| Manual Review Rate    | 33.1% |
| Reject Rate           | 48.0% |
| Automatic Accept Rate | 19.0% |

---

## Interpretación de resultados

El sistema final:

* no rechaza peras buenas en la validación corregida,
* no acepta peras malas en la validación corregida,
* rechaza automáticamente 129 peras malas,
* envía 85 peras malas a revisión,
* envía 4 peras buenas a revisión,
* mantiene una política conservadora.

La revisión manual queda como mecanismo de seguridad.

---

## Validación con peras de supermercado

Se validó también con un conjunto de 86 capturas de peras de supermercado.

Resultado:

```text
86/86 -> PASA
0 -> REVISAR
0 -> RECHAZA
```

Interpretación:

El sistema no rechazó peras comercialmente válidas del conjunto propio de supermercado.

---

## Validación funcional V6 con cámara real

Se probó manualmente la app V6 en cámara real.

Casos probados:

| Caso                                           | Resultado esperado      | Resultado observado |
| ---------------------------------------------- | ----------------------- | ------------------- |
| Fondo blanco sin pera                          | SIN PERA                | Correcto            |
| Pera buena                                     | PASA                    | Correcto            |
| Pera con marca dudosa                          | REVISAR                 | Correcto            |
| Pera colocada en otra cara con zona sospechosa | REVISAR                 | Correcto            |
| Persona en cámara                              | SIN PERA / MALA CAPTURA | Correcto            |
| Mano o brazo en cámara                         | SIN PERA / MALA CAPTURA | Correcto            |
| Fondo blanco bien iluminado                    | SIN PERA                | Correcto            |
| Mala iluminación                               | REVISAR / MALA CAPTURA  | Esperado            |

---

# Aplicación V6 de cámara en tiempo real

## Script principal

```text
scripts/pearvision_qc_realtime_camera_pro_v6.py
```

---

## Comando principal

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_realtime_camera_pro_v6.py --camera 0 --infer-every 5
```

---

## Comando alternativo si la cámara 0 falla

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_realtime_camera_pro_v6.py --camera 1 --infer-every 5
```

---

## Parámetros principales

| Parámetro       | Descripción                      | Valor habitual |
| --------------- | -------------------------------- | -------------- |
| `--camera`      | Índice de cámara OpenCV          | 0              |
| `--infer-every` | Ejecuta inferencia cada N frames | 5              |
| `--width`       | Ancho de captura                 | 1280           |
| `--height`      | Alto de captura                  | 720            |
| `--smoothing`   | Ventana de suavizado             | 7              |

---

## Interfaz V6

La interfaz muestra:

* título del sistema,
* estado de cámara,
* FPS,
* número de frame,
* versión del modelo,
* frame principal,
* contorno de segmentación,
* bounding box,
* miniatura original,
* miniatura de máscara,
* ROI procesado,
* entrada para U3,
* panel de datos técnicos,
* barra de probabilidad,
* resultado final grande.

---

## Panel técnico

El panel técnico incluye:

* `capture_status`,
* `mask_valid`,
* `bg_calibrated`,
* `strategy`,
* `instant_decision`,
* `stable_decision`,
* `smoothing`,
* `u3_pred`,
* `p_good`,
* `p_bad`,
* `thr_live_good`,
* `thr_bad_reject`,
* `pear_area_ratio`,
* `bbox_w_ratio`,
* `bbox_h_ratio`,
* `rectangularity`,
* `solidity`,
* `border_cut`,
* `bbox`,
* `preproc_ms`,
* `infer_ms`,
* `total_ms`,
* `saved_count`.

---

## Controles de teclado

| Tecla | Acción                       |
| ----- | ---------------------------- |
| `Q`   | Cerrar aplicación            |
| `ESC` | Cerrar aplicación            |
| `S`   | Guardar evidencia            |
| `B`   | Calibrar fondo               |
| `C`   | Limpiar fondo calibrado      |
| `P`   | Pausar o reanudar            |
| `R`   | Resetear suavizado           |
| `H`   | Mostrar u ocultar ayuda      |
| `M`   | Mostrar u ocultar miniaturas |

---

## Uso recomendado de la app

1. Abrir la aplicación.
2. Colocar fondo blanco o claro.
3. Comprobar que sin pera aparece `SIN PERA`.
4. Colocar una pera completa dentro del encuadre.
5. Revisar que la máscara rodea solo la pera.
6. Confirmar que la decisión aparece en grande.
7. Pulsar `S` para guardar evidencias.
8. Repetir con otra pera o con otra orientación.
9. Cerrar con `Q` o `ESC`.

---

## Comportamiento esperado

| Situación                   | Salida esperada         |
| --------------------------- | ----------------------- |
| Fondo vacío                 | SIN PERA                |
| Fondo blanco bien iluminado | SIN PERA                |
| Persona en cámara           | SIN PERA o MALA CAPTURA |
| Mano en cámara              | SIN PERA o MALA CAPTURA |
| Pera sana centrada          | PASA                    |
| Pera con zona dudosa        | REVISAR                 |
| Pera muy defectuosa         | RECHAZA o REVISAR       |
| Pera cortada por el borde   | REVISAR o MALA CAPTURA  |
| Iluminación mala            | REVISAR o MALA CAPTURA  |

---

# API e interfaz web local

## Script principal

```text
scripts/pearvision_qc_web_local_v1.py
```

---

## Comando de ejecución

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_web_local_v1.py --camera 0 --host 0.0.0.0 --port 8000 --infer-every 5
```

---

## Acceso local

Desde el propio ordenador:

```text
http://localhost:8000
```

---

## Acceso desde móvil en la misma red

Desde un móvil conectado al mismo Wi-Fi:

```text
http://IP_DEL_PC:8000
```

Ejemplo:

```text
http://192.168.1.128:8000
```

---

## Cómo obtener la IP local en Windows

En CMD:

```powershell
ipconfig
```

Buscar:

```text
Adaptador de LAN inalámbrica Wi-Fi
Dirección IPv4
```

Ejemplo:

```text
192.168.1.128
```

---

## Notas de red

La API funciona en local.

No requiere:

* pagar servidores,
* subir imágenes a internet,
* usar nube,
* abrir puertos públicos,
* contratar hosting.

Solo hace falta que:

* el PC y el móvil estén en la misma red,
* el firewall permita la conexión,
* el servidor esté ejecutándose.

---

## Evidencias de la API

La API local guarda evidencias en:

```text
outputs/live_camera_qc_web_v1/
```

---

# Análisis por lotes

Además de cámara en vivo, el sistema permite analizar carpetas de imágenes.

---

## Modo carpeta con la app V6

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_realtime_camera_pro_v6.py --image-folder data\samples
```

---

## Salidas generadas

```text
outputs/live_camera_qc_pro_v6/folder_test/
```

Archivos típicos:

| Archivo                           | Contenido                       |
| --------------------------------- | ------------------------------- |
| `predictions.csv`                 | Decisiones por imagen           |
| `contact_sheet_all.jpg`           | Mosaico de todos los resultados |
| `contact_sheet_review_reject.jpg` | Mosaico de casos problemáticos  |
| `summary.txt`                     | Resumen de conteos              |

---

## Script general de análisis

```text
scripts/analyze_quality.py
```

---

## Analizar una imagen

```powershell
.\.venv\Scripts\python.exe scripts\analyze_quality.py --image data\samples\pear_01.jpg --use-detector --use-quality-u3 --show
```

---

## Analizar una carpeta

```powershell
.\.venv\Scripts\python.exe scripts\analyze_quality.py --source data\samples --use-detector --use-quality-u3 --save
```

---

## Analizar con más capas

```powershell
.\.venv\Scripts\python.exe scripts\analyze_quality.py --source data\samples --use-detector --use-quality-u3 --use-defect-model --save
```

---

## Argumentos útiles

| Argumento            | Uso                                          |
| -------------------- | -------------------------------------------- |
| `--image PATH`       | Analiza una imagen                           |
| `--source DIR`       | Analiza una carpeta                          |
| `--use-detector`     | Usa detector de pera                         |
| `--use-quality-u3`   | Usa clasificador U3                          |
| `--use-defect-model` | Usa detector de defectos como señal auxiliar |
| `--use-quality-cls`  | Usa clasificador GOOD/BAD auxiliar           |
| `--save`             | Guarda resultados                            |
| `--show`             | Muestra ventana OpenCV                       |
| `--debug`            | Guarda máscaras intermedias                  |
| `--rules PATH`       | Usa archivo de reglas personalizado          |
| `--max-size INT`     | Redimensiona imágenes grandes                |

---

# Instalación

## Requisitos recomendados

| Elemento          | Recomendación           |
| ----------------- | ----------------------- |
| Sistema operativo | Windows 10/11           |
| Python            | 3.13 o compatible       |
| Entorno           | `.venv`                 |
| Gestor            | `uv`                    |
| Cámara            | Webcam integrada o USB  |
| GPU               | Opcional                |
| RAM               | 8 GB mínimo recomendado |

---

## Instalación con uv

Desde la raíz del proyecto:

```powershell
uv sync
```

---

## Verificación de entorno

```powershell
.\.venv\Scripts\python.exe -c "import cv2, torch, ultralytics, fastapi; print('Environment OK')"
```

---

## Comprobar versión de Python

```powershell
.\.venv\Scripts\python.exe --version
```

---

## Comprobar OpenCV

```powershell
.\.venv\Scripts\python.exe -c "import cv2; print(cv2.__version__)"
```

---

## Comprobar PyTorch

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

---

## Comprobar Ultralytics

```powershell
.\.venv\Scripts\python.exe -c "import ultralytics; print('ultralytics OK')"
```

---

# Dependencias

## Dependencias principales

| Dependencia     | Uso                                        |
| --------------- | ------------------------------------------ |
| `opencv-python` | Captura de cámara, segmentación e interfaz |
| `numpy`         | Operaciones matriciales                    |
| `torch`         | Inferencia del clasificador U3             |
| `torchvision`   | Arquitectura MobileNetV3-small             |
| `ultralytics`   | Modelos YOLO                               |
| `fastapi`       | API/interfaz local                         |
| `uvicorn`       | Servidor ASGI                              |
| `pillow`        | Carga y transformación de imágenes         |
| `pyyaml`        | Configuración YAML                         |
| `scikit-image`  | Procesado morfológico auxiliar             |
| `scipy`         | Cálculos auxiliares                        |
| `matplotlib`    | Gráficas y validación                      |

---

## Archivos de dependencias

El proyecto incluye:

```text
pyproject.toml
uv.lock
requirements.txt
```

`uv.lock` permite reproducir el entorno con mayor estabilidad.

---

# Estructura del repositorio

```text
Pearvision-QC/
├── configs/
│   ├── classes.yaml
│   ├── custom_pear_defects.yaml
│   ├── quality_rules.yaml
│   ├── thresholds.yaml
│   └── yolo_pearvision.yaml
│
├── reports/
│   ├── build_fruits360_quality_v2_report.md
│   ├── build_quality_roi_masked_clean_u3_dataset_report.md
│   ├── compare_v2_vs_u3_roi_masked_clean.md
│   ├── consolidate_fruits360_quality_v1_report.md
│   ├── eclpod_inspection_report.md
│   ├── final_quality_pipeline_status_v1.md
│   ├── final_u3_bad_regression_corrected_labels_report.md
│   ├── final_u3_bad_regression_dataset_audit.md
│   ├── final_u3_bad_regression_report.md
│   ├── pearvision_qc_final_pipeline_summary_v1.md
│   ├── pearvision_qc_final_validation_checklist_v1.md
│   ├── pearvision_qc_how_to_use_current_pipeline_v1.md
│   ├── pearvision_qc_realtime_camera_pro_v6_functional_validation_report.md
│   └── other technical reports
│
├── scripts/
│   ├── analyze_quality.py
│   ├── pearvision_qc_realtime_camera_pro_v6.py
│   ├── pearvision_qc_web_local_v1.py
│   ├── train_quality_roi_masked_clean_u3.py
│   ├── evaluate_final_u3_pipeline_bad_regression_v1.py
│   ├── build_quality_roi_masked_clean_u3_dataset.py
│   ├── curate_fruits360_quality_dataset.py
│   ├── inspect_external_dataset.py
│   ├── train_yolo.py
│   ├── validate_expectations.py
│   └── other training, validation and utility scripts
│
├── scripts_clase/
│   └── class practice scripts and image-processing exercises
│
├── src/
│   ├── __init__.py
│   ├── segmentation.py
│   └── quality_analysis.py
│
├── tests/
│   └── automated or manual test scripts
│
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

# Carpetas locales no subidas a Git

Algunas carpetas pueden existir localmente pero no estar subidas a GitHub porque son pesadas.

Ejemplos:

```text
data/
outputs/
runs/
.venv/
.venv_yolo/
```

Esto es normal en proyectos de visión artificial.

---

## Motivo para no subir carpetas pesadas

No conviene subir directamente a GitHub:

* entornos virtuales,
* datasets completos,
* miles de imágenes,
* outputs generados,
* checkpoints pesados,
* carpetas de entrenamiento,
* cachés,
* resultados temporales.

Para eso se puede usar:

* Google Drive,
* GitHub Releases,
* Git LFS,
* OneDrive,
* entrega ZIP aparte,
* o documentación con capturas y reportes.

---

# Modelos y artefactos necesarios

## Modelo U3

Para ejecutar la app final, debe existir localmente:

```text
outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt
```

---

## Umbrales seleccionados

También puede existir:

```text
outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json
```

---

## Detector YOLO opcional

Si se usan partes del pipeline con detector:

```text
runs/detect/runs/pear_detector/eclpod_v1/weights/best.pt
```

---

## Importante para evaluación

Si el repositorio de GitHub no incluye los pesos `.pt`, el profesor podrá revisar el código y la documentación, pero no podrá ejecutar exactamente el mismo modelo final salvo que reciba también los pesos.

Para entrega completa se recomienda adjuntar:

```text
best_model.pt
selected_thresholds.json
evidencias de cámara
reportes finales
```

mediante ZIP, Drive o Release.

---

# Evidencias generadas

## Evidencias de cámara V6

Ruta:

```text
outputs/live_camera_qc_pro_v6/
```

Subcarpetas habituales:

```text
frames_original/
frames_overlay/
masks/
roi_processed/
snapshots/
metadata/
```

---

## Contenido de cada evidencia

| Carpeta            | Contenido                           |
| ------------------ | ----------------------------------- |
| `frames_original/` | Frame original capturado por cámara |
| `frames_overlay/`  | Imagen con máscara y decisión       |
| `masks/`           | Máscara binaria de la pera          |
| `roi_processed/`   | Recorte procesado                   |
| `snapshots/`       | Captura completa de la interfaz     |
| `metadata/`        | JSON con datos técnicos             |

---

## CSV de predicciones en vivo

La app puede generar:

```text
live_predictions.csv
```

Este archivo permite revisar:

* nombre de imagen,
* decisión,
* `p_good`,
* `p_bad`,
* estado de captura,
* validez de máscara,
* observación visual,
* conclusión.

---

# Validación funcional

## Casos mínimos para demostrar el proyecto

Para evaluar el proyecto en directo, se recomienda probar:

1. Fondo blanco sin pera.
2. Pera sana.
3. Pera con pequeña marca.
4. Pera claramente mala o imagen de pera mala.
5. Persona delante de la cámara.
6. Mano u objeto no pera.
7. Guardado de evidencia con `S`.

---

## Resultado esperado

| Prueba              | Resultado esperado      |
| ------------------- | ----------------------- |
| Fondo blanco        | SIN PERA                |
| Pera sana           | PASA                    |
| Pera con marca leve | PASA o REVISAR          |
| Pera dudosa         | REVISAR                 |
| Pera muy mala       | RECHAZA o REVISAR       |
| Persona             | SIN PERA o MALA CAPTURA |
| Mano                | SIN PERA o MALA CAPTURA |
| Pera cortada        | REVISAR o MALA CAPTURA  |

---

## Validación real realizada

Durante las pruebas finales:

* el fondo blanco fue detectado como `SIN PERA`,
* las peras buenas fueron detectadas como `PASA`,
* una pera con pequeño agujerito o zona sospechosa pasó a `REVISAR`,
* persona/mano no fue aceptada como pera,
* la iluminación se confirmó como factor importante,
* la máscara mejoró notablemente en V5/V6,
* la app V6 quedó como versión final funcional.

---

# Configuración de reglas

## Archivo principal

```text
configs/quality_rules.yaml
```

---

## Uso de reglas

El archivo permite modificar la lógica rule-based sin reentrenar el modelo.

Contiene umbrales como:

* porcentaje máximo de defecto,
* porcentaje de podredumbre,
* región máxima sospechosa,
* condiciones combinadas de rechazo,
* reglas de revisión.

---

## Umbrales de cámara V6

En la aplicación V6 se utilizan umbrales específicos para cámara real:

```text
LIVE_GOOD_ACCEPT_THRESHOLD = 0.60
BAD_REJECT_THRESHOLD = 0.995
```

---

## Por qué `p_bad` requiere 0.995

El rechazo automático se deja muy estricto para evitar falsos rechazos.

La política es:

```text
mejor revisar una pera dudosa que rechazar mal una pera válida
```

---

# Diseño de la interfaz

## Objetivo visual

La interfaz se diseñó para parecer una herramienta profesional de inspección industrial.

Incluye:

* tema oscuro,
* panel técnico,
* decisión grande,
* colores por estado,
* overlays,
* miniaturas,
* métricas,
* información de cámara,
* latencia.

---

## Colores de decisión

| Decisión     | Color                     |
| ------------ | ------------------------- |
| PASA         | Verde                     |
| REVISAR      | Naranja                   |
| RECHAZA      | Rojo                      |
| SIN PERA     | Azul                      |
| MALA CAPTURA | Azul/Naranja según estado |

---

## Razón del panel técnico

El panel técnico permite justificar la decisión del sistema.

No solo se muestra el resultado final.

También se muestran:

* probabilidades,
* umbrales,
* estado de máscara,
* área,
* bounding box,
* latencias,
* decisión instantánea,
* decisión suavizada.

Esto facilita la explicación del proyecto ante el profesor.

---

# Limitaciones

## 1. Iluminación

La iluminación afecta mucho al resultado.

Una mala iluminación puede provocar:

* máscara incompleta,
* zonas oscuras falsas,
* confianza menor,
* decisión `REVISAR`,
* decisión `MALA CAPTURA`.

---

## 2. Fondo

El sistema funciona mejor con fondo blanco, gris o neutro.

Fondos con objetos, manos, ropa, cables o sombras pueden afectar a la segmentación.

---

## 3. Solo una pera

El prototipo está diseñado para una pera por imagen.

No está preparado para una cinta industrial con muchas peras simultáneas.

---

## 4. Defectos internos

El sistema no puede detectar daños internos.

Para eso harían falta tecnologías como:

* hiperespectral,
* rayos X,
* OCT,
* NIR,
* sensores específicos,
* análisis destructivo.

---

## 5. Russeting

El russeting natural puede parecer defecto.

El sistema intenta manejarlo mediante:

* U3,
* umbrales conservadores,
* revisión manual,
* política de no rechazo si hay duda.

---

## 6. Dataset limitado

El dataset usado es suficiente para prototipo académico, pero no para producción industrial.

Para producción haría falta:

* más variedades,
* más cámaras,
* más iluminación,
* más lotes,
* más defectos reales,
* validación por expertos,
* test en línea real.

---

## 7. Métrica comercial simplificada

Las decisiones `PASA`, `REVISAR` y `RECHAZA` son una simplificación académica.

Una central hortofrutícola real podría tener:

* Categoría Extra,
* Categoría I,
* Categoría II,
* industria,
* descarte,
* transformación.

---

# Uso recomendado para evaluación

## Evaluación rápida en GitHub

El profesor puede revisar:

1. `README.md`
2. `scripts/pearvision_qc_realtime_camera_pro_v6.py`
3. `scripts/pearvision_qc_web_local_v1.py`
4. `scripts/analyze_quality.py`
5. `src/segmentation.py`
6. `src/quality_analysis.py`
7. `configs/quality_rules.yaml`
8. `reports/`

---

## Evaluación funcional local

Para ejecutar la demo:

```powershell
uv sync
```

Después:

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_realtime_camera_pro_v6.py --camera 0 --infer-every 5
```

---

## Evaluación de API local

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_web_local_v1.py --camera 0 --host 0.0.0.0 --port 8000 --infer-every 5
```

Después abrir:

```text
http://localhost:8000
```

o desde móvil:

```text
http://IP_DEL_PC:8000
```

---

## Evidencia que demuestra el esfuerzo

El proyecto demuestra trabajo en:

* búsqueda de imágenes,
* preparación de dataset,
* segmentación,
* entrenamiento,
* validación,
* depuración de falsos positivos,
* corrección de etiquetas,
* creación de app,
* creación de API,
* pruebas reales con cámara,
* documentación técnica,
* generación de evidencias.

---

# Archivos importantes del proyecto

## Código principal

| Archivo                                           | Función                       |
| ------------------------------------------------- | ----------------------------- |
| `scripts/pearvision_qc_realtime_camera_pro_v6.py` | Aplicación final de cámara    |
| `scripts/pearvision_qc_web_local_v1.py`           | API/interfaz web local        |
| `scripts/analyze_quality.py`                      | Análisis por imagen o carpeta |
| `src/segmentation.py`                             | Segmentación                  |
| `src/quality_analysis.py`                         | Lógica de calidad             |
| `configs/quality_rules.yaml`                      | Reglas de decisión            |

---

## Scripts de entrenamiento y evaluación

| Archivo                                                   | Función                     |
| --------------------------------------------------------- | --------------------------- |
| `scripts/train_quality_roi_masked_clean_u3.py`            | Entrena U3                  |
| `scripts/evaluate_final_u3_pipeline_bad_regression_v1.py` | Evalúa regresión BAD        |
| `scripts/build_quality_roi_masked_clean_u3_dataset.py`    | Construye dataset U3        |
| `scripts/validate_expectations.py`                        | Valida expectativas humanas |
| `scripts/train_yolo.py`                                   | Entrenamiento YOLO          |
| `scripts/inspect_external_dataset.py`                     | Inspección de datasets      |

---

## Reportes

| Reporte                                                                | Descripción                 |
| ---------------------------------------------------------------------- | --------------------------- |
| `final_quality_pipeline_status_v1.md`                                  | Estado del pipeline inicial |
| `final_u3_bad_regression_report.md`                                    | Regresión BAD               |
| `final_u3_bad_regression_corrected_labels_report.md`                   | Corrección de etiquetas     |
| `pearvision_qc_final_pipeline_summary_v1.md`                           | Resumen final               |
| `pearvision_qc_final_validation_checklist_v1.md`                       | Checklist de validación     |
| `pearvision_qc_how_to_use_current_pipeline_v1.md`                      | Guía de uso                 |
| `pearvision_qc_realtime_camera_pro_v6_functional_validation_report.md` | Validación funcional V6     |

---

# Reproducibilidad

## Clonar repositorio

```powershell
git clone https://github.com/Jose-https1/Pearvision-QC.git
cd Pearvision-QC
```

---

## Instalar entorno

```powershell
uv sync
```

---

## Ejecutar cámara

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_realtime_camera_pro_v6.py --camera 0 --infer-every 5
```

---

## Ejecutar API

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_web_local_v1.py --camera 0 --host 0.0.0.0 --port 8000 --infer-every 5
```

---

## Ejecutar análisis por carpeta

```powershell
.\.venv\Scripts\python.exe scripts\analyze_quality.py --source data\samples --use-detector --use-quality-u3 --save
```

---

# Notas sobre GitHub

El repositorio se ha preparado para subir principalmente:

* código,
* configuración,
* scripts,
* reportes,
* README,
* dependencias.

No se recomienda subir directamente:

* `.venv`,
* `.venv_yolo`,
* `data` completa,
* `outputs` completos,
* `runs` completos,
* vídeos,
* caché,
* modelos pesados sin Git LFS.

---

## Recomendación para entrega completa

Para que el profesor pueda ejecutar exactamente la demo final, entregar también:

```text
outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt
outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json
outputs/live_camera_qc_pro_v6/
```

Si no se quieren subir a GitHub, se pueden entregar por:

* ZIP,
* Google Drive,
* OneDrive,
* GitHub Release,
* Git LFS.

---

# Seguridad y privacidad

La aplicación funciona localmente.

No envía imágenes a servidores externos.

La API local solo expone el sistema dentro de la red si se ejecuta con:

```text
--host 0.0.0.0
```

Para uso privado en el mismo PC, usar solo:

```text
localhost
```

---

# Conclusión

PearVision QC demuestra un pipeline completo de visión artificial aplicado a control de calidad superficial de peras.

El proyecto cubre:

1. adquisición de imágenes,
2. preparación de dataset,
3. segmentación,
4. clasificación neuronal,
5. reglas de decisión,
6. validación humana,
7. corrección de errores,
8. aplicación en tiempo real,
9. interfaz profesional,
10. API local,
11. guardado de evidencias,
12. documentación técnica.

El resultado final es un prototipo académico funcional que permite analizar peras en tiempo real con la cámara del portátil y obtener una decisión clara:

```text
PASA / REVISAR / RECHAZA
```

La versión final V6 es la versión recomendada para demostración.

---

# Final demo command

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_realtime_camera_pro_v6.py --camera 0 --infer-every 5
```

# Final local web/API command

```powershell
.\.venv\Scripts\python.exe scripts\pearvision_qc_web_local_v1.py --camera 0 --host 0.0.0.0 --port 8000 --infer-every 5
```

# Expected final behaviour

```text
Fondo blanco      -> SIN PERA
Pera sana         -> PASA
Pera dudosa       -> REVISAR
Pera muy dañada   -> RECHAZA o REVISAR
Persona/mano      -> SIN PERA o MALA CAPTURA
```

# Project status

```text
Final academic prototype: COMPLETED
Recommended version: V6
Execution mode: Local
Cloud dependency: No
Real-time camera: Yes
Local API/web: Yes
Evidence saving: Yes
```

````

Después de pegarlo en `README.md`, ejecuta:

```powershell
git add README.md
git commit -m "Expand README with full final project documentation"
git push
````
