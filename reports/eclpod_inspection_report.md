# ECLPOD Dataset — Informe de Inspección

**Fecha:** 2026-05-17  
**Proyecto:** PearVision QC  
**Dataset:** ECLPOD (Extremely Compressed Lightweight Model for Pear Object Detection)

---

## 1. Estructura encontrada

```
data_external/ECLPOD/
├── README.md
├── .gitattributes
├── images/          ← 4 516 archivos .JPG
└── labels/          ← 4 760 archivos .txt (YOLO)
```

- No hay subcarpetas `train/val/test` — dataset plano sin splits.
- No hay archivo `.yaml`, `.json` ni configuración de clases incluida.
- Nombres de archivo: `pear<id>.JPG` / `pear<id>.txt` con IDs en rango ~13302–24xxx.
- 244 archivos de label sin imagen correspondiente (rango pear24xxx no descargado).
- 0 imágenes sin label — cobertura completa en el subconjunto descargado.

---

## 2. Número de imágenes y labels

| Elemento | Cantidad |
|---|---|
| Imágenes (.JPG) | 4 516 |
| Archivos de label (.txt) | 4 760 |
| Labels huérfanos (sin imagen) | 244 |
| Imágenes sin label | 0 |
| **Total anotaciones (instancias)** | **21 770** |

---

## 3. Formato de anotación detectado

**YOLO Detection** (bounding box), NO segmentación.

Cada línea de un `.txt`:
```
<class_id> <x_center> <y_center> <width> <height>
```
Todos los valores normalizados [0, 1] relativo al tamaño de imagen.

Ejemplo real (`pear13302.txt`):
```
0 0.49867 0.48000 0.16822 0.24400    ← pear body
0 0.66606 0.49950 0.17653 0.25300    ← pear body
2 0.50831 0.42550 0.06316 0.08300    ← calyx
1 0.69249 0.41925 0.03657 0.04550    ← stem
```

---

## 4. Clases detectadas

| Clase | Instancias | Área bbox media (norm.) | Interpretación |
|---|---|---|---|
| **0** | 10 519 | 0.077 | **Pear body** — cuerpo completo de la pera (bbox grande) |
| **1** | 7 516 | 0.003 | **Stem** — pedúnculo/rabillo (bbox muy pequeño) |
| **2** | 3 409 | 0.007 | **Calyx** — cáliz/ojo (bbox pequeño) |
| **3** | 326 | 0.006 | **Desconocido** — parte minoritaria, bbox similar al cáliz |

**Fuente:** El README de ECLPOD menciona explícitamente:  
> *"overcome the limitations in detecting the features of the **pear body**, **stem**, and **calyx**"*

Las clases 0, 1 y 2 se corresponden directamente con esas tres estructuras.  
La clase 3 (326 instancias, ~1.5 % del total) podría ser una marca especial, defecto visible o pedúnculo lateral — no documentada en el README publicado.

**El dataset NO anota defectos superficiales** (golpes, podredumbre, marcas de ramita).

---

## 5. ¿Para qué sirve ECLPOD en PearVision QC?

| Uso | ¿Viable? | Notas |
|---|---|---|
| Entrenar detector de pera (clase 0) | **Sí, el mejor uso** | 10 519 bbox de cuerpo de pera. Resuelve el problema de GrabCut/color-seg. |
| Entrenar segmentador de pera | No directamente | Solo hay bbox, no máscaras de polígono. Posible con SAM+bbox como prompt. |
| Recortar ROI antes del análisis | **Sí** | Usar predicción de clase 0 como ROI → recortar → pasar al pipeline de defectos. |
| Detectar defectos superficiales | **No** | El dataset no tiene anotaciones de defectos. Ese es el dominio de PearVision QC propio. |

**Conclusión:** ECLPOD sirve para localizar la pera (clase 0) con un detector YOLO ligero. Proporciona un ROI limpio que evita capturar mano, fondo, mantel u hojas — el problema actual de la segmentación clásica.

---

## 6. Recomendación del siguiente paso exacto

**Entrenar YOLOv8n usando solo clase 0 de ECLPOD como detector de pera.**

Pasos concretos:

1. **Crear el split** — dividir los 4 516 pares imagen/label en train (80 %) / val (20 %).
2. **Filtrar etiquetas** — opcionalmente, conservar solo líneas de clase 0 si se quiere un detector mono-clase (pear body).
3. **Crear `configs/eclpod_pear_detector.yaml`** con las rutas y la definición de clase.
4. **Fine-tuning** desde `yolov8n.pt` (modelo más ligero, adecuado para prototipo académico):
   ```
   yolo detect train model=yolov8n.pt data=configs/eclpod_pear_detector.yaml epochs=30 imgsz=640
   ```
5. **Integrar** el detector en el pipeline: sustituir `GrabCut` / segmentación HSV por el bbox de clase 0 como máscara de ROI.

Este paso es reversible y no modifica el pipeline actual hasta que el detector sea validado.
