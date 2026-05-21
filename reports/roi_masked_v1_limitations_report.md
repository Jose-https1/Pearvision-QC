# ROI Masked V1 — Limitations Report

**Fecha:** 2026-05-21
**Basado en:** `scripts/prepare_quality_roi_masked_previews.py` + `scripts/fix_roi_masked_contact_sheets.py`

---

## 1. Método usado en V1

### Detección de bbox
- **YOLO (eclpod_v1):** detector YOLOv8 entrenado para detectar peras. Devuelve bbox (x1,y1,x2,y2).
- **Fallback:** si YOLO no detecta ninguna pera, se usa un rectángulo central del 70% de la imagen.

### Segmentación
- **GrabCut** de OpenCV, inicializado con el rect YOLO o el fallback central.
- 5 iteraciones de GrabCut.
- Clasificación final de píxeles: GC_FGD | GC_PR_FGD → foreground.

### Post-procesado V1
1. `cv2.morphologyEx(MORPH_CLOSE, kernel_ellipse_7x7, iterations=3)` — cierra huecos pequeños.
2. `cv2.morphologyEx(MORPH_OPEN,  kernel_ellipse_7x7, iterations=1)` — elimina proyecciones pequeñas.
3. Flood-fill desde (0,0) para rellenar huecos internos.

---

## 2. Por qué V1 incluye sombras y halos

### a) GrabCut no distingue sombra de fruta en bordes ambiguos
GrabCut trabaja con modelos de mezcla gaussiana en el espacio RGB. Las sombras proyectadas
por la pera sobre el fondo (especialmente en fondo blanco) tienen una mezcla de color entre
la pera y el fondo. GrabCut los clasifica como PR_FGD (probable foreground) porque su color
es intermedio entre el fondo blanco conocido y la pera.

### b) La inicialización con bbox sobredimensionado incluye zona de sombra
El margen extra del 12% alrededor del bbox YOLO asegura que no se recorta la pera, pero
también incluye la zona de sombra en la región inicializada como probable foreground.
GrabCut tiende entonces a incluir esa sombra en la máscara final.

### c) No hay eliminación activa de píxeles parecidos al fondo
V1 no compara los píxeles del borde de la máscara con el color del fondo. Si un píxel
está dentro del bbox y tiene color ambiguo (gris claro = sombra sobre fondo blanco),
GrabCut lo incluye como foreground.

### d) Closing agresivo (iterations=3) puede "pegar" sombras al contorno
El cierre morfológico con kernel 7x7 e iteraciones 3 puede unir zonas de sombra cercanas
al borde de la pera con el cuerpo principal, especialmente si la sombra es fina pero
continua.

### e) Fondo negro/azul: contaminación de color
En batch_v2 con fondo azul/negro, GrabCut sin información de color de fondo puede
incluir zonas del fondo azul cerca de la pera en la máscara, especialmente en
transiciones de color gradual (zona de contacto pera-fondo).

---

## 3. Limitaciones del fallback de las imágenes sin YOLO

Las imágenes de batch_v2/v3 que en V1 no fueron procesadas por YOLO ni por el script
de fix usaron el fallback de centro. En las imágenes generadas por `fix_roi_masked_contact_sheets.py`
(con GrabCut sin YOLO), el resultado es aceptable en fondo blanco pero subóptimo en
fondos azul/negro porque el rectángulo central captura fondo cromático.

---

## 4. Qué mejora V2

| Problema | V1 | V2 propuesto |
|---|---|---|
| Sombras en bordes | No tratado | Eliminación por similitud de color con fondo |
| Halos blancos | Closing agresivo los pega | Erosión conservadora + limpieza border |
| Componentes aislados | Apertura suave | Retener solo LCC (largest connected component) |
| Relleno de huecos | Flood-fill desde (0,0) | Fill-holes correcto + apertura suave |
| Fondo azul/negro | No hay robustez | Estimación de color de fondo por esquinas |

---

## 5. Cuántos crops V1 existen

- **47 grupos completos** (original + mask + gray_bg + white_bg).
- Cubren: batch_v1 (20), batch_v2 (22), batch_v3 primeras 5 (790-794).
- Faltan: 17 imágenes de batch_v3 (795-815).

El script V2 reutilizará los `_original.jpg` de V1 para las 47 imágenes ya procesadas
(peras ya centradas a 224×224) y cargará las 17 restantes desde las fuentes originales
usando PIL (para evitar el problema de rutas con caracteres acentuados en cv2.imread).
