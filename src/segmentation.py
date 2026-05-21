import cv2
import numpy as np
from pathlib import Path


def imread_unicode(path):
    """cv2.imread compatible con rutas Unicode en Windows."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar: {path}")
    return image


def load_image(path):
    """Load BGR image; raise if path is invalid or file unreadable."""
    return imread_unicode(path)


def _isolate_pear_body(mask, config):
    """Rompe puentes finos entre la pera y el fondo/mantel mediante erosion + componente mayor + redilatacion.

    Evita que zonas de fondo conectadas accidentalmente amplien la mascara de la pera.
    Si la erosion destruye la mascara completamente, devuelve la mascara original.
    """
    erode_px = int(config.get("bridge_erode_px", 10))
    iterations = int(config.get("bridge_erode_iterations", 2))

    k_size = max(3, erode_px * 2 + 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    eroded = cv2.erode(mask, k, iterations=iterations)

    contour = get_largest_contour(eroded)
    if contour is None:
        return mask  # erosion destruyo todo; devolver original

    main = np.zeros_like(mask)
    cv2.drawContours(main, [contour], -1, 255, thickness=cv2.FILLED)

    dilated = cv2.dilate(main, k, iterations=iterations)
    return cv2.bitwise_and(dilated, mask)


def segment_pear(image, config):
    """Dispatcher: selecciona HSV o GrabCut según config['method'] y aplica postproceso común."""
    method = str(config.get("method", "hsv")).lower()

    if method == "grabcut":
        mask = segment_pear_grabcut(image, config)
    else:
        mask = _segment_pear_hsv(image, config)

    # Postproceso común para ambos métodos
    if config.get("remove_bottom_artifacts", False):
        band_ratio = float(config.get("bottom_band_ratio", 0.18))
        row_ratio = float(config.get("min_row_width_ratio", 0.25))
        mask = remove_bottom_artifacts(mask, band_ratio, row_ratio)

    # Aislamiento del cuerpo principal: rompe puentes finos con fondo/mantel
    if config.get("isolate_pear_body", True):
        mask = _isolate_pear_body(mask, config)

    return mask


def segment_pear_grabcut(image, config):
    """Segmentación con GrabCut a partir de un rectángulo con margen fijo.

    GrabCut clasifica cada píxel como fondo / probable fondo / probable frente / frente.
    Se retiene el frente definitivo + probable frente y se rellena el contorno mayor.
    """
    margin_ratio = float(config.get("grabcut_margin_ratio", 0.08))
    iterations = int(config.get("grabcut_iterations", 5))
    min_area_ratio = float(config.get("min_area_ratio", 0.05))
    max_area_ratio = float(config.get("max_mask_area_ratio", 0.90))

    h, w = image.shape[:2]
    mx = max(1, int(w * margin_ratio))
    my = max(1, int(h * margin_ratio))
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc_mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)

    # Foreground definitivo + probable foreground → máscara binaria
    fg = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Filtro de área y relleno del contorno principal
    contour = get_largest_contour(fg)
    if contour is None:
        return np.zeros(fg.shape, dtype=np.uint8)

    image_area = h * w
    contour_area = cv2.contourArea(contour)
    if contour_area < min_area_ratio * image_area or contour_area > max_area_ratio * image_area:
        return np.zeros(fg.shape, dtype=np.uint8)

    filled = np.zeros(fg.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def _segment_pear_hsv(image, config):
    """Segmentación por threshold HSV con eliminación de sombras y despegado morfológico.

    Pasos:
      1. Blur + threshold HSV de color
      2. Eliminación de píxeles con baja saturación/valor (sombra)
      3. Opening/closing morfológico
      4. Filtro por área mínima/máxima
      5. Relleno del contorno principal
      6. Despegado de sombra conectada (opcional)
      7. Re-extracción y relleno del contorno principal
    """
    blur_k = int(config.get("blur_kernel", 5))
    lower = np.array(config.get("hsv_lower", [15, 30, 30]), dtype=np.uint8)
    upper = np.array(config.get("hsv_upper", [90, 255, 255]), dtype=np.uint8)
    morph_k = int(config.get("morph_kernel", 7))
    shadow_sat_min = int(config.get("shadow_saturation_min", 45))
    shadow_val_min = int(config.get("shadow_value_min", 40))
    min_area_ratio = float(config.get("min_area_ratio", 0.05))
    max_area_ratio = float(config.get("max_mask_area_ratio", 0.90))

    blurred = cv2.GaussianBlur(image, (blur_k, blur_k), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 1. Threshold de color
    color_mask = cv2.inRange(hsv, lower, upper)

    # 2. Eliminar píxeles de sombra: baja saturación o muy oscuros
    low_sat = hsv[:, :, 1] < shadow_sat_min
    low_val = hsv[:, :, 2] < shadow_val_min
    shadow_px = ((low_sat | low_val).astype(np.uint8)) * 255
    mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(shadow_px))

    # 3. Opening/closing para eliminar ruido y rellenar huecos pequeños
    mask = clean_mask(mask, morph_k)

    # 4. Filtrar por área: descartar si es ruido o cubre toda la imagen
    contour = get_largest_contour(mask)
    if contour is None:
        return np.zeros(mask.shape, dtype=np.uint8)

    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(contour)
    if contour_area < min_area_ratio * image_area or contour_area > max_area_ratio * image_area:
        return np.zeros(mask.shape, dtype=np.uint8)

    # 5. Rellenar el contorno principal
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

    # 6. Despegar sombra conectada mediante erosión/dilatación
    if config.get("use_shadow_detach", False):
        kernel_size = int(config.get("detach_shadow_kernel", 9))
        iterations = int(config.get("detach_shadow_iterations", 2))
        filled = detach_connected_shadow(filled, kernel_size, iterations)

    # 7. Volver a quedarse con el contorno principal tras el despegado
    contour = get_largest_contour(filled)
    if contour is None:
        return np.zeros(filled.shape, dtype=np.uint8)
    final = np.zeros(filled.shape, dtype=np.uint8)
    cv2.drawContours(final, [contour], -1, 255, thickness=cv2.FILLED)
    return final


def clean_mask(mask, morph_kernel):
    """Opening elimina ruido; closing rellena huecos internos."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def get_largest_contour(mask):
    """Devuelve el contorno de mayor área, o None si la máscara está vacía."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def detach_connected_shadow(mask, kernel_size, iterations):
    """Erosiona para romper conexiones finas entre pera y sombra, luego redilata.

    Pasos:
      1. Erosión fuerte: separa componentes conectados por zonas estrechas
      2. Conservar solo el componente mayor (cuerpo de la pera)
      3. Re-dilatación: recupera el tamaño aproximado original
      4. AND con la máscara original: no se inventan píxeles nuevos
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    eroded = cv2.erode(mask, k, iterations=iterations)

    contour = get_largest_contour(eroded)
    if contour is None:
        return mask  # Erosión destruyó todo; devolver original

    main_component = np.zeros_like(mask)
    cv2.drawContours(main_component, [contour], -1, 255, thickness=cv2.FILLED)

    dilated = cv2.dilate(main_component, k, iterations=iterations)

    return cv2.bitwise_and(dilated, mask)


def remove_bottom_artifacts(mask, bottom_band_ratio, min_row_width_ratio):
    """Elimina apéndices inferiores estrechos conectados a la pera (sombra, soporte, mesa).

    Analiza la banda inferior del bbox. Si una fila tiene ancho de píxeles
    blancos menor que min_row_width_ratio * ancho_máximo, corta desde ahí hacia abajo.
    """
    contour = get_largest_contour(mask)
    if contour is None:
        return mask

    x, y, w, h = cv2.boundingRect(contour)

    max_width = 0
    for row in range(y, y + h):
        max_width = max(max_width, int(np.count_nonzero(mask[row, x:x + w])))

    if max_width == 0:
        return mask

    band_start = y + h - max(1, int(h * bottom_band_ratio))

    cut_row = None
    for row in range(band_start, y + h):
        row_width = int(np.count_nonzero(mask[row, x:x + w]))
        if row_width < min_row_width_ratio * max_width:
            cut_row = row
            break

    if cut_row is not None:
        cleaned = mask.copy()
        cleaned[cut_row:, :] = 0
        return cleaned

    return mask


def apply_mask(image, mask):
    """Pone a cero todos los píxeles fuera de la máscara binaria."""
    return cv2.bitwise_and(image, image, mask=mask)


def compute_body_mask(pear_mask, rules):
    """Mascara del cuerpo central: erosiona el borde y excluye rabo y base.

    Si el resultado es demasiado pequeno (< min_body_area_pct del area de pera),
    hace fallback a solo la erosion sin cortes, para no quedarse sin area de analisis.
    """
    border_px = int(rules.get("body_border_erode_px", 10))
    top_ratio = float(rules.get("body_top_exclude_ratio", 0.08))
    bot_ratio = float(rules.get("body_bottom_exclude_ratio", 0.05))
    min_body_pct = float(rules.get("min_body_area_pct", 0.30))

    k_size = max(3, border_px * 2 + 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    inner = cv2.erode(pear_mask, k, iterations=1)

    contour = get_largest_contour(inner)
    if contour is None:
        return inner

    _x, y, _w, h = cv2.boundingRect(contour)
    body = inner.copy()
    body[: y + int(h * top_ratio), :] = 0
    body[y + h - max(1, int(h * bot_ratio)) :, :] = 0

    # Fallback si el cuerpo queda demasiado pequeno
    pear_area = int(np.count_nonzero(pear_mask))
    if pear_area > 0 and int(np.count_nonzero(body)) < min_body_pct * pear_area:
        return inner

    return body
