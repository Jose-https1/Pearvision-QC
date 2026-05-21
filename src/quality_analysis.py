"""
quality_analysis.py — Motor de calidad rule-based v8 para PearVision QC.

v8: validate_capture reforzado con aspect ratio, metricas geometricas exportadas
    al CSV y early-exit antes de detect_defects si la captura es invalida.
v3: deteccion de sombras por contraste local en espacio LAB.

Solo evalua defectos superficiales visibles.
No mide sabor, Brix, firmeza, madurez interna ni calidad nutricional.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Deteccion de sombras
# ---------------------------------------------------------------------------

def _build_shadow_mask(image, body_mask, rules):
    """
    Identifica zonas de sombra: areas oscuras, amplias y suaves que siguen
    el gradiente de iluminacion — no son defectos superficiales reales.

    Metodo:
      1. Convierte a LAB (L = luminosidad perceptualmente uniforme, 0-255).
      2. Estima iluminacion global con un Gaussian muy grande (blur_big).
      3. global_dark = iluminacion_estimada - L  (cuanto mas oscuro que esperado).
      4. local_contrast = |L - blur_pequeno(L)|  (variacion a escala inmediata).
      5. Sombra: pixel globalmente oscuro Y con bajo contraste local (gradiente suave).
      6. Filtra regiones pequenas para no confundir manchas reales con sombra.
    """
    blur_big_k = int(rules.get("shadow_illumination_blur", 101))
    if blur_big_k % 2 == 0:
        blur_big_k += 1

    blur_local_k = int(rules.get("shadow_local_blur", 21))
    if blur_local_k % 2 == 0:
        blur_local_k += 1

    dark_thr = float(rules.get("shadow_contrast_threshold", 15.0))
    min_area_ratio = float(rules.get("shadow_min_area_ratio", 0.05))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Iluminacion global estimada (gradiente de luz en la escena)
    illumination = cv2.GaussianBlur(L, (blur_big_k, blur_big_k), 0)
    global_dark = np.clip(illumination - L, 0, 255).astype(np.float32)

    # Contraste local inmediato (variacion a escala pequena)
    local_avg = cv2.GaussianBlur(L, (blur_local_k, blur_local_k), 0)
    local_contrast = np.abs(L - local_avg).astype(np.float32)

    # Sombra: globalmente oscuro Y localmente suave (transicion gradual)
    shadow_raw = (
        (global_dark > dark_thr) & (local_contrast < dark_thr * 0.6)
    ).astype(np.uint8) * 255

    shadow_raw = cv2.bitwise_and(shadow_raw, body_mask)

    # Cerrar huecos pequenos dentro de la zona de sombra
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    shadow_raw = cv2.morphologyEx(shadow_raw, cv2.MORPH_CLOSE, k_close)

    # Solo sombras de area significativa (evita confundir manchas con sombra)
    body_area = int(np.count_nonzero(body_mask))
    min_shadow_px = max(100, int(body_area * min_area_ratio))
    shadow_mask = _filter_by_min_area(shadow_raw, min_shadow_px)

    return shadow_mask


# ---------------------------------------------------------------------------
# Deteccion de defectos
# ---------------------------------------------------------------------------

def detect_defects(image, pear_mask, body_mask, rules):
    """
    v3: Detecta defectos reales en el cuerpo de la pera, excluyendo sombras suaves.

    Pipeline:
      1. Construir shadow_mask (zonas oscuras graduales = no defecto).
      2. Detectar candidatos mediante umbrales HSV en body_mask.
      3. Restar shadow_mask de los candidatos (sombra no es defecto).
      4. Limpieza morfologica y filtro de area minima.

    Retorna:
        final_defect_mask (np.uint8): defectos reales sin sombras
        rot_mask          (np.uint8): podredumbre oscura sin sombras
        shadow_mask       (np.uint8): sombras detectadas (para debug y visualizacion)
        defect_mask_raw   (np.uint8): candidatos antes de restar sombra (debug)
    """
    blur_k = int(rules.get("analysis_blur_kernel", 5))
    min_area = int(rules.get("min_region_area_px", 80))

    rot_lower = np.array(rules.get("rot_hsv_lower", [0, 40, 0]), dtype=np.uint8)
    rot_upper = np.array(rules.get("rot_hsv_upper", [25, 255, 75]), dtype=np.uint8)
    brown_lower = np.array(rules.get("brown_hsv_lower", [5, 30, 55]), dtype=np.uint8)
    brown_upper = np.array(rules.get("brown_hsv_upper", [25, 200, 135]), dtype=np.uint8)

    # Fallback: si body_mask esta vacia, usar pear_mask
    analysis_mask = body_mask if (body_mask is not None and body_mask.any()) else pear_mask

    # 1. Sombras (se calculan antes para poder restarlas)
    shadow_mask = _build_shadow_mask(image, analysis_mask, rules)

    # 2. Candidatos HSV
    blurred = cv2.GaussianBlur(image, (blur_k, blur_k), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    rot_raw = cv2.bitwise_and(cv2.inRange(hsv, rot_lower, rot_upper), analysis_mask)
    brown_raw = cv2.bitwise_and(cv2.inRange(hsv, brown_lower, brown_upper), analysis_mask)
    combined_raw = cv2.bitwise_or(rot_raw, brown_raw)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    defect_mask_raw = cv2.morphologyEx(combined_raw, cv2.MORPH_OPEN, k)
    defect_mask_raw = cv2.morphologyEx(defect_mask_raw, cv2.MORPH_CLOSE, k)
    rot_raw_clean = cv2.morphologyEx(rot_raw, cv2.MORPH_OPEN, k)

    # 3. Restar sombras: solo quedan defectos con contraste local real
    not_shadow = cv2.bitwise_not(shadow_mask)
    final_defect_mask = cv2.bitwise_and(defect_mask_raw, not_shadow)
    rot_mask = cv2.bitwise_and(rot_raw_clean, not_shadow)

    # 4. Filtrar ruido por area minima
    final_defect_mask = _filter_by_min_area(final_defect_mask, min_area)
    rot_mask = _filter_by_min_area(rot_mask, min_area)

    return final_defect_mask, rot_mask, shadow_mask, defect_mask_raw


def _filter_by_min_area(mask, min_area):
    """Elimina componentes conexos con area < min_area."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 255
    return out


def _largest_region_pct(defect_mask, ref_area_px):
    """Porcentaje de la region de defecto mas grande respecto al area de referencia."""
    if ref_area_px == 0:
        return 0.0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    if n_labels <= 1:
        return 0.0
    largest_px = int(max(stats[1:, cv2.CC_STAT_AREA]))
    return (largest_px / ref_area_px) * 100.0


# ---------------------------------------------------------------------------
# Mascara de pera guiada por ROI del detector (GrabCut + fallbacks)
# ---------------------------------------------------------------------------

def _keep_largest_blob(mask):
    """Mantiene solo el componente conexo mas grande de una mascara binaria."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def _morpho_clean(mask):
    """Cierra huecos pequenos y elimina ruido con morfologia eliptica."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _mask_geometry_ok(mask, h, w):
    """
    Valida geometria basica de la mascara para una pera.
    Retorna (ok: bool, razon: str).
    """
    area = int(np.count_nonzero(mask))
    image_area = max(1, h * w)
    if area < image_area * 0.12:
        return False, "too_small"
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "no_contours"
    largest = max(contours, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(largest)
    aspect = bh / max(1, bw)
    if aspect < 0.28 or aspect > 4.2:
        return False, f"aspect_{aspect:.2f}"
    bbox_area = max(1, bw * bh)
    fill = area / bbox_area
    if fill < 0.18 or fill > 0.93:
        return False, f"fill_{fill:.2f}"
    return True, ""


def _pear_color_mask(img):
    """Segmentacion por color HSV para peras (amarillo-verde, no blanco/gris de fondo)."""
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([12, 15, 35], dtype=np.uint8)
    upper = np.array([115, 255, 252], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _ellipse_mask(h, w, scale=0.80):
    """Mascara elipse conservadora centrada en la imagen (fallback)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2),
                (int(w * scale / 2), int(h * scale / 2)),
                0, 0, 360, 255, -1)
    return mask


def build_detector_roi_pear_mask(crop_img, rules=None):
    """
    Genera una mascara robusta de pera desde un crop YOLO usando GrabCut.

    Estrategia en cascada:
      1. GrabCut con rect centrado (metodo primario).
      2. Segmentacion por color HSV (fallback 1).
      3. Mascara elipse conservadora (fallback 2).

    Retorna:
        pear_mask : np.uint8 mascara binaria
        info      : dict con mask_source, mask_quality_ok, mask_fail_reason
    """
    h, w = crop_img.shape[:2]
    crop_area = max(1, h * w)
    min_area_frac = 0.12

    # --- Intento 1: GrabCut ---
    margin_x = max(8, int(w * 0.08))
    margin_y = max(8, int(h * 0.08))
    gc_rect = (margin_x, margin_y,
               max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    gc_mask = np.zeros((h, w), np.uint8)

    try:
        cv2.grabCut(crop_img, gc_mask, gc_rect,
                    bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        fg = _keep_largest_blob(fg)
        fg = _morpho_clean(fg)
        if np.count_nonzero(fg) >= crop_area * min_area_frac:
            ok, reason = _mask_geometry_ok(fg, h, w)
            if ok:
                return fg, {
                    "mask_source": "grabcut",
                    "mask_quality_ok": True,
                    "mask_fail_reason": "",
                }
    except Exception:
        pass

    # --- Intento 2: Color HSV ---
    color_mask = _pear_color_mask(crop_img)
    color_mask = _keep_largest_blob(color_mask)
    color_mask = _morpho_clean(color_mask)
    if np.count_nonzero(color_mask) >= crop_area * min_area_frac:
        ok, reason = _mask_geometry_ok(color_mask, h, w)
        if ok:
            return color_mask, {
                "mask_source": "color_hsv",
                "mask_quality_ok": True,
                "mask_fail_reason": "",
            }

    # --- Fallback: Elipse ---
    ellipse = _ellipse_mask(h, w, scale=0.80)
    return ellipse, {
        "mask_source": "ellipse_fallback",
        "mask_quality_ok": False,
        "mask_fail_reason": "grabcut_and_color_failed",
    }


# ---------------------------------------------------------------------------
# Detector de defectos YOLO auxiliar
# ---------------------------------------------------------------------------

def run_defect_model_on_crop(crop_image, body_mask, defect_model, defect_conf):
    """
    Ejecuta el modelo YOLO de defectos sobre el crop de la pera y filtra falsos positivos.

    Filtros aplicados:
      D) Rechaza boxes cerca del borde del crop (< 3%).
      E) Rechaza formas muy alargadas cerca del borde (rabillo/pedunculo).
      A) El centro de la bbox debe caer dentro de body_core_mask (erosion del cuerpo).
      B) >= 50% del area de la bbox debe solapar con body_mask.
      C) >= 30% del area de la bbox debe solapar con body_core_mask.

    Retorna:
        valid_dets   : list de (x1, y1, x2, y2, conf, cls)
        ignored_dets : list de (x1, y1, x2, y2, conf, cls, razon)
        metrics      : dict con yolo_defect_count, yolo_defect_area_pct,
                       yolo_defect_max_conf, yolo_defect_classes
    """
    h_c, w_c = crop_image.shape[:2]

    # Mascara de cuerpo central: erosion para eliminar bordes, rabillo y zonas finas
    k_sz = max(7, min(21, int(min(h_c, w_c) * 0.04)))
    if k_sz % 2 == 0:
        k_sz += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    body_core_mask = cv2.erode(body_mask, kernel, iterations=1)

    border_frac = 0.03

    results = defect_model.predict(
        crop_image, conf=defect_conf, imgsz=416, device="cpu", verbose=False
    )
    boxes = results[0].boxes if results and results[0].boxes is not None else None

    valid_dets = []
    ignored_dets = []

    if boxes is None or len(boxes) == 0:
        return valid_dets, ignored_dets, {
            "yolo_defect_count": 0,
            "yolo_defect_area_pct": 0.0,
            "yolo_defect_max_conf": 0.0,
            "yolo_defect_classes": "",
        }

    body_area_px = max(1, int(np.count_nonzero(body_mask)))
    total_valid_area = 0

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(boxes.conf[i].cpu().numpy())
        cls = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        # D) Rechazar boxes demasiado cerca del borde del crop
        bpx = int(w_c * border_frac)
        bpy = int(h_c * border_frac)
        if x1 < bpx or y1 < bpy or x2 > (w_c - bpx) or y2 > (h_c - bpy):
            ignored_dets.append((x1, y1, x2, y2, conf, cls, "border"))
            continue

        # E) Rechazar formas muy alargadas cerca del borde (tipico de rabillo)
        aspect = bh / max(1, bw)
        near_edge = (x1 < w_c * 0.10 or y1 < h_c * 0.10 or
                     x2 > w_c * 0.90 or y2 > h_c * 0.90)
        if (aspect > 3.0 or aspect < 0.33) and near_edge:
            ignored_dets.append((x1, y1, x2, y2, conf, cls, "elongated"))
            continue

        # A) El centro debe caer dentro de body_core_mask
        cx = int(np.clip((x1 + x2) // 2, 0, w_c - 1))
        cy = int(np.clip((y1 + y2) // 2, 0, h_c - 1))
        if body_core_mask[cy, cx] == 0:
            ignored_dets.append((x1, y1, x2, y2, conf, cls, "center_outside_core"))
            continue

        # B) >= 50% del area de la bbox debe estar dentro de body_mask
        bx1 = max(0, x1); by1 = max(0, y1)
        bx2 = min(w_c, x2); by2 = min(h_c, y2)
        bbox_area = max(1, (bx2 - bx1) * (by2 - by1))
        in_body = int(np.count_nonzero(body_mask[by1:by2, bx1:bx2]))
        if in_body / bbox_area < 0.50:
            ignored_dets.append((x1, y1, x2, y2, conf, cls, "low_body_overlap"))
            continue

        # C) >= 30% del area debe solapar con body_core_mask
        in_core = int(np.count_nonzero(body_core_mask[by1:by2, bx1:bx2]))
        if in_core / bbox_area < 0.30:
            ignored_dets.append((x1, y1, x2, y2, conf, cls, "low_core_overlap"))
            continue

        valid_dets.append((x1, y1, x2, y2, conf, cls))
        total_valid_area += bw * bh

    yolo_defect_area_pct = (total_valid_area / body_area_px) * 100.0
    yolo_defect_max_conf = max((d[4] for d in valid_dets), default=0.0)

    class_names = []
    if hasattr(defect_model, "names") and defect_model.names:
        for det in valid_dets:
            name = defect_model.names.get(det[5], str(det[5]))
            if name not in class_names:
                class_names.append(name)
    else:
        for det in valid_dets:
            s = str(det[5])
            if s not in class_names:
                class_names.append(s)

    return valid_dets, ignored_dets, {
        "yolo_defect_count": len(valid_dets),
        "yolo_defect_area_pct": round(yolo_defect_area_pct, 2),
        "yolo_defect_max_conf": round(yolo_defect_max_conf, 3),
        "yolo_defect_classes": ",".join(class_names),
    }


def compute_color_metrics(crop_image, body_mask):
    """
    Calcula metricas de degradacion de color sobre el cuerpo de la pera.

    Metricas:
      brown_dark_pct          : pixeles marron-calido oscuros (hue 5-25, V < 140)
      dark_area_pct           : todos los pixeles oscuros (V < 80) en el cuerpo
      low_saturation_dark_pct : pixeles oscuros de baja saturacion (gris/necrotico)
    """
    body_area_px = max(1, int(np.count_nonzero(body_mask)))
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    body = body_mask > 0

    dark_area_pct = (np.count_nonzero((V < 80) & body) / body_area_px) * 100.0
    brown_dark_pct = (
        np.count_nonzero((H >= 5) & (H <= 25) & (V < 140) & (S > 20) & body)
        / body_area_px * 100.0
    )
    low_saturation_dark_pct = (
        np.count_nonzero((V < 100) & (S < 40) & body) / body_area_px * 100.0
    )

    return {
        "brown_dark_pct": round(brown_dark_pct, 2),
        "dark_area_pct": round(dark_area_pct, 2),
        "low_saturation_dark_pct": round(low_saturation_dark_pct, 2),
    }


# ---------------------------------------------------------------------------
# Validacion de captura (v6)
# ---------------------------------------------------------------------------

def validate_capture(pear_mask, body_mask, image_shape, rules):
    """
    v8: Valida si la captura es adecuada para analisis comercial.

    Puede llamarse ANTES de detect_defects para omitir el analisis pesado
    si la geometria de la mascara es claramente invalida.

    Comprueba: tamano visible, toque de bordes, fragmentacion, irregularidad,
    relleno del bbox y relacion alto/ancho.

    Retorna dict con:
        capture_valid           : bool
        capture_label           : "OK" | "REPETIR FOTO"
        capture_reason          : descripcion del problema (ASCII)
        mask_warning            : bool
        mask_area_pct           : % imagen cubierta por mascara pera
        body_area_pct           : % imagen cubierta por cuerpo
        pear_visible_pct        : igual que mask_area_pct (compatibilidad)
        body_visible_pct        : igual que body_area_pct (compatibilidad)
        bbox_fill_ratio         : area_mascara / area_bbox
        bbox_aspect_ratio       : bbox_h / bbox_w
        mask_components         : num componentes significativos
        border_touch_pct        : % bordes imagen tocados por mascara
        mask_irregularity_ratio : perimetro^2 / (4*pi*area)
    """
    h, w = image_shape[:2]
    image_area = max(1, h * w)

    # Metricas geometricas basicas
    pear_area_px = int(np.count_nonzero(pear_mask))
    body_area_px = int(np.count_nonzero(body_mask)) if body_mask is not None else 0
    mask_area_pct = (pear_area_px / image_area) * 100.0
    body_area_pct_val = (body_area_px / image_area) * 100.0

    # Toque de borde de imagen
    border_px = (
        int(np.count_nonzero(pear_mask[0, :])) +
        int(np.count_nonzero(pear_mask[-1, :])) +
        int(np.count_nonzero(pear_mask[:, 0])) +
        int(np.count_nonzero(pear_mask[:, -1]))
    )
    total_border = max(1, 2 * (h + w) - 4)
    border_touch_pct = (border_px / total_border) * 100.0

    # Componentes conexos significativos
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(pear_mask, connectivity=8)
    min_comp_area = max(100, int(image_area * 0.01))
    significant = sum(1 for i in range(1, n_labels)
                      if stats[i, cv2.CC_STAT_AREA] >= min_comp_area)

    # Metricas de contorno: irregularidad, fill y aspecto del bbox
    bbox_fill_ratio = 0.0
    bbox_aspect_ratio = 1.0
    mask_irregularity_ratio = 1.0

    contours_val, _ = cv2.findContours(pear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_val:
        largest = max(contours_val, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        _bx, _by, bw, bh = cv2.boundingRect(largest)
        bbox_area = max(1, bw * bh)

        if area > 0 and perimeter > 0:
            mask_irregularity_ratio = (perimeter * perimeter) / (4.0 * np.pi * area)
        bbox_fill_ratio = pear_area_px / bbox_area
        bbox_aspect_ratio = bh / max(1, bw)

    geo = {
        "mask_area_pct": round(mask_area_pct, 2),
        "body_area_pct": round(body_area_pct_val, 2),
        "pear_visible_pct": round(mask_area_pct, 2),
        "body_visible_pct": round(body_area_pct_val, 2),
        "bbox_fill_ratio": round(bbox_fill_ratio, 3),
        "bbox_aspect_ratio": round(bbox_aspect_ratio, 3),
        "mask_components": significant,
        "border_touch_pct": round(border_touch_pct, 2),
        "mask_irregularity_ratio": round(mask_irregularity_ratio, 2),
    }

    def _invalid(reason):
        r = {"capture_valid": False, "capture_label": "REPETIR FOTO",
             "capture_reason": reason, "mask_warning": True}
        r.update(geo)
        return r

    if not rules.get("capture_validation_enabled", True):
        r = {"capture_valid": True, "capture_label": "OK",
             "capture_reason": "", "mask_warning": False}
        r.update(geo)
        return r

    # 1. Pera demasiado pequena en la imagen
    min_pear = float(rules.get("min_pear_visible_pct_for_quality", 18.0))
    if mask_area_pct < min_pear:
        return _invalid(f"Pera muy pequena ({mask_area_pct:.1f}% < {min_pear:.0f}%)")

    # 2. Cuerpo insuficiente tras erosion de bordes
    min_body = float(rules.get("min_body_visible_pct_for_quality", 12.0))
    if body_area_pct_val < min_body:
        return _invalid(f"Cuerpo insuficiente ({body_area_pct_val:.1f}% < {min_body:.0f}%)")

    # 3. Mascara toca demasiado el borde (tipico de fondo conectado)
    max_border = float(rules.get("max_border_touch_pct", 12.0))
    if border_touch_pct > max_border:
        return _invalid(f"Mascara toca borde ({border_touch_pct:.1f}% > {max_border:.0f}%)")

    # 4. Mascara demasiado fragmentada
    max_comp = int(rules.get("max_mask_components", 3))
    if significant > max_comp:
        return _invalid(f"Mascara fragmentada ({significant} componentes)")

    # 5. Forma muy irregular
    max_irreg = float(rules.get("max_mask_irregularity_ratio", 8.0))
    if mask_irregularity_ratio > max_irreg:
        return _invalid(f"Mascara irregular (ratio={mask_irregularity_ratio:.1f})")

    # 6. BBox fill: muy rectangular = fondo; muy escasa = fragmentada
    max_fill = float(rules.get("max_mask_bbox_fill_ratio", 0.85))
    min_fill = float(rules.get("min_mask_bbox_fill_ratio", 0.25))
    if bbox_fill_ratio > max_fill:
        return _invalid(f"Mascara rectangular: posible fondo (fill={bbox_fill_ratio:.2f})")
    if bbox_fill_ratio < min_fill:
        return _invalid(f"Mascara muy escasa (fill={bbox_fill_ratio:.2f})")

    # 7. Relacion alto/ancho extraña para una pera
    min_aspect = float(rules.get("min_bbox_aspect_ratio", 0.4))
    max_aspect = float(rules.get("max_bbox_aspect_ratio", 3.5))
    if bbox_aspect_ratio < min_aspect:
        return _invalid(f"Forma extraña: muy ancha (aspecto={bbox_aspect_ratio:.2f})")
    if bbox_aspect_ratio > max_aspect:
        return _invalid(f"Forma extraña: muy estrecha (aspecto={bbox_aspect_ratio:.2f})")

    r = {"capture_valid": True, "capture_label": "OK",
         "capture_reason": "", "mask_warning": False}
    r.update(geo)
    return r


# ---------------------------------------------------------------------------
# Brillo medio del cuerpo de la pera (LAB L)
# ---------------------------------------------------------------------------

def compute_body_l_mean(image, body_mask):
    """
    Devuelve la luminosidad media LAB-L del cuerpo de la pera.

    Usado para detectar imágenes ambiguas: peras naturalmente oscuras o
    subexpuestas donde las métricas HSV de color no son fiables.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    body_px = L[body_mask > 0]
    if len(body_px) == 0:
        return 128.0
    return float(body_px.mean())


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide(body_area_px, defect_mask, rot_mask, rules, pear_area_px=0, image_area=0):
    """
    v4: Logica comercial actualizada para evitar falsos rechazos.

    Manchas marrones suaves, russeting, textura natural y sombras leves NO deben
    provocar rechazo automatico. El rechazo se reserva para defectos extensos,
    podredumbre clara o regiones unicas grandes y graves.

    Clasificacion por categoria (condicion triple: defecto, podredumbre y region mayor):
      EXTRA        defect_pct < 1.0  AND rot_pct < 0.5  AND largest_pct < 1.0
      CATEGORIA I  defect_pct < 3.0  AND rot_pct < 1.5  AND largest_pct < 2.0
      CATEGORIA II defect_pct < 15.0 AND rot_pct < 5.0  AND largest_pct < 12.0
      NO COMERCIAL en caso contrario

    Decision (v5 — rechazo conservador con doble condicion):
      RECHAZA  evidencia_individual_fuerte OR combinacion_triple
               - individual: defect >= 30 OR rot >= 12 OR largest >= 20
               - combo     : defect >= 12 AND rot >= 8 AND largest >= 7
               Color natural alto (defect alto, rot=0) NO es suficiente para RECHAZA.
      REVISAR  zona limitrofe (cat2 <= valor < rechazo) o cualquier CATEGORIA II
      PASA     EXTRA o CATEGORIA I sin condiciones de REVISAR/RECHAZA

    decision : PASA | REVISAR | RECHAZA
    category : EXTRA | CATEGORIA I | CATEGORIA II | NO COMERCIAL  (ASCII puro)
    metrics  : dict con porcentajes y conteos de pixeles
    """
    defect_px = int(np.count_nonzero(defect_mask))
    rot_px = int(np.count_nonzero(rot_mask))

    ref_px = body_area_px if body_area_px > 0 else pear_area_px

    pear_visible_pct = (pear_area_px / image_area * 100.0) if image_area > 0 else 0.0
    body_visible_pct = (body_area_px / image_area * 100.0) if image_area > 0 else 0.0

    if ref_px == 0:
        metrics = {
            "defect_pct": 0.0, "rot_pct": 0.0,
            "largest_defect_pct": 0.0,
            "pear_area_px": pear_area_px, "body_area_px": body_area_px,
            "defect_px": 0, "rot_px": 0,
            "pear_visible_pct": round(pear_visible_pct, 2),
            "body_visible_pct": round(body_visible_pct, 2),
            "estimated_category": "NO COMERCIAL",
            "display_label": "REVISAR - POSIBLE NO COMERCIAL",
        }
        return "REVISAR", "NO COMERCIAL", metrics

    defect_pct = (defect_px / ref_px) * 100.0
    rot_pct = (rot_px / ref_px) * 100.0
    largest_pct = _largest_region_pct(defect_mask, ref_px)

    metrics = {
        "defect_pct": round(defect_pct, 2),
        "rot_pct": round(rot_pct, 2),
        "largest_defect_pct": round(largest_pct, 2),
        "pear_area_px": pear_area_px,
        "body_area_px": body_area_px,
        "defect_px": defect_px,
        "rot_px": rot_px,
        "pear_visible_pct": round(pear_visible_pct, 2),
        "body_visible_pct": round(body_visible_pct, 2),
    }

    # Umbrales de categoria (triple condicion: defecto total, podredumbre, region mayor)
    extra_def = float(rules.get("max_extra_defect_pct", 1.0))
    extra_rot = float(rules.get("max_extra_rot_pct", 0.5))
    extra_reg = float(rules.get("max_extra_region_pct", 1.0))

    cat1_def = float(rules.get("max_category_i_defect_pct", 3.0))
    cat1_rot = float(rules.get("max_category_i_rot_pct", 1.5))
    cat1_reg = float(rules.get("max_category_i_region_pct", 2.0))

    cat2_def = float(rules.get("max_category_ii_defect_pct", 15.0))
    cat2_rot = float(rules.get("max_category_ii_rot_pct", 5.0))
    cat2_reg = float(rules.get("max_category_ii_region_pct", 12.0))

    # Umbrales de rechazo individual — evidencia fuerte en una sola metrica
    reject_def = float(rules.get("reject_defect_pct", 30.0))
    reject_rot = float(rules.get("reject_dark_rot_pct", 12.0))
    reject_reg = float(rules.get("reject_region_pct", 20.0))

    # Umbrales de rechazo combinado — las tres metricas deben superarse juntas
    combo_def = float(rules.get("combo_reject_defect_pct", 12.0))
    combo_rot = float(rules.get("combo_reject_rot_pct", 8.0))
    combo_reg = float(rules.get("combo_reject_region_pct", 7.0))

    # --- Clasificacion comercial (triple condicion) ---
    if defect_pct < extra_def and rot_pct < extra_rot and largest_pct < extra_reg:
        category = "EXTRA"
    elif defect_pct < cat1_def and rot_pct < cat1_rot and largest_pct < cat1_reg:
        category = "CATEGORIA I"
    elif defect_pct < cat2_def and rot_pct < cat2_rot and largest_pct < cat2_reg:
        category = "CATEGORIA II"
    else:
        category = "NO COMERCIAL"

    # --- Decision final (v5) ---
    # RECHAZA requiere evidencia fuerte: un unico valor muy alto O la triple combinacion.
    # Coloracion natural alta (defect alto, rot=0) nunca debe provocar RECHAZA directo.
    reject_single = (defect_pct >= reject_def or rot_pct >= reject_rot or largest_pct >= reject_reg)
    reject_combo = (defect_pct >= combo_def and rot_pct >= combo_rot and largest_pct >= combo_reg)

    if reject_single or reject_combo:
        decision = "RECHAZA"
        category = "NO COMERCIAL"
    elif defect_pct >= cat2_def or rot_pct >= cat2_rot or largest_pct >= cat2_reg:
        # Zona limitrofe entre CATEGORIA II y NO COMERCIAL: revision humana obligatoria
        decision = "REVISAR"
    elif category == "CATEGORIA II":
        # CATEGORIA II siempre requiere revision (defectos visibles pero no graves)
        decision = "REVISAR"
    else:
        decision = "PASA"

    # display_label: etiqueta visible final con distincion POSIBLE NO COMERCIAL
    if decision == "PASA":
        display_label = f"PASA - {category}"
    elif decision == "REVISAR" and category == "NO COMERCIAL":
        display_label = "REVISAR - POSIBLE NO COMERCIAL"
    else:
        display_label = f"{decision} - {category}"

    metrics["estimated_category"] = category
    metrics["display_label"] = display_label

    return decision, category, metrics


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------

def visualize(image, pear_mask, defect_mask, rot_mask, decision, category, metrics,
              warning=None, body_mask=None, shadow_mask=None, capture_reason=None,
              yolo_valid_boxes=None, yolo_ignored_boxes=None, yolo_metrics=None):
    """
    v3: Overlay azul suave para sombras detectadas (informativo, no penaliza).
    Texto ASCII puro para evitar "???" en cv2.putText Windows.

    Muestra:
      - overlay verde suave: pera segmentada
      - overlay azul suave: zona de sombra (no cuenta como defecto)
      - overlay rojo: defectos reales
      - overlay naranja: podredumbre
      - contorno verde: borde de la pera
      - decision y categoria arriba; metricas a la izquierda
    """
    vis = image.copy()
    overlay = np.zeros_like(vis)

    # Overlay pera (verde muy suave)
    overlay[pear_mask > 0] = (0, 60, 0)
    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    # Overlay sombra (azul suave, informativo)
    if shadow_mask is not None and shadow_mask.any():
        overlay[:] = 0
        overlay[shadow_mask > 0] = (120, 40, 0)
        cv2.addWeighted(overlay, 0.28, vis, 0.72, 0, vis)

    # Overlay defectos reales (rojo)
    overlay[:] = 0
    overlay[defect_mask > 0] = (0, 0, 200)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    # Overlay podredumbre (naranja, encima de defectos)
    overlay[:] = 0
    overlay[rot_mask > 0] = (0, 100, 230)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    # Contorno verde de la pera
    contours, _ = cv2.findContours(pear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 210, 0), 2)

    # Cajas YOLO de defectos: validos en azul, ignorados en gris (opcionales)
    if yolo_valid_boxes:
        for (x1, y1, x2, y2, conf, cls) in yolo_valid_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 0), 1, cv2.LINE_AA)

    color_map = {
        "PASA": (0, 200, 0),
        "REVISAR": (0, 165, 255),
        "RECHAZA": (0, 0, 220),
    }
    dec_color = color_map.get(decision, (200, 200, 200))

    # Texto ASCII puro — sin tildes ni guion largo para evitar "???" en cv2.putText
    top_label = metrics.get("display_label", f"{decision} - {category}")
    lines = [(top_label, dec_color, 0.72)]

    # Si captura no valida, mostrar motivo justo debajo del label principal
    if capture_reason:
        safe_reason = capture_reason.encode("ascii", errors="replace").decode("ascii")
        lines.append((f"Motivo: {safe_reason}", (200, 200, 100), 0.46))

    lines += [
        (f"DEFECTO TOTAL : {metrics.get('defect_pct', 0):.1f}%", (220, 220, 220), 0.52),
        (f"PODREDUMBRE   : {metrics.get('rot_pct', 0):.1f}%", (220, 220, 220), 0.52),
        (f"REGION MAYOR  : {metrics.get('largest_defect_pct', 0):.1f}%", (220, 220, 220), 0.52),
        (f"PERA VISIBLE  : {metrics.get('pear_visible_pct', 0):.1f}%", (180, 180, 180), 0.46),
        (f"CUERPO VISIBLE: {metrics.get('body_visible_pct', 0):.1f}%", (180, 180, 180), 0.46),
    ]

    if yolo_metrics:
        n_valid = yolo_metrics.get("yolo_defect_count", 0)
        n_ignored = len(yolo_ignored_boxes) if yolo_ignored_boxes else 0
        brown_pct = yolo_metrics.get("brown_dark_pct", 0.0)
        lines += [
            (f"YOLO defects valid  : {n_valid}", (255, 180, 50), 0.46),
            (f"YOLO defects ignored: {n_ignored}", (160, 160, 160), 0.42),
            (f"brown_dark_pct      : {brown_pct:.1f}%", (170, 220, 160), 0.42),
        ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30
    for text, color, scale in lines:
        cv2.putText(vis, text, (10, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), font, scale, color, 1, cv2.LINE_AA)
        y += int(scale * 46 + 6)

    if warning:
        safe_warn = warning.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(vis, f"! {safe_warn}", (10, y + 4), font, 0.44, (0, 165, 255), 1, cv2.LINE_AA)

    return vis
