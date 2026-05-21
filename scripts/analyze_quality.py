"""
analyze_quality.py — Analiza calidad visual de peras con el motor rule-based v1.

Pipeline: carga → [detección YOLO opcional] → segmentación → detección de defectos → decisión → visualización.

El sistema solo evalúa defectos superficiales visibles.
No mide sabor, azúcar/Brix, firmeza, madurez interna ni calidad nutricional.

Uso:
    # Una sola imagen
    python scripts/analyze_quality.py --image data/samples/pear_01.jpg --show

    # Carpeta completa, guardar visualizaciones
    python scripts/analyze_quality.py --source data/samples --save

    # Con reglas personalizadas
    python scripts/analyze_quality.py --source data/samples --rules configs/quality_rules.yaml --save

    # Con detector YOLO para recorte previo
    python scripts/analyze_quality.py --source data/samples --save --use-detector --detect-conf 0.50
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from src.segmentation import load_image, segment_pear, compute_body_mask
from src.quality_analysis import (
    detect_defects, decide, visualize, validate_capture,
    run_defect_model_on_crop, compute_color_metrics,
    build_detector_roi_pear_mask, compute_body_l_mean,
)

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imwrite_unicode(path, image):
    """cv2.imwrite compatible con rutas Unicode en Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"No se pudo codificar imagen: {path}")
    buffer.tofile(str(path))


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _collect_images(source: Path) -> list:
    if source.is_file():
        return [source]
    imgs = []
    for ext in VALID_EXT:
        imgs.extend(source.glob(f"*{ext}"))
        imgs.extend(source.glob(f"*{ext.upper()}"))
    return sorted(set(imgs))


def _resize_if_needed(image, max_size):
    """Redimensiona si alguna dimension supera max_size, manteniendo proporcion.

    Retorna (image, orig_w, orig_h, proc_w, proc_h).
    Si max_size <= 0 o la imagen ya cabe, devuelve la imagen original sin cambios.
    """
    h, w = image.shape[:2]
    if max_size <= 0 or (h <= max_size and w <= max_size):
        return image, w, h, w, h
    scale = max_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, w, h, new_w, new_h


def _load_yolo_model(model_path: Path):
    """Carga el modelo YOLO desde ultralytics. Retorna None si falla."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        print(f"ERROR cargando YOLO: {e}")
        return None


def _load_u3_classifier(model_path: Path, thresholds_path: Path):
    """Carga el clasificador U3 (MobileNetV3-small PyTorch). Retorna (model, thresholds) o (None, None) si falla."""
    try:
        import json
        import torch
        import torchvision.models as tvm
        model = tvm.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()
        with open(str(thresholds_path), encoding="utf-8") as fh:
            thr = json.load(fh)
        return model, thr
    except Exception as e:
        print(f"ERROR cargando U3: {e}")
        return None, None


def _make_u3_gray_input(image_bgr: np.ndarray, size: int = 224):
    """Convierte imagen BGR a gray_bg_clean PIL 224x224 para inferencia U3.

    Estima el color de fondo por las esquinas, genera mascara por distancia LAB,
    reemplaza el fondo por gris neutro (128,128,128). Retorna imagen PIL RGB o None si falla.
    """
    try:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)
        cs = max(3, size // 20)
        corners = np.concatenate([
            img_r[:cs, :cs].reshape(-1, 3),
            img_r[:cs, -cs:].reshape(-1, 3),
            img_r[-cs:, :cs].reshape(-1, 3),
            img_r[-cs:, -cs:].reshape(-1, 3),
        ])
        bg_rgb = np.median(corners, axis=0).astype(np.uint8)
        lab = cv2.cvtColor(img_r, cv2.COLOR_RGB2LAB).astype(float)
        bg_lab = cv2.cvtColor(bg_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        dist = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=2))
        fg = (dist > 25).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
        gray_bg = np.full_like(img_r, 128)
        result_rgb = np.where(fg[:, :, np.newaxis] > 0, img_r, gray_bg).astype(np.uint8)
        from PIL import Image as _PIL
        return _PIL.fromarray(result_rgb)
    except Exception:
        return None


def _run_u3_inference(u3_model, gray_pil, thresholds: dict):
    """Corre inferencia U3 sobre imagen PIL gray_bg_clean. Retorna (p_bad, p_good, decision_raw)."""
    try:
        import torch
        import torchvision.transforms as tT
        _tf = tT.Compose([
            tT.Resize((224, 224)),
            tT.ToTensor(),
            tT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = _tf(gray_pil).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(u3_model(tensor), dim=1)[0].cpu().numpy()
        p_bad, p_good = float(probs[0]), float(probs[1])
        bad_thr  = thresholds.get("bad_reject_threshold",  0.60)
        good_thr = thresholds.get("good_accept_threshold", 0.55)
        if p_bad >= bad_thr:
            raw = "U3_BAD"
        elif p_good >= good_thr:
            raw = "U3_GOOD"
        else:
            raw = "U3_REVIEW"
        return p_bad, p_good, raw
    except Exception as e:
        return 0.5, 0.5, "U3_ERROR"


def _detect_pear_roi(image, yolo_model, detect_conf, crop_margin):
    """
    Ejecuta detección YOLO sobre image y retorna info del recorte.

    Selecciona solo la detección de mayor confianza (ignora el resto).

    Returns dict con:
      found, crop, bbox_vis, detector_conf, crop_x1, crop_y1, crop_x2, crop_y2
    """
    h_img, w_img = image.shape[:2]
    results = yolo_model.predict(
        image, conf=detect_conf, imgsz=416, device="cpu", verbose=False
    )

    boxes = results[0].boxes if results and results[0].boxes is not None else None

    if boxes is None or len(boxes) == 0:
        return {
            "found": False,
            "crop": None,
            "bbox_vis": image.copy(),
            "detector_conf": 0.0,
            "crop_x1": 0, "crop_y1": 0,
            "crop_x2": w_img, "crop_y2": h_img,
        }

    confs = boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    best_conf = float(confs[best_idx])
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * crop_margin)
    my = int(bh * crop_margin)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w_img, x2 + mx)
    cy2 = min(h_img, y2 + my)

    bbox_vis = image.copy()
    cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        bbox_vis, f"pear {best_conf:.2f}",
        (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )

    return {
        "found": True,
        "crop": image[cy1:cy2, cx1:cx2],
        "bbox_vis": bbox_vis,
        "detector_conf": best_conf,
        "crop_x1": cx1, "crop_y1": cy1,
        "crop_x2": cx2, "crop_y2": cy2,
    }


def _process_one(img_path: Path, seg_config: dict, rules: dict, args,
                 yolo_model=None, defect_model=None, defect_conf=0.25,
                 quality_cls_model=None, quality_cls_bad_thr=0.85,
                 quality_cls_affect_decision=False,
                 u3_model=None, u3_thresholds=None, u3_safe_mode=True) -> dict:
    image = load_image(img_path)

    max_size = getattr(args, "max_size", 1280)
    image, orig_w, orig_h, proc_w, proc_h = _resize_if_needed(image, max_size)
    if orig_w != proc_w or orig_h != proc_h:
        print(f"    original={orig_w}x{orig_h} -> resized={proc_w}x{proc_h}")

    detect_conf_used = getattr(args, "detect_conf", 0.50)
    crop_margin_used = getattr(args, "crop_margin", 0.08)

    crop_info = {
        "detector_used": yolo_model is not None,
        "detector_conf": 0.0,
        "crop_x1": 0, "crop_y1": 0,
        "crop_x2": image.shape[1], "crop_y2": image.shape[0],
        "crop_margin": crop_margin_used if yolo_model is not None else 0.0,
        "capture_reason": "",
    }

    image_for_analysis = image
    bbox_vis = None
    crop_image = None
    use_roi_mask = False
    roi_mask_info = {"mask_source": "classic", "mask_quality_ok": True, "mask_fail_reason": ""}

    if yolo_model is not None:
        det = _detect_pear_roi(image, yolo_model, detect_conf_used, crop_margin_used)
        bbox_vis = det["bbox_vis"]

        crop_info["detector_conf"] = det["detector_conf"]
        crop_info["crop_x1"] = det["crop_x1"]
        crop_info["crop_y1"] = det["crop_y1"]
        crop_info["crop_x2"] = det["crop_x2"]
        crop_info["crop_y2"] = det["crop_y2"]

        if not det["found"]:
            capture_reason = f"YOLO: pera no detectada (conf>={detect_conf_used})"
            crop_info["capture_reason"] = capture_reason

            decision = "REVISAR"
            category = "CAPTURA NO VALIDA"
            estimated_category = "CAPTURA NO VALIDA"
            display_label = "REPETIR FOTO - PERA NO DETECTADA"

            metrics = {
                "defect_pct": 0.0, "rot_pct": 0.0, "largest_defect_pct": 0.0,
                "pear_visible_pct": 0.0, "body_visible_pct": 0.0,
                "pear_area_px": 0, "body_area_px": 0,
                "defect_px": 0, "rot_px": 0,
                "estimated_category": estimated_category,
                "display_label": display_label,
                "mask_warning": True, "capture_valid": False,
                "capture_label": "REPETIR FOTO", "capture_reason": capture_reason,
                "original_width": orig_w, "original_height": orig_h,
                "processed_width": proc_w, "processed_height": proc_h,
                "mask_area_pct": 0.0, "body_area_pct": 0.0,
                "bbox_fill_ratio": 0.0, "bbox_aspect_ratio": 0.0,
                "mask_components": 0, "border_touch_pct": 0.0,
                "mask_irregularity_ratio": 0.0,
            }
            h_z, w_z = image.shape[:2]
            zero = np.zeros((h_z, w_z), dtype=np.uint8)

            print(
                f"  {img_path.name:<28} "
                f"{decision:<9}{'CAPTURA NO VALIDA':<16}{display_label:<34} "
                f"  0.0%    0.0%    0.0%    0.0%     0.0% [!]"
            )
            print(f"    CAPTURA: {capture_reason}")

            if args.save or args.show:
                vis = visualize(
                    image, zero, zero, zero,
                    decision, category, metrics, warning=None,
                    body_mask=zero, shadow_mask=zero,
                    capture_reason=capture_reason,
                )
                if args.show:
                    cv2.imshow(img_path.name, vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                if args.save:
                    out_dir = PROJECT_ROOT / "outputs" / "quality_analysis"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    imwrite_unicode(out_dir / img_path.name, vis)
                    imwrite_unicode(out_dir / (img_path.stem + "_bbox.jpg"), bbox_vis)

            return {
                "decision": decision, "category": category,
                "estimated_category": estimated_category,
                "display_label": display_label,
                "mask_warning": True, "capture_valid": False,
                "capture_label": "REPETIR FOTO", "capture_reason": capture_reason,
                "orig_w": orig_w, "orig_h": orig_h,
                "proc_w": proc_w, "proc_h": proc_h,
                "metrics": metrics,
                "pear_mask": zero, "body_mask": zero,
                "defect_mask": zero, "defect_mask_raw": zero,
                "shadow_mask": zero, "rot_mask": zero,
                "warning": None, "image": img_path.name,
                "crop_info": crop_info,
            }

        # Pera detectada: usar recorte para el análisis
        image_for_analysis = det["crop"]
        crop_image = det["crop"]
        use_roi_mask = True
        print(
            f"    YOLO conf={det['detector_conf']:.2f} "
            f"crop=({det['crop_x1']},{det['crop_y1']},{det['crop_x2']},{det['crop_y2']})"
        )

    # --- Análisis de calidad sobre image_for_analysis (recorte o imagen completa) ---
    image_area = image_for_analysis.shape[0] * image_for_analysis.shape[1]

    if use_roi_mask:
        pear_mask, roi_mask_info = build_detector_roi_pear_mask(image_for_analysis, rules)
        print(
            f"    Mascara ROI: source={roi_mask_info['mask_source']} "
            f"ok={roi_mask_info['mask_quality_ok']}"
            + (f" reason={roi_mask_info['mask_fail_reason']}" if roi_mask_info["mask_fail_reason"] else "")
        )
    else:
        pear_mask = segment_pear(image_for_analysis, seg_config)
    pear_area_px = int((pear_mask > 0).sum())
    pear_area_image_pct = (pear_area_px / image_area) * 100.0

    body_mask = compute_body_mask(pear_mask, rules)
    body_area_px = int((body_mask > 0).sum())

    warning = None
    mask_warning = False
    min_area_pct = float(rules.get("min_pear_area_pct", 5.0))
    if pear_area_image_pct < min_area_pct:
        warning = f"Pera pequena ({pear_area_image_pct:.1f}% < {min_area_pct}%); se analiza de todas formas"
        mask_warning = True

    capture_info = validate_capture(pear_mask, body_mask, image_for_analysis.shape, rules)
    mask_warning = mask_warning or capture_info["mask_warning"]

    if mask_warning and capture_info["capture_valid"]:
        capture_info = dict(capture_info)
        capture_info["capture_valid"] = False
        capture_info["capture_label"] = "REPETIR FOTO"
        capture_info["capture_reason"] = (
            warning if warning else "Mascara dudosa: area pequena"
        )
        capture_info["mask_warning"] = True

    if not capture_info["capture_valid"]:
        decision = "REVISAR"
        category = "CAPTURA NO VALIDA"
        estimated_category = "CAPTURA NO VALIDA"
        display_label = "REPETIR FOTO - CAPTURA NO VALIDA"
        metrics = {
            "defect_pct": 0.0, "rot_pct": 0.0, "largest_defect_pct": 0.0,
            "pear_visible_pct": capture_info["pear_visible_pct"],
            "body_visible_pct": capture_info["body_visible_pct"],
            "pear_area_px": pear_area_px, "body_area_px": body_area_px,
            "defect_px": 0, "rot_px": 0,
            "estimated_category": estimated_category,
            "display_label": display_label,
        }
        defect_mask = np.zeros(pear_mask.shape, dtype=np.uint8)
        rot_mask = np.zeros(pear_mask.shape, dtype=np.uint8)
        shadow_mask = np.zeros(pear_mask.shape, dtype=np.uint8)
        defect_mask_raw = np.zeros(pear_mask.shape, dtype=np.uint8)
    else:
        defect_mask, rot_mask, shadow_mask, defect_mask_raw = detect_defects(
            image_for_analysis, pear_mask, body_mask, rules
        )
        decision, category, metrics = decide(
            body_area_px, defect_mask, rot_mask, rules,
            pear_area_px=pear_area_px, image_area=image_area,
        )
        display_label = metrics.get("display_label", f"{decision} - {category}")
        estimated_category = metrics.get("estimated_category", category)

    # --- Cap por brillo corporal ambiguo ---
    # Si el cuerpo tiene luminosidad LAB-L en el rango 45-70 (oscuro pero no necrosis)
    # Y rot_pct > 50%, las metricas HSV no son fiables (pera oscura natural o subexpuesta).
    # Por debajo de L=45 la oscuridad es extrema y probablemente real; encima de L=70 es normal.
    body_l_mean = 128.0
    if capture_info["capture_valid"] and body_mask is not None and body_mask.any():
        body_l_mean = compute_body_l_mean(image_for_analysis, body_mask)
        _L_LOW, _L_HIGH, _ROT_THR = 45.0, 70.0, 50.0
        if (_L_LOW < body_l_mean < _L_HIGH and
                metrics.get("rot_pct", 0.0) > _ROT_THR and
                decision == "RECHAZA"):
            decision = "REVISAR"
            display_label = (
                f"REVISAR - brillo ambiguo (L={body_l_mean:.0f} "
                f"rot={metrics['rot_pct']:.0f}%)"
            )
            metrics["display_label"] = display_label
    metrics["body_l_mean"] = round(body_l_mean, 1)

    # Si la mascara es fallback (elipse), cap RECHAZA → REVISAR (punto 4 del prompt)
    if not roi_mask_info.get("mask_quality_ok", True) and decision == "RECHAZA":
        decision = "REVISAR"
        metrics["display_label"] = "REVISAR - POSIBLE NO COMERCIAL (mascara fallback)"
        display_label = metrics["display_label"]

    capture_reason = capture_info["capture_reason"]
    if yolo_model is not None:
        crop_info["capture_reason"] = capture_reason

    # Almacenar info de mascara en crop_info para CSV y debug
    crop_info["mask_source"] = roi_mask_info.get("mask_source", "classic")
    crop_info["mask_quality_ok"] = roi_mask_info.get("mask_quality_ok", True)
    crop_info["mask_fail_reason"] = roi_mask_info.get("mask_fail_reason", "")

    metrics["mask_warning"] = mask_warning
    metrics["capture_valid"] = capture_info["capture_valid"]
    metrics["capture_label"] = capture_info["capture_label"]
    metrics["capture_reason"] = capture_reason
    metrics["original_width"] = orig_w
    metrics["original_height"] = orig_h
    metrics["processed_width"] = proc_w
    metrics["processed_height"] = proc_h
    metrics["mask_area_pct"] = capture_info["mask_area_pct"]
    metrics["body_area_pct"] = capture_info["body_area_pct"]
    metrics["bbox_fill_ratio"] = capture_info["bbox_fill_ratio"]
    metrics["bbox_aspect_ratio"] = capture_info["bbox_aspect_ratio"]
    metrics["mask_components"] = capture_info["mask_components"]
    metrics["border_touch_pct"] = capture_info["border_touch_pct"]
    metrics["mask_irregularity_ratio"] = capture_info["mask_irregularity_ratio"]

    # --- Modelo YOLO de defectos (señal auxiliar) ---
    yolo_valid_boxes = []
    yolo_ignored_boxes = []
    yolo_defect_metrics = {
        "yolo_defect_count": 0, "yolo_defect_area_pct": 0.0,
        "yolo_defect_max_conf": 0.0, "yolo_defect_classes": "",
        "brown_dark_pct": 0.0, "dark_area_pct": 0.0,
        "low_saturation_dark_pct": 0.0,
    }

    if defect_model is not None and capture_info["capture_valid"]:
        valid_dets, ignored_dets, det_metrics = run_defect_model_on_crop(
            image_for_analysis, body_mask, defect_model, defect_conf
        )
        color_metrics = compute_color_metrics(image_for_analysis, body_mask)
        yolo_valid_boxes = valid_dets
        yolo_ignored_boxes = ignored_dets
        yolo_defect_metrics.update(det_metrics)
        yolo_defect_metrics.update(color_metrics)

        # El modelo PSD puede subir la decision a REVISAR pero nunca directamente a RECHAZA
        if decision == "PASA":
            n_valid = det_metrics["yolo_defect_count"]
            max_conf_yolo = det_metrics["yolo_defect_max_conf"]
            brown_pct = color_metrics["brown_dark_pct"]
            if n_valid >= 2 or max_conf_yolo > 0.65:
                decision = "REVISAR"
                metrics["display_label"] = f"REVISAR - YOLO ({n_valid} defectos)"
                display_label = metrics["display_label"]
            elif brown_pct > 15.0:
                decision = "REVISAR"
                metrics["display_label"] = f"REVISAR - color degradado ({brown_pct:.1f}%)"
                display_label = metrics["display_label"]

        print(
            f"    YOLO defectos: validos={len(yolo_valid_boxes)} "
            f"ignorados={len(yolo_ignored_boxes)} "
            f"max_conf={det_metrics['yolo_defect_max_conf']:.2f}"
        )
        print(
            f"    COLOR: brown_dark={color_metrics['brown_dark_pct']:.1f}% "
            f"dark={color_metrics['dark_area_pct']:.1f}%"
        )

    metrics.update(yolo_defect_metrics)

    # --- Clasificador auxiliar de calidad GOOD/BAD (Mendeley) ---
    quality_cls_action_text = ""
    quality_cls_metrics = {
        "quality_cls_used": False,
        "quality_cls_source": "not_used",
        "quality_cls_pred": "unknown",
        "quality_cls_good_conf": 0.0,
        "quality_cls_bad_conf": 0.0,
        "quality_cls_max_conf": 0.0,
        "quality_cls_action": "",
    }

    if quality_cls_model is not None and capture_info["capture_valid"]:
        cls_img = crop_image if crop_image is not None else image_for_analysis
        cls_source = "crop" if crop_image is not None else "full_image"
        try:
            cls_results = quality_cls_model.predict(
                cls_img, imgsz=224, device="cpu", verbose=False
            )
            probs = cls_results[0].probs
            if probs is not None:
                cls_names = quality_cls_model.names
                good_idx_cls = None
                bad_idx_cls = None
                for idx, name in cls_names.items():
                    if name.lower() == "good":
                        good_idx_cls = idx
                    elif name.lower() == "bad":
                        bad_idx_cls = idx
                if good_idx_cls is None:
                    good_idx_cls = 1
                if bad_idx_cls is None:
                    bad_idx_cls = 0
                all_probs_cls = probs.data.cpu().numpy()
                good_conf_cls = float(all_probs_cls[good_idx_cls]) if good_idx_cls < len(all_probs_cls) else 0.0
                bad_conf_cls = float(all_probs_cls[bad_idx_cls]) if bad_idx_cls < len(all_probs_cls) else 0.0
                pred_idx_cls = int(probs.top1)
                pred_class_cls = cls_names.get(pred_idx_cls, "unknown").lower()
                max_conf_cls = float(probs.top1conf.cpu().numpy())

                quality_cls_metrics = {
                    "quality_cls_used": True,
                    "quality_cls_source": cls_source,
                    "quality_cls_pred": pred_class_cls,
                    "quality_cls_good_conf": round(good_conf_cls, 4),
                    "quality_cls_bad_conf": round(bad_conf_cls, 4),
                    "quality_cls_max_conf": round(max_conf_cls, 4),
                    "quality_cls_action": "sin_efecto",
                }
                print(
                    f"    QUALITY CLS: {pred_class_cls.upper()} {max_conf_cls:.2f} "
                    f"(good={good_conf_cls:.2f} bad={bad_conf_cls:.2f}) source={cls_source}"
                )

                if quality_cls_affect_decision:
                    prev_dec = decision
                    if decision == "PASA" and pred_class_cls == "bad" and bad_conf_cls >= quality_cls_bad_thr:
                        decision = "REVISAR"
                        estimated_category = "POSIBLE NO COMERCIAL"
                        display_label = "REVISAR - POSIBLE NO COMERCIAL"
                        metrics["display_label"] = display_label
                        metrics["estimated_category"] = estimated_category
                        quality_cls_metrics["quality_cls_action"] = f"{prev_dec}->REVISAR (bad={bad_conf_cls:.2f})"
                        quality_cls_action_text = f"QUALITY ACTION: {prev_dec} -> REVISAR"
                        print(f"    QUALITY ACTION: {prev_dec} -> REVISAR (bad_conf={bad_conf_cls:.2f})")
                    elif decision == "REVISAR" and pred_class_cls == "bad" and bad_conf_cls >= quality_cls_bad_thr:
                        quality_cls_metrics["quality_cls_action"] = f"REVISAR confirmado (bad={bad_conf_cls:.2f})"
                        print(f"    QUALITY ACTION: REVISAR confirmado (bad_conf={bad_conf_cls:.2f})")
                    elif decision == "RECHAZA":
                        quality_cls_metrics["quality_cls_action"] = "RECHAZA_sin_cambio"
                    else:
                        quality_cls_metrics["quality_cls_action"] = f"sin_efecto ({pred_class_cls})"
            else:
                quality_cls_metrics["quality_cls_source"] = "error"
                quality_cls_metrics["quality_cls_action"] = "probs_none"
        except Exception as e:
            quality_cls_metrics["quality_cls_source"] = "error"
            quality_cls_metrics["quality_cls_action"] = f"error:{str(e)[:40]}"
            print(f"    QUALITY CLS error: {e}")

    metrics.update(quality_cls_metrics)

    # --- Clasificador U3 ROI masked clean (PyTorch MobileNetV3-small) ---
    u3_metrics = {
        "quality_u3_enabled": u3_model is not None,
        "quality_u3_status": "not_used",
        "quality_u3_pred": "",
        "quality_u3_p_good": 0.0,
        "quality_u3_p_bad": 0.0,
        "quality_u3_decision_raw": "",
        "quality_u3_decision_safe": "",
        "final_decision_before_u3": decision,
        "final_decision_after_u3": decision,
        "final_decision_reason": "u3_not_used",
    }

    if u3_model is not None and capture_info["capture_valid"] and u3_thresholds is not None:
        gray_pil = _make_u3_gray_input(image_for_analysis)
        decision_before_u3 = decision

        if gray_pil is None:
            u3_metrics["quality_u3_status"] = "MASK_FAIL"
            u3_metrics["quality_u3_decision_raw"] = "U3_REVIEW"
            u3_metrics["quality_u3_decision_safe"] = "REVIEW"
            u3_metrics["final_decision_reason"] = "mask_fail"
            if decision == "PASA":
                decision = "REVISAR"
                display_label = "REVISAR - U3 mascara fallida"
                metrics["display_label"] = display_label
        else:
            p_bad, p_good, u3_raw = _run_u3_inference(u3_model, gray_pil, u3_thresholds)
            # strong_defect: evidencia REAL de defecto (YOLO detector o necrosis extrema).
            # NO incluye: decision_before_u3==RECHAZA, porque eso es exactamente el falso rechazo
            # que U3 debe corregir. NO incluye russeting, color marrón, sombra ni piel rugosa.
            strong_defect = (
                yolo_defect_metrics.get("yolo_defect_count", 0) >= 2
                or yolo_defect_metrics.get("yolo_defect_max_conf", 0.0) > 0.65
                or (metrics.get("rot_pct", 0.0) > 50.0 and metrics.get("body_l_mean", 128.0) < 45.0)
            )

            # Umbral calibrado: el más agresivo con GOOD->RECHAZA=0 sobre dataset corregido.
            # Ver: outputs/u3_bad_to_reject_policy_calibration_v1/selected_policy.json
            BAD_REJECT_POLICY_THR = 0.995

            if u3_raw == "U3_BAD":
                if p_bad >= BAD_REJECT_POLICY_THR:
                    # Alta confianza: rechaza directamente sin depender de safe_mode ni strong_defect.
                    u3_safe = "BAD"
                    reason = f"U3_BAD_STRONG_REJECT(p_bad={p_bad:.3f}>={BAD_REJECT_POLICY_THR})"
                elif u3_safe_mode and not strong_defect:
                    u3_safe = "REVIEW"
                    reason = f"U3_BAD_LOW_CONF_REVIEW(p_bad={p_bad:.3f}<{BAD_REJECT_POLICY_THR})"
                else:
                    u3_safe = "BAD"
                    reason = f"U3_BAD_AND_STRONG_DEFECT(p_bad={p_bad:.3f})"
            elif u3_raw == "U3_REVIEW":
                u3_safe = "REVIEW"
                reason = f"U3_REVIEW (p_bad={p_bad:.2f})"
            else:  # U3_GOOD
                u3_safe = "GOOD"
                reason = f"U3_GOOD (p_good={p_good:.2f})"

            # Fusión corregida: U3=GOOD protege activamente contra falsos rechazos.
            if u3_safe == "BAD":
                decision = "RECHAZA"
                display_label = f"RECHAZA - U3 BAD (p_bad={p_bad:.2f})"
                metrics["display_label"] = display_label
            elif u3_safe == "REVIEW":
                if decision == "PASA":
                    decision = "REVISAR"
                    display_label = f"REVISAR - U3 dudoso (p_bad={p_bad:.2f})"
                    metrics["display_label"] = display_label
                reason = f"U3_REVIEW (p_bad={p_bad:.2f})"
            elif u3_safe == "GOOD":
                if p_good >= 0.85 and not strong_defect:
                    decision = "PASA"
                    display_label = f"PASA - U3 GOOD fuerte (p_good={p_good:.2f})"
                    metrics["display_label"] = display_label
                    reason = f"U3_GOOD_STRONG_NO_STRONG_DEFECT (p_good={p_good:.2f})"
                elif p_good >= 0.85 and strong_defect:
                    if decision == "RECHAZA":
                        decision = "REVISAR"
                        display_label = f"REVISAR - U3 GOOD vs defecto (p_good={p_good:.2f})"
                        metrics["display_label"] = display_label
                    reason = f"U3_GOOD_BUT_STRONG_DEFECT_REVIEW (p_good={p_good:.2f})"
                elif p_good >= 0.55:
                    if decision == "RECHAZA" and not strong_defect:
                        decision = "REVISAR"
                        display_label = f"REVISAR - U3 GOOD protege de RECHAZA (p_good={p_good:.2f})"
                        metrics["display_label"] = display_label
                        reason = f"U3_GOOD_WEAK_PROTECT_FROM_REJECT (p_good={p_good:.2f})"
                    elif decision == "PASA":
                        reason = f"U3_GOOD_WEAK_WAS_PASA (p_good={p_good:.2f})"
                    else:
                        reason = f"U3_GOOD_WEAK_REVISAR (p_good={p_good:.2f})"

            u3_metrics.update({
                "quality_u3_status": "OK" if u3_raw != "U3_ERROR" else "ERROR",
                "quality_u3_pred": "bad" if p_bad > p_good else "good",
                "quality_u3_p_good": round(p_good, 4),
                "quality_u3_p_bad": round(p_bad, 4),
                "quality_u3_decision_raw": u3_raw,
                "quality_u3_decision_safe": u3_safe,
                "final_decision_before_u3": decision_before_u3,
                "final_decision_after_u3": decision,
                "final_decision_reason": reason,
            })
            print(
                f"    U3: {u3_raw} safe={u3_safe}  p_bad={p_bad:.3f} p_good={p_good:.3f}"
                + (" -> " + decision if decision != decision_before_u3 else "")
            )

    u3_metrics["final_decision_after_u3"] = decision
    metrics.update(u3_metrics)

    result = {
        "decision": decision,
        "category": category,
        "estimated_category": estimated_category,
        "display_label": display_label,
        "mask_warning": mask_warning,
        "capture_valid": capture_info["capture_valid"],
        "capture_label": capture_info["capture_label"],
        "capture_reason": capture_reason,
        "orig_w": orig_w, "orig_h": orig_h,
        "proc_w": proc_w, "proc_h": proc_h,
        "metrics": metrics,
        "pear_mask": pear_mask,
        "body_mask": body_mask,
        "defect_mask": defect_mask,
        "defect_mask_raw": defect_mask_raw,
        "shadow_mask": shadow_mask,
        "rot_mask": rot_mask,
        "warning": warning,
        "crop_info": crop_info,
    }

    m = metrics
    warn_tag = " [!]" if (mask_warning or not capture_info["capture_valid"]) else ""
    print(
        f"  {img_path.name:<28} "
        f"{decision:<9}"
        f"{estimated_category:<16}"
        f"{display_label:<34} "
        f"{m['defect_pct']:5.1f}%"
        f"  {m['rot_pct']:5.1f}%"
        f"  {m['largest_defect_pct']:5.1f}%"
        f"  {m['pear_visible_pct']:5.1f}%"
        f"  {m['body_visible_pct']:5.1f}%"
        f"{warn_tag}"
    )
    if capture_reason:
        print(f"    CAPTURA: {capture_reason}")
    elif warning:
        print(f"    AVISO: {warning}")

    if args.save or args.show:
        vis = visualize(
            image_for_analysis, pear_mask, defect_mask, rot_mask,
            decision, category, metrics, warning=warning,
            body_mask=body_mask, shadow_mask=shadow_mask,
            capture_reason=capture_reason if capture_reason else None,
            yolo_valid_boxes=yolo_valid_boxes if defect_model is not None else None,
            yolo_ignored_boxes=yolo_ignored_boxes if defect_model is not None else None,
            yolo_metrics=yolo_defect_metrics if defect_model is not None else None,
        )
        if quality_cls_model is not None:
            pred_cls = quality_cls_metrics.get("quality_cls_pred", "unknown")
            max_c_cls = quality_cls_metrics.get("quality_cls_max_conf", 0.0)
            cls_txt = f"QUALITY CLS: {pred_cls.upper()} {max_c_cls:.2f}"
            cls_color = (0, 200, 0) if pred_cls == "good" else (0, 0, 220) if pred_cls == "bad" else (180, 180, 180)
            h_v, w_v = vis.shape[:2]
            y1_cls = max(20, h_v - 50)
            y2_cls = max(20, h_v - 25)
            cv2.putText(vis, cls_txt, (8, y1_cls), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, cls_txt, (8, y1_cls), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cls_color, 1, cv2.LINE_AA)
            if quality_cls_action_text:
                cv2.putText(vis, quality_cls_action_text, (8, y2_cls), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, quality_cls_action_text, (8, y2_cls), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)
        if u3_model is not None and u3_metrics.get("quality_u3_status") not in ("not_used",):
            h_v, w_v = vis.shape[:2]
            u3_raw_ = u3_metrics.get("quality_u3_decision_raw", "")
            u3_safe_ = u3_metrics.get("quality_u3_decision_safe", "")
            p_bad_ = u3_metrics.get("quality_u3_p_bad", 0.0)
            p_good_ = u3_metrics.get("quality_u3_p_good", 0.0)
            if u3_safe_ == "BAD":
                u3_color = (0, 0, 220)
                u3_txt = f"U3 BAD p_bad={p_bad_:.2f}"
            elif u3_safe_ == "REVIEW" and u3_raw_ == "U3_BAD":
                u3_color = (0, 200, 255)
                u3_txt = f"U3 BAD->REVIEW SAFE p_bad={p_bad_:.2f}"
            elif u3_safe_ == "REVIEW":
                u3_color = (0, 180, 255)
                u3_txt = f"U3 REVIEW p_bad={p_bad_:.2f}"
            else:
                u3_color = (0, 220, 0)
                u3_txt = f"U3 GOOD p_good={p_good_:.2f}"
            y_u3 = max(20, h_v - 75)
            cv2.putText(vis, u3_txt, (8, y_u3), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, u3_txt, (8, y_u3), cv2.FONT_HERSHEY_SIMPLEX, 0.50, u3_color, 1, cv2.LINE_AA)
        if args.show:
            cv2.imshow(img_path.name, vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.save:
            out_dir = PROJECT_ROOT / "outputs" / "quality_analysis"
            out_dir.mkdir(parents=True, exist_ok=True)
            imwrite_unicode(out_dir / img_path.name, vis)
            if yolo_model is not None and bbox_vis is not None:
                imwrite_unicode(out_dir / (img_path.stem + "_bbox.jpg"), bbox_vis)
            if yolo_model is not None and crop_image is not None:
                imwrite_unicode(out_dir / (img_path.stem + "_crop.jpg"), crop_image)
            if pear_mask is not None and pear_mask.any():
                mask_vis = cv2.cvtColor(pear_mask, cv2.COLOR_GRAY2BGR)
                imwrite_unicode(out_dir / (img_path.stem + "_mask.jpg"), mask_vis)

    if getattr(args, "debug", False):
        debug_dir = PROJECT_ROOT / "outputs" / "quality_analysis_debug" / img_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(debug_dir / "original.jpg", image_for_analysis)
        imwrite_unicode(debug_dir / "pear_mask.png", pear_mask)
        imwrite_unicode(debug_dir / "body_mask.png", body_mask)
        imwrite_unicode(debug_dir / "shadow_mask.png", shadow_mask)
        imwrite_unicode(debug_dir / "defect_mask_raw.png", defect_mask_raw)
        imwrite_unicode(debug_dir / "final_defect_mask.png", defect_mask)
        imwrite_unicode(debug_dir / "rot_mask.png", rot_mask)
        vis_dbg = visualize(
            image_for_analysis, pear_mask, defect_mask, rot_mask,
            decision, category, metrics, warning=warning,
            body_mask=body_mask, shadow_mask=shadow_mask,
            capture_reason=capture_reason if capture_reason else None,
            yolo_valid_boxes=yolo_valid_boxes if defect_model is not None else None,
            yolo_ignored_boxes=yolo_ignored_boxes if defect_model is not None else None,
            yolo_metrics=yolo_defect_metrics if defect_model is not None else None,
        )
        if quality_cls_model is not None:
            pred_cls_d = quality_cls_metrics.get("quality_cls_pred", "unknown")
            max_c_cls_d = quality_cls_metrics.get("quality_cls_max_conf", 0.0)
            cls_txt_d = f"QUALITY CLS: {pred_cls_d.upper()} {max_c_cls_d:.2f}"
            cls_color_d = (0, 200, 0) if pred_cls_d == "good" else (0, 0, 220) if pred_cls_d == "bad" else (180, 180, 180)
            h_vd, w_vd = vis_dbg.shape[:2]
            y1_d = max(20, h_vd - 50)
            y2_d = max(20, h_vd - 25)
            cv2.putText(vis_dbg, cls_txt_d, (8, y1_d), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_dbg, cls_txt_d, (8, y1_d), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cls_color_d, 1, cv2.LINE_AA)
            if quality_cls_action_text:
                cv2.putText(vis_dbg, quality_cls_action_text, (8, y2_d), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis_dbg, quality_cls_action_text, (8, y2_d), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)
        imwrite_unicode(debug_dir / "overlay.jpg", vis_dbg)
        if yolo_model is not None and bbox_vis is not None:
            imwrite_unicode(debug_dir / "bbox.jpg", bbox_vis)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — análisis de calidad visual rule-based v1"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Ruta a una imagen individual")
    group.add_argument("--source", type=Path, help="Carpeta con imágenes a analizar")

    parser.add_argument(
        "--seg-config",
        type=Path,
        default=Path("configs/thresholds.yaml"),
        help="Config de segmentación (default: configs/thresholds.yaml)",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("configs/quality_rules.yaml"),
        help="Umbrales de calidad (default: configs/quality_rules.yaml)",
    )
    parser.add_argument("--save", action="store_true",
                        help="Guardar visualizaciones en outputs/quality_analysis/")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar cada imagen en ventana cv2 (presiona tecla para avanzar)")
    parser.add_argument("--debug", action="store_true",
                        help="Guardar mascaras debug en outputs/quality_analysis_debug/")
    parser.add_argument("--max-size", type=int, default=1280,
                        help="Redimensionar imagen si ancho o alto supera este valor (0=sin limite, default=1280)")

    # Argumentos del detector YOLO
    parser.add_argument("--use-detector", action="store_true",
                        help="Usar YOLO para detectar y recortar la pera antes del análisis")
    parser.add_argument(
        "--detect-model",
        type=Path,
        default=Path("runs/detect/runs/pear_detector/eclpod_v1/weights/best.pt"),
        help="Ruta al modelo YOLO detector de pera",
    )
    parser.add_argument("--detect-conf", type=float, default=0.50,
                        help="Confianza mínima de detección YOLO (default: 0.50)")
    parser.add_argument("--crop-margin", type=float, default=0.08,
                        help="Margen alrededor de la bbox como porcentaje del tamaño de la bbox (default: 0.08)")

    # Argumentos del modelo auxiliar de defectos
    parser.add_argument("--use-defect-model", action="store_true",
                        help="Usar modelo YOLO de defectos como señal auxiliar sobre el crop")
    parser.add_argument(
        "--defect-model",
        type=Path,
        default=Path("runs/detect/runs/pear_defects/psd_smoke_v1/weights/best.pt"),
        help="Ruta al modelo YOLO detector de defectos PSD",
    )
    parser.add_argument("--defect-conf", type=float, default=0.25,
                        help="Confianza mínima para el detector de defectos (default: 0.25)")

    # Argumentos del clasificador auxiliar GOOD/BAD (Mendeley)
    parser.add_argument("--use-quality-cls", action="store_true",
                        help="Usar clasificador GOOD/BAD Mendeley como señal auxiliar")
    parser.add_argument(
        "--quality-cls-model",
        type=Path,
        default=Path("runs/pear_quality_cls/mendeley_good_bad_v1/weights/best.pt"),
        help="Ruta al modelo clasificador GOOD/BAD (default: mendeley_good_bad_v1/weights/best.pt)",
    )
    parser.add_argument("--quality-cls-bad-thr", type=float, default=0.85,
                        help="Confianza mínima BAD para afectar decisión (default: 0.85)")
    parser.add_argument("--quality-cls-affect-decision", action="store_true",
                        help="Si activo, PASA -> REVISAR cuando CLS dice BAD con confianza >= bad_thr")

    # Argumentos del clasificador U3 ROI masked clean (PyTorch)
    parser.add_argument("--use-quality-u3", action="store_true",
                        help="Usar clasificador U3 ROI masked clean sobre gray_bg input")
    parser.add_argument(
        "--quality-u3-model",
        type=Path,
        default=Path("outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt"),
        help="Ruta al modelo U3 (default: outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt)",
    )
    parser.add_argument(
        "--quality-u3-thresholds",
        type=Path,
        default=Path("outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json"),
        help="Ruta al JSON de umbrales U3",
    )
    parser.add_argument("--quality-u3-safe-mode", action="store_true", default=True,
                        help="Safe mode: U3=BAD solo produce RECHAZA si reglas/detector confirman defecto fuerte (default: True)")

    args = parser.parse_args()

    seg_config_path = (PROJECT_ROOT / args.seg_config).resolve()
    rules_path = (PROJECT_ROOT / args.rules).resolve()

    for path, label in [(seg_config_path, "--seg-config"), (rules_path, "--rules")]:
        if not path.exists():
            print(f"ERROR: {label} no encontrado: {path}")
            sys.exit(1)

    seg_data = _load_yaml(seg_config_path)
    seg_config = seg_data.get("segmentation", seg_data)
    rules = _load_yaml(rules_path)

    # Cargar modelo YOLO si se solicita
    yolo_model = None
    if args.use_detector:
        model_path = (PROJECT_ROOT / args.detect_model).resolve()
        if not model_path.exists():
            print(f"ERROR: modelo YOLO no encontrado: {model_path}")
            sys.exit(1)
        print(f"  Cargando detector YOLO: {model_path}")
        yolo_model = _load_yolo_model(model_path)
        if yolo_model is None:
            print("ERROR: no se pudo cargar el modelo YOLO")
            sys.exit(1)
        print(f"  Detector listo  conf>={args.detect_conf}  margen={args.crop_margin}")

    # Cargar clasificador auxiliar GOOD/BAD si se solicita (una sola vez)
    quality_cls_model = None
    if args.use_quality_cls:
        quality_cls_path = (PROJECT_ROOT / args.quality_cls_model).resolve()
        if not quality_cls_path.exists():
            print(f"ERROR: clasificador de calidad no encontrado: {quality_cls_path}")
            sys.exit(1)
        print(f"  Cargando clasificador de calidad: {quality_cls_path}")
        quality_cls_model = _load_yolo_model(quality_cls_path)
        if quality_cls_model is None:
            print("ERROR: no se pudo cargar el clasificador de calidad")
            sys.exit(1)
        print(
            f"  Clasificador calidad listo  "
            f"bad_thr={args.quality_cls_bad_thr}  "
            f"affect_decision={args.quality_cls_affect_decision}"
        )

    # Cargar modelo de defectos si se solicita (una sola vez)
    defect_model = None
    if args.use_defect_model:
        defect_model_path = (PROJECT_ROOT / args.defect_model).resolve()
        if not defect_model_path.exists():
            print(f"ERROR: modelo de defectos no encontrado: {defect_model_path}")
            sys.exit(1)
        print(f"  Cargando modelo de defectos: {defect_model_path}")
        defect_model = _load_yolo_model(defect_model_path)
        if defect_model is None:
            print("ERROR: no se pudo cargar el modelo de defectos")
            sys.exit(1)
        print(f"  Modelo defectos listo  conf>={args.defect_conf}")

    # Cargar clasificador U3 si se solicita
    u3_model = None
    u3_thresholds = None
    if getattr(args, "use_quality_u3", False):
        u3_model_path = (PROJECT_ROOT / args.quality_u3_model).resolve()
        u3_thr_path   = (PROJECT_ROOT / args.quality_u3_thresholds).resolve()
        for p, lbl in [(u3_model_path, "--quality-u3-model"), (u3_thr_path, "--quality-u3-thresholds")]:
            if not p.exists():
                print(f"ERROR: U3 {lbl} no encontrado: {p}")
                sys.exit(1)
        print(f"  Cargando clasificador U3: {u3_model_path}")
        u3_model, u3_thresholds = _load_u3_classifier(u3_model_path, u3_thr_path)
        if u3_model is None:
            print("ERROR: no se pudo cargar el clasificador U3")
            sys.exit(1)
        u3_safe = getattr(args, "quality_u3_safe_mode", True)
        print(f"  U3 listo  safe_mode={u3_safe}  bad_thr={u3_thresholds.get('bad_reject_threshold')}")

    source = (PROJECT_ROOT / (args.image or args.source)).resolve()
    images = _collect_images(source)

    if not images:
        print(f"ERROR: no se encontraron imágenes en {source}")
        sys.exit(1)

    sep = "=" * 112
    print(sep)
    print("PearVision QC — Motor de calidad rule-based v3")
    print(sep)
    print(f"  Fuente     : {source}")
    print(f"  Reglas     : {rules_path}")
    print(f"  Imagenes   : {len(images)}")
    print(f"  Detector   : {'YOLO activo' if yolo_model is not None else 'desactivado'}")
    print(f"  Defectos   : {'YOLO PSD activo' if defect_model is not None else 'desactivado'}")
    print(f"  Calidad CLS: {'activo' if quality_cls_model is not None else 'desactivado'}")
    print()
    print(
        f"  {'IMAGEN':<28}"
        f"{'DECISION':<9}"
        f"{'CAT.ESTIMADA':<16}"
        f"{'DISPLAY_LABEL':<34} "
        f"{'DEF%':>6}"
        f"  {'ROT%':>6}"
        f"  {'MAX%':>6}"
        f"  {'PERA%':>6}"
        f"  {'CUERPO%':>7}"
    )
    print("  " + "-" * 110)

    counts = {"PASA": 0, "REVISAR": 0, "RECHAZA": 0}
    all_results = []
    for img_path in images:
        result = _process_one(
            img_path, seg_config, rules, args,
            yolo_model=yolo_model,
            defect_model=defect_model,
            defect_conf=args.defect_conf if args.use_defect_model else 0.25,
            quality_cls_model=quality_cls_model,
            quality_cls_bad_thr=args.quality_cls_bad_thr if args.use_quality_cls else 0.85,
            quality_cls_affect_decision=args.quality_cls_affect_decision if args.use_quality_cls else False,
            u3_model=u3_model,
            u3_thresholds=u3_thresholds,
            u3_safe_mode=getattr(args, "quality_u3_safe_mode", True),
        )
        result["image"] = img_path.name
        counts[result["decision"]] = counts.get(result["decision"], 0) + 1
        all_results.append(result)

    print("  " + "-" * 110)
    total = len(images)
    print(
        f"  TOTAL {total}  ->  "
        f"PASA: {counts['PASA']}  "
        f"REVISAR: {counts['REVISAR']}  "
        f"RECHAZA: {counts['RECHAZA']}"
    )

    if args.save:
        import csv
        out_dir = PROJECT_ROOT / "outputs" / "quality_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "resultados_calidad.csv"
        fieldnames = [
            "image", "decision", "estimated_category", "display_label",
            "capture_valid", "capture_label", "capture_reason",
            "defect_pct", "dark_rot_pct", "max_region_pct",
            "pear_visible_pct", "body_visible_pct",
            "mask_area_pct", "body_area_pct",
            "bbox_fill_ratio", "bbox_aspect_ratio",
            "mask_components", "border_touch_pct", "mask_irregularity_ratio",
            "mask_warning",
            "original_width", "original_height",
            "processed_width", "processed_height",
            # Columnas del detector YOLO de pera
            "detector_used", "detector_conf",
            "crop_x1", "crop_y1", "crop_x2", "crop_y2",
            "crop_margin",
            # Columnas auxiliares del modelo de defectos PSD
            "yolo_defect_count", "yolo_defect_area_pct",
            "yolo_defect_max_conf", "yolo_defect_classes",
            "brown_dark_pct", "dark_area_pct",
            # Columnas de calidad de mascara ROI
            "mask_source", "mask_quality_ok", "mask_fail_reason",
            # Columnas del clasificador auxiliar GOOD/BAD
            "quality_cls_used", "quality_cls_source", "quality_cls_pred",
            "quality_cls_good_conf", "quality_cls_bad_conf", "quality_cls_max_conf",
            "quality_cls_action",
            # Brillo medio del cuerpo (LAB L) para auditar imágenes oscuras
            "body_l_mean",
            # Columnas del clasificador U3 (ROI masked clean)
            "quality_u3_enabled",
            "quality_u3_status",
            "quality_u3_pred",
            "quality_u3_p_good",
            "quality_u3_p_bad",
            "quality_u3_decision_raw",
            "quality_u3_decision_safe",
            "final_decision_before_u3",
            "final_decision_after_u3",
            "final_decision_reason",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                m = r["metrics"]
                ci = r.get("crop_info", {})
                writer.writerow({
                    "image": r["image"],
                    "decision": r["decision"],
                    "estimated_category": r["estimated_category"],
                    "display_label": r["display_label"],
                    "capture_valid": r["capture_valid"],
                    "capture_label": r["capture_label"],
                    "capture_reason": r["capture_reason"],
                    "defect_pct": m["defect_pct"],
                    "dark_rot_pct": m["rot_pct"],
                    "max_region_pct": m["largest_defect_pct"],
                    "pear_visible_pct": m["pear_visible_pct"],
                    "body_visible_pct": m["body_visible_pct"],
                    "mask_area_pct": m.get("mask_area_pct", 0.0),
                    "body_area_pct": m.get("body_area_pct", 0.0),
                    "bbox_fill_ratio": m.get("bbox_fill_ratio", 0.0),
                    "bbox_aspect_ratio": m.get("bbox_aspect_ratio", 0.0),
                    "mask_components": m.get("mask_components", 0),
                    "border_touch_pct": m.get("border_touch_pct", 0.0),
                    "mask_irregularity_ratio": m.get("mask_irregularity_ratio", 0.0),
                    "mask_warning": r["mask_warning"],
                    "original_width": r["orig_w"],
                    "original_height": r["orig_h"],
                    "processed_width": r["proc_w"],
                    "processed_height": r["proc_h"],
                    "detector_used": ci.get("detector_used", False),
                    "detector_conf": ci.get("detector_conf", 0.0),
                    "crop_x1": ci.get("crop_x1", 0),
                    "crop_y1": ci.get("crop_y1", 0),
                    "crop_x2": ci.get("crop_x2", 0),
                    "crop_y2": ci.get("crop_y2", 0),
                    "crop_margin": ci.get("crop_margin", 0.0),
                    "yolo_defect_count": m.get("yolo_defect_count", 0),
                    "yolo_defect_area_pct": m.get("yolo_defect_area_pct", 0.0),
                    "yolo_defect_max_conf": m.get("yolo_defect_max_conf", 0.0),
                    "yolo_defect_classes": m.get("yolo_defect_classes", ""),
                    "brown_dark_pct": m.get("brown_dark_pct", 0.0),
                    "dark_area_pct": m.get("dark_area_pct", 0.0),
                    "mask_source": ci.get("mask_source", "classic"),
                    "mask_quality_ok": ci.get("mask_quality_ok", True),
                    "mask_fail_reason": ci.get("mask_fail_reason", ""),
                    "quality_cls_used": m.get("quality_cls_used", False),
                    "quality_cls_source": m.get("quality_cls_source", "not_used"),
                    "quality_cls_pred": m.get("quality_cls_pred", "unknown"),
                    "quality_cls_good_conf": m.get("quality_cls_good_conf", 0.0),
                    "quality_cls_bad_conf": m.get("quality_cls_bad_conf", 0.0),
                    "quality_cls_max_conf": m.get("quality_cls_max_conf", 0.0),
                    "quality_cls_action": m.get("quality_cls_action", ""),
                    "body_l_mean": m.get("body_l_mean", 128.0),
                    "quality_u3_enabled": m.get("quality_u3_enabled", False),
                    "quality_u3_status": m.get("quality_u3_status", "not_used"),
                    "quality_u3_pred": m.get("quality_u3_pred", "unknown"),
                    "quality_u3_p_good": m.get("quality_u3_p_good", 0.0),
                    "quality_u3_p_bad": m.get("quality_u3_p_bad", 0.0),
                    "quality_u3_decision_raw": m.get("quality_u3_decision_raw", ""),
                    "quality_u3_decision_safe": m.get("quality_u3_decision_safe", ""),
                    "final_decision_before_u3": m.get("final_decision_before_u3", ""),
                    "final_decision_after_u3": m.get("final_decision_after_u3", ""),
                    "final_decision_reason": m.get("final_decision_reason", ""),
                })
        print(f"\n  Visualizaciones : {out_dir}")
        print(f"  CSV resultados  : {csv_path}")


if __name__ == "__main__":
    main()
