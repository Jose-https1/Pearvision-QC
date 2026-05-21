#!/usr/bin/env python3
"""
PearVision QC - Real-Time Camera App Pro V6

Cambios respecto a V5:
  - LIVE_GOOD_ACCEPT_THRESHOLD = 0.60 (baja de 0.70 para mayor tolerancia real)
  - BAD_REJECT_THRESHOLD = 0.995 (igual que siempre)
  - Bloqueo duro ANTES de U3: is_valid_live_pear_candidate()
      * border_cut=YES -> SIN PERA / MALA CAPTURA, nunca REVISAR
      * bbox too large / too small / bad shape -> SIN PERA / MALA CAPTURA
      * U3 NO se ejecuta sobre persona/fondo/camiseta
  - apply_policy_v6: ya no recibe border_cut (bloqueado upstream)
  - Salida en outputs/live_camera_qc_pro_v6/

Controles:
    Q / ESC : salir
    S       : guardar evidencia
    B       : calibrar fondo
    C       : limpiar fondo calibrado
    P       : pausar / reanudar
    R       : resetear smoothing
    H       : mostrar/ocultar ayuda
    M       : mostrar/ocultar miniaturas

Uso:
    .venv\\Scripts\\python.exe scripts\\pearvision_qc_realtime_camera_pro_v6.py --camera 0 --infer-every 5
    python scripts/pearvision_qc_realtime_camera_pro_v6.py --image-folder RUTA
"""

import argparse
import csv
import datetime
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

try:
    from PIL import Image
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as tT
    _TORCH_OK = True
except ImportError as _e:
    print(f"[WARN] PyTorch/PIL no disponible: {_e}")
    _TORCH_OK = False
    Image = None  # type: ignore

# ── Layout canvas 1600x900 ────────────────────────────────────────────────────
CW, CH     = 1600, 900
HEADER_H   = 55
CAM_W      = 960
PANEL_W    = CW - CAM_W
THUMB_H    = 180
CAM_ZONE_H = CH - HEADER_H - THUMB_H
RESULT_H   = 290
TECH_H     = CH - HEADER_H - RESULT_H

# ── Colores BGR ───────────────────────────────────────────────────────────────
BG        = (15,  15,  20)
PANEL_COL = (25,  28,  35)
BORDER    = (55,  58,  65)
WHITE     = (220, 225, 235)
GRAY      = (120, 125, 135)
ACCENT    = (180, 160,  70)

PASA_C    = ( 60, 200,  70)
REVISAR_C = (  0, 145, 255)
RECHAZA_C = ( 45,  45, 220)
SINPERA_C = (180, 130,  70)
MALCAP_C  = ( 20, 120, 230)
ERROR_C   = (140,  80, 180)

DECISION_COLORS: dict = {
    "PASA":          PASA_C,
    "REVISAR":       REVISAR_C,
    "RECHAZA":       RECHAZA_C,
    "SIN PERA":      SINPERA_C,
    "MALA CAPTURA":  MALCAP_C,
    "ERROR":         ERROR_C,
}

# ── Politica V6 (camara real) ─────────────────────────────────────────────────
LIVE_GOOD_ACCEPT_THRESHOLD = 0.60   # bajado de 0.70 para tolerar angulos reales
BAD_REJECT_THRESHOLD       = 0.995  # estricto: solo RECHAZA con certeza alta

# ── Gating candidatos ─────────────────────────────────────────────────────────
MIN_AREA_RATIO      = 0.01
MAX_AREA_RATIO      = 0.45
MAX_BBOX_W_RATIO    = 0.80
MAX_BBOX_H_RATIO    = 0.80
MAX_RECTANGULARITY  = 0.93
MAX_BORDER_TOUCHES  = 3
MIN_CANDIDATE_SAT   = 18    # saturacion HSV media; menor => probable fondo

# ── Gating pre-U3 (bloqueo duro antes de inferencia) ─────────────────────────
PRE_U3_MIN_AREA     = 0.004   # pear_area_ratio minimo para ejecutar U3
PRE_U3_MAX_AREA     = 0.45    # pear_area_ratio maximo para ejecutar U3
PRE_U3_MAX_BW       = 0.80
PRE_U3_MAX_BH       = 0.80
PRE_U3_MAX_RECT     = 0.93    # rectangularity maxima
PRE_U3_MAX_RECT_AREA= 0.10    # si rect > 0.93 y area > 10% -> fondo rectangular


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA U3
# ══════════════════════════════════════════════════════════════════════════════

def load_u3(model_path: Path, thr_path: Path):
    if not _TORCH_OK:
        return None, {}
    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return None, {}
    try:
        model = tvm.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, 2)
        model.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True))
        model.eval()
        thr: dict = {}
        if thr_path.exists():
            with open(str(thr_path), encoding="utf-8") as fh:
                thr = json.load(fh)
        print(f"[OK] U3 cargado: {model_path.name}")
        return model, thr
    except Exception as exc:
        print(f"[ERROR] Cargando U3: {exc}")
        return None, {}


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTACION V6 (igual que V5)
# ══════════════════════════════════════════════════════════════════════════════

def _seg_by_saturation(frame_bgr: np.ndarray,
                       sat_thr: int = 30,
                       val_min: int = 35,
                       val_max: int = 242) -> np.ndarray:
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, sat_thr, val_min), (180, 255, val_max))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    return mask


def _seg_by_pear_color(frame_bgr: np.ndarray) -> np.ndarray:
    hsv    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ranges = [
        ((25,  30,  50), ( 90, 255, 255)),
        ((15,  40, 100), ( 30, 255, 255)),
        (( 5,  20,  40), ( 25, 210, 200)),
        ((10,  50, 100), ( 22, 255, 255)),
        ((40,  15, 100), (100, 100, 255)),
    ]
    mask = np.zeros(frame_bgr.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    return mask


def _seg_with_background(frame_bgr: np.ndarray,
                          bg_bgr: np.ndarray) -> np.ndarray:
    h, w   = frame_bgr.shape[:2]
    bg_res = cv2.resize(bg_bgr, (w, h))
    lab_f  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b  = cv2.cvtColor(bg_res,    cv2.COLOR_BGR2LAB).astype(np.float32)
    diff   = np.linalg.norm(lab_f - lab_b, axis=2)
    mask   = (diff > 16).astype(np.uint8) * 255
    hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    bg_gray = cv2.cvtColor(bg_res,    cv2.COLOR_BGR2GRAY).astype(np.float32)
    fr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shadow  = (hsv[:, :, 1] < 22) & (fr_gray < bg_gray - 10)
    mask[shadow] = 0
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    return mask


def _is_candidate_valid_v6(contour, frame_bgr: np.ndarray) -> tuple:
    h, w  = frame_bgr.shape[:2]
    area  = cv2.contourArea(contour)
    ratio = area / (w * h)

    if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
        return False, {}

    x, y, bw, bh = cv2.boundingRect(contour)
    bw_r = bw / w
    bh_r = bh / h

    if bw_r > MAX_BBOX_W_RATIO or bh_r > MAX_BBOX_H_RATIO:
        return False, {}

    hull      = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    solidity  = area / hull_area
    rect_val  = area / max(bw * bh, 1)
    aspect    = bw / max(bh, 1)

    if rect_val > MAX_RECTANGULARITY and ratio > 0.12:
        return False, {}

    touches = sum([x <= 2, y <= 2,
                   x + bw >= w - 2,
                   y + bh >= h - 2])
    if touches >= MAX_BORDER_TOUCHES:
        return False, {}

    hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    roi_sat = float(np.mean(hsv[y:y + bh, x:x + bw, 1]))
    if roi_sat < MIN_CANDIDATE_SAT:
        return False, {}

    perimeter = cv2.arcLength(contour, True)
    compact   = (4 * np.pi * area / max(perimeter ** 2, 1))

    metrics = dict(
        area=area, ratio=ratio,
        bw_r=bw_r, bh_r=bh_r,
        bbox=(x, y, bw, bh),
        solidity=solidity,
        rectangularity=rect_val,
        aspect=aspect,
        touches=touches,
        compactness=compact,
        roi_sat=roi_sat,
    )
    return True, metrics


def _score_candidate_v6(metrics: dict, frame_bgr: np.ndarray) -> float:
    h, w = frame_bgr.shape[:2]
    ratio    = metrics["ratio"]
    solidity = metrics["solidity"]
    rect_val = metrics["rectangularity"]
    compact  = metrics["compactness"]
    aspect   = metrics["aspect"]
    touches  = metrics["touches"]
    roi_sat  = metrics["roi_sat"]
    x, y, bw, bh = metrics["bbox"]

    ideal_area  = 0.15
    area_score  = max(0.0, 1.0 - abs(ratio - ideal_area) * 4.0)

    cx = x + bw / 2; cy = y + bh / 2
    center_dist  = np.sqrt(((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2)
    center_score = max(0.0, 1.0 - center_dist * 2.0)

    shape_score   = solidity * max(0.0, 1.0 - max(0, rect_val - 0.75) * 2)
    compact_score = max(0.0, 1.0 - abs(compact - 0.70) * 2.0)
    aspect_score  = max(0.0, 1.0 - max(0, abs(aspect - 0.9) - 0.7) * 1.5)
    sat_score     = min(1.0, roi_sat / 70.0)
    border_pen    = touches * 0.25

    return (area_score    * 0.25 +
            center_score  * 0.15 +
            shape_score   * 0.25 +
            compact_score * 0.10 +
            aspect_score  * 0.10 +
            sat_score     * 0.15 -
            border_pen)


def _collect_candidates_v6(mask: np.ndarray, frame_bgr: np.ndarray) -> list:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        valid, metrics = _is_candidate_valid_v6(c, frame_bgr)
        if valid:
            score = _score_candidate_v6(metrics, frame_bgr)
            results.append((c, metrics, score))
    return results


def detect_pear_v6(frame_bgr: np.ndarray, bg_frame=None) -> dict:
    h, w = frame_bgr.shape[:2]

    _no_pear = dict(
        mask=np.zeros((h, w), np.uint8),
        contour=None, bbox=None,
        pear_area_ratio=0.0,
        cap_status="SIN_PERA", mask_valid=False,
        border_cut=False, notes="Sin candidatos validos",
        metrics={}, strategy_used="none")

    mask_sat   = _seg_by_saturation(frame_bgr)
    mask_color = _seg_by_pear_color(frame_bgr)

    all_candidates = []
    all_candidates += [(c, m, s, "sat")
                       for c, m, s in _collect_candidates_v6(mask_sat, frame_bgr)]
    all_candidates += [(c, m, s, "color")
                       for c, m, s in _collect_candidates_v6(mask_color, frame_bgr)]

    if bg_frame is not None:
        mask_bg = _seg_with_background(frame_bgr, bg_frame)
        all_candidates += [(c, m, s, "bg_diff")
                           for c, m, s in _collect_candidates_v6(mask_bg, frame_bgr)]

    if not all_candidates:
        combined = cv2.bitwise_or(mask_sat, mask_color)
        _no_pear["mask"] = combined
        return _no_pear

    best = max(all_candidates, key=lambda t: t[2])
    best_contour, best_metrics, best_score, best_strategy = best

    if best_score < 0:
        combined = cv2.bitwise_or(mask_sat, mask_color)
        _no_pear["mask"] = combined
        _no_pear["notes"] = f"Score negativo ({best_score:.2f})"
        return _no_pear

    best_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(best_mask, [best_contour], -1, 255, -1)

    bbox       = best_metrics["bbox"]
    ratio      = best_metrics["ratio"]
    touches    = best_metrics["touches"]
    border_cut = touches >= 1
    notes      = (f"strategy={best_strategy} score={best_score:.2f}"
                  f" cands={len(all_candidates)}")

    return dict(
        mask=best_mask,
        contour=best_contour,
        bbox=bbox,
        pear_area_ratio=ratio,
        cap_status="OK",
        mask_valid=True,
        border_cut=border_cut,
        notes=notes,
        metrics=best_metrics,
        strategy_used=best_strategy)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOQUEO DURO PRE-U3 (V6 nueva funcion)
# ══════════════════════════════════════════════════════════════════════════════

def is_valid_live_pear_candidate(det: dict, frame_bgr: np.ndarray) -> tuple:
    """
    Bloqueo duro antes de ejecutar U3.
    Devuelve (valid: bool, reason: str).
    Si valid=False, no se ejecuta U3 y se emite SIN PERA o MALA CAPTURA.

    Reglas V6:
    - mask_valid=False           -> bloquear
    - border_cut=True            -> bloquear (persona/fondo suele tocar borde)
    - bbox ausente               -> bloquear
    - pear_area_ratio > 0.45     -> demasiado grande, no es pera aislada
    - pear_area_ratio < 0.004    -> ruido/particula minima
    - bbox_w_ratio > 0.80        -> ocupa casi toda la pantalla
    - bbox_h_ratio > 0.80        -> ocupa casi toda la pantalla
    - rectangularity > 0.93      -> forma de fondo rectangular
    - candidato en esquina       -> centro del bbox muy cerca de una esquina
    """
    if not det.get("mask_valid", False):
        return False, "NO_VALID_MASK"

    bbox = det.get("bbox")
    if bbox is None:
        return False, "NO_BBOX"

    # border_cut: toca al menos un borde -> probable fondo/persona/ropa
    if det.get("border_cut", False):
        return False, "BORDER_CUT_BLOCKED"

    metrics = det.get("metrics", {})
    ratio   = det.get("pear_area_ratio", 0.0)
    bw_r    = metrics.get("bw_r", 0.0)
    bh_r    = metrics.get("bh_r", 0.0)
    rect    = metrics.get("rectangularity", 0.0)

    if ratio > PRE_U3_MAX_AREA:
        return False, f"AREA_TOO_LARGE ratio={ratio:.3f}"

    if ratio < PRE_U3_MIN_AREA:
        return False, f"AREA_TOO_SMALL ratio={ratio:.3f}"

    if bw_r > PRE_U3_MAX_BW:
        return False, f"BBOX_W_TOO_WIDE bw_r={bw_r:.3f}"

    if bh_r > PRE_U3_MAX_BH:
        return False, f"BBOX_H_TOO_TALL bh_r={bh_r:.3f}"

    if rect > PRE_U3_MAX_RECT and ratio > PRE_U3_MAX_RECT_AREA:
        return False, f"RECTANGULARITY_TOO_HIGH rect={rect:.3f}"

    # Candidato en esquina: centro del bbox muy pegado a las 4 esquinas
    if frame_bgr is not None:
        fh, fw = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        cx = (x + bw / 2) / fw
        cy = (y + bh / 2) / fh
        corner_dist = min(
            np.sqrt(cx ** 2 + cy ** 2),
            np.sqrt((1 - cx) ** 2 + cy ** 2),
            np.sqrt(cx ** 2 + (1 - cy) ** 2),
            np.sqrt((1 - cx) ** 2 + (1 - cy) ** 2),
        )
        if corner_dist < 0.12:
            return False, f"CANDIDATE_IN_CORNER dist={corner_dist:.3f}"

    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESADO PARA U3 (sin cambios respecto a V5)
# ══════════════════════════════════════════════════════════════════════════════

def make_gray_bg_clean(frame_bgr: np.ndarray, size: int = 224):
    img = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    cs  = 12
    corners = np.vstack([
        img[:cs, :cs].reshape(-1, 3), img[:cs, -cs:].reshape(-1, 3),
        img[-cs:, :cs].reshape(-1, 3), img[-cs:, -cs:].reshape(-1, 3),
    ])
    bg_bgr = np.median(corners, axis=0).astype(np.uint8)
    lab    = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_bgr.reshape(1, 1, 3),
                           cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist   = np.linalg.norm(lab - bg_lab, axis=2)
    mask   = (dist > 25).astype(np.uint8) * 255
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb[mask == 0] = [128, 128, 128]
    pil = Image.fromarray(rgb) if _TORCH_OK else None
    return pil, mask, bg_bgr


def make_u3_input(frame_bgr: np.ndarray, bbox=None, size: int = 224):
    if bbox is not None:
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        margin = max(int(max(bw, bh) * 0.25), 15)
        x1 = max(0, x - margin);      y1 = max(0, y - margin)
        x2 = min(w, x + bw + margin); y2 = min(h, y + bh + margin)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
            return make_gray_bg_clean(crop, size)
    return make_gray_bg_clean(frame_bgr, size)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCIA U3
# ══════════════════════════════════════════════════════════════════════════════

def run_u3(model, gray_pil):
    if model is None or gray_pil is None:
        return "ERROR", 0.0, 0.0
    try:
        _tf = tT.Compose([
            tT.Resize((224, 224)),
            tT.ToTensor(),
            tT.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            probs = torch.softmax(
                model(_tf(gray_pil).unsqueeze(0)), dim=1)[0].cpu().numpy()
        p_bad, p_good = float(probs[0]), float(probs[1])
        return ("BAD" if p_bad > p_good else "GOOD"), p_good, p_bad
    except Exception as exc:
        print(f"[WARN] Error inferencia U3: {exc}")
        return "ERROR", 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  POLITICA DE DECISION V6
# ══════════════════════════════════════════════════════════════════════════════

def apply_policy_v6(u3_pred: str, p_good: float, p_bad: float) -> tuple:
    """
    Politica V6 — solo llamada cuando is_valid_live_pear_candidate() devuelve True.
    border_cut ya esta bloqueado upstream: nunca llega aqui.
    SIN PERA y MALA CAPTURA se gestionan fuera de esta funcion.
    """
    if u3_pred == "ERROR":
        return "REVISAR", "U3_INFERENCE_ERROR"
    if u3_pred == "GOOD" and p_good >= LIVE_GOOD_ACCEPT_THRESHOLD:
        return "PASA", f"U3=GOOD p_good={p_good:.4f} >= {LIVE_GOOD_ACCEPT_THRESHOLD}"
    if u3_pred == "BAD" and p_bad >= BAD_REJECT_THRESHOLD:
        return "RECHAZA", f"U3=BAD p_bad={p_bad:.4f} >= {BAD_REJECT_THRESHOLD}"
    if u3_pred == "BAD":
        return "REVISAR", f"U3=BAD p_bad={p_bad:.4f} < {BAD_REJECT_THRESHOLD}"
    return "REVISAR", f"p_good={p_good:.4f} < {LIVE_GOOD_ACCEPT_THRESHOLD}"


# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTHING BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class SmoothingBuffer:
    def __init__(self, window: int = 7) -> None:
        self.window = window
        self._buf: deque = deque(maxlen=window)

    def add(self, decision: str) -> None:
        if decision not in ("SIN PERA", "MALA CAPTURA"):
            self._buf.append(decision)

    def stable(self) -> str:
        if not self._buf:
            return "SIN PERA"
        counts  = Counter(self._buf)
        majority, cnt = counts.most_common(1)[0]
        if majority == "RECHAZA" and cnt < max(3, self.window // 2):
            return "REVISAR"
        if majority == "PASA" and cnt < max(2, self.window // 3):
            return "REVISAR"
        return majority

    def reset(self) -> None:
        self._buf.clear()

    def count(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE DIBUJO
# ══════════════════════════════════════════════════════════════════════════════

def _pt(img, text: str, x: int, y: int,
        scale: float = 0.45, color: tuple = WHITE,
        thickness: int = 1, bold: bool = False) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _panel(img, x1, y1, x2, y2,
           fill: tuple = PANEL_COL, border: tuple = BORDER) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)


def _make_canvas() -> np.ndarray:
    return np.full((CH, CW, 3), BG, dtype=np.uint8)


# ── Header ────────────────────────────────────────────────────────────────────

def draw_header(canvas, fps: float, cam_idx: int,
                fw: int, fh: int, frame_id: int,
                show_help: bool, bg_calibrated: bool) -> None:
    _panel(canvas, 0, 0, CW, HEADER_H, (18, 22, 32), BORDER)
    _pt(canvas, "PearVision QC - Real-Time Inspection V6", 12, 34,
        0.7, (190, 210, 255), 1, True)

    bg_tag = "[BG:OK]" if bg_calibrated else "[BG:OFF]"
    bg_col = PASA_C if bg_calibrated else GRAY
    _pt(canvas, bg_tag, CW - 360, 34, 0.46, bg_col, 1, True)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = (f"{now_str}   FPS:{fps:5.1f}   CAM {cam_idx}   "
            f"{fw}x{fh}   FRAME {frame_id:06d}   thr_good={LIVE_GOOD_ACCEPT_THRESHOLD}")
    tw, _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    _pt(canvas, info, CW - tw[0] - 370, 32, 0.38, GRAY)

    if show_help:
        _pt(canvas,
            "Q/ESC:Salir  S:Guardar  B:Cal.fondo  C:Limpiar  P:Pausa  R:Reset  H:Ayuda  M:Minis",
            14, 50, 0.30, (100, 140, 180))


# ── Zona camara con mascara semitransparente ──────────────────────────────────

def draw_camera_zone(canvas, frame_bgr, mask, contour, bbox, decision: str):
    zx, zy = 0, HEADER_H
    zw, zh = CAM_W, CAM_ZONE_H
    h, w   = frame_bgr.shape[:2]
    scale  = min(zw / w, zh / h)
    dw, dh = int(w * scale), int(h * scale)
    ox     = zx + (zw - dw) // 2
    oy     = zy + (zh - dh) // 2

    resized = cv2.resize(frame_bgr, (dw, dh))
    canvas[oy:oy + dh, ox:ox + dw] = resized

    color = DECISION_COLORS.get(decision, GRAY)

    if mask is not None and np.any(mask > 0):
        mask_disp = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        zone = canvas[oy:oy + dh, ox:ox + dw].copy()
        zone[mask_disp > 0] = np.clip(
            zone[mask_disp > 0].astype(np.int32) * 0.65 +
            np.array(color, dtype=np.int32) * 0.35, 0, 255).astype(np.uint8)
        canvas[oy:oy + dh, ox:ox + dw] = zone

    cv2.rectangle(canvas, (ox, oy), (ox + dw - 1, oy + dh - 1), color, 3)

    if contour is not None:
        dc = contour.copy().astype(np.float32)
        dc[:, :, 0] = dc[:, :, 0] * scale + ox
        dc[:, :, 1] = dc[:, :, 1] * scale + oy
        cv2.drawContours(canvas, [dc.astype(np.int32)], -1, color, 2)

    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(canvas,
                      (int(bx * scale + ox), int(by * scale + oy)),
                      (int((bx + bw) * scale + ox), int((by + bh) * scale + oy)),
                      color, 1)

    if contour is not None:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"] * scale + ox)
            cy = int(M["m01"] / M["m00"] * scale + oy)
            cv2.circle(canvas, (cx, cy), 5, color, -1)
            cv2.circle(canvas, (cx, cy), 5, WHITE, 1)

    return ox, oy, dw, dh


# ── Panel tecnico V6 ──────────────────────────────────────────────────────────

def draw_tech_panel(canvas, state: dict) -> None:
    px, py = CAM_W, HEADER_H
    _panel(canvas, px, py, CW, py + TECH_H)
    _pt(canvas, "DATOS TECNICOS", px + 12, py + 20, 0.48, ACCENT, 1, True)
    cv2.line(canvas, (px + 8, py + 28), (CW - 8, py + 28), BORDER, 1)

    no_pear = state.get("no_pear_mode", False)

    def _col(k, v):
        if k in ("instant_dec", "stable_dec"):
            return DECISION_COLORS.get(str(v), WHITE)
        if k == "u3_pred":
            return RECHAZA_C if v == "BAD" else PASA_C if v == "GOOD" else GRAY
        if k == "capture_status":
            return PASA_C if v == "OK" else SINPERA_C if "SIN" in str(v) else MALCAP_C
        if k in ("mask_valid", "bg_calibrated"):
            return PASA_C if v == "YES" else REVISAR_C
        if k == "border_cut":
            return REVISAR_C if v == "YES" else PASA_C
        if k == "u3_blocked":
            return MALCAP_C if v == "YES" else PASA_C
        return WHITE

    p_good = state.get("p_good", 0.0)
    p_bad  = state.get("p_bad",  0.0)
    p_g_s  = "N/A" if no_pear else f"{p_good:.4f}"
    p_b_s  = "N/A" if no_pear else f"{p_bad:.4f}"
    u3_s   = "N/A" if no_pear else state.get("u3_pred", "--")

    rows = [
        ("capture_status",  state.get("capture_status",  "--")),
        ("mask_valid",      "YES" if state.get("mask_valid", False) else "NO"),
        ("bg_calibrated",   "YES" if state.get("bg_calibrated", False) else "NO"),
        ("strategy",        state.get("strategy_used", "--")),
        ("instant_dec",     state.get("instant_decision", "--")),
        ("stable_dec",      state.get("stable_decision",  "--")),
        ("smoothing",       f"{state.get('smoothing_count',0)}/{state.get('smoothing_window',7)}"),
        ("u3_blocked",      "YES" if state.get("u3_blocked", False) else "NO"),
        ("---", ""),
        ("u3_pred",         u3_s),
        ("p_good",          p_g_s),
        ("p_bad",           p_b_s),
        ("thr_live_good",   f"{LIVE_GOOD_ACCEPT_THRESHOLD}"),
        ("thr_bad_rej",     f"{BAD_REJECT_THRESHOLD}"),
        ("---", ""),
        ("pear_area_ratio", f"{state.get('pear_area_ratio', 0.0):.4f}"),
        ("bbox_w_ratio",    f"{state.get('bbox_w_ratio', 0.0):.3f}"),
        ("bbox_h_ratio",    f"{state.get('bbox_h_ratio', 0.0):.3f}"),
        ("rectangularity",  f"{state.get('rectangularity', 0.0):.3f}"),
        ("solidity",        f"{state.get('solidity', 0.0):.3f}"),
        ("border_cut",      "YES" if state.get("border_cut", False) else "NO"),
        ("bbox",            state.get("bbox_str", "--")[:24]),
        ("---", ""),
        ("preproc_ms",      f"{state.get('preproc_ms', 0.0):.1f}"),
        ("infer_ms",        f"{state.get('infer_ms',   0.0):.1f}"),
        ("total_ms",        f"{state.get('total_ms',   0.0):.1f}"),
        ("---", ""),
        ("saved_count",     str(state.get("saved_count", 0))),
        ("last_saved",      state.get("last_saved", "--")[:24]),
        ("reason",          state.get("reason", "")[:34]),
    ]

    lx = px + 10; row_h = 18
    for i, (k, v) in enumerate(rows):
        y = py + 36 + i * row_h
        if y > py + TECH_H - 8:
            break
        if k == "---":
            cv2.line(canvas, (px + 8, y - 3), (CW - 8, y - 3), BORDER, 1)
            continue
        _pt(canvas, f"{k}:", lx, y, 0.32, GRAY)
        _pt(canvas, str(v),  lx + 163, y, 0.32, _col(k, v))

    bar_y = py + TECH_H - 22; bar_x = px + 10
    bar_w = PANEL_W - 20;     bar_h = 10
    cv2.rectangle(canvas, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), BORDER, -1)
    if not no_pear:
        thr_x = bar_x + int(bar_w * LIVE_GOOD_ACCEPT_THRESHOLD)
        cv2.line(canvas, (thr_x, bar_y - 3), (thr_x, bar_y + bar_h + 3),
                 (200, 200, 80), 2)
        g_f = int(bar_w * min(1.0, p_good))
        b_f = int(bar_w * min(1.0, p_bad))
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + g_f, bar_y + bar_h), PASA_C, -1)
        cv2.rectangle(canvas, (bar_x + bar_w - b_f, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), RECHAZA_C, -1)
    _pt(canvas, "p_good", bar_x,              bar_y - 3, 0.27, PASA_C)
    _pt(canvas, "p_bad",  bar_x + bar_w - 35, bar_y - 3, 0.27, RECHAZA_C)


# ── Banner resultado ──────────────────────────────────────────────────────────

def draw_result_banner(canvas, stable: str, instant: str, paused: bool) -> None:
    bx = CAM_W; by = CH - RESULT_H; bw = PANEL_W; bh = RESULT_H
    color = DECISION_COLORS.get(stable, GRAY)
    tint  = tuple(int(c * 0.20) for c in color)
    _panel(canvas, bx, by, bx + bw, by + bh, tint, color)

    if paused:
        _pt(canvas, "PAUSADO", bx + 10, by + 20, 0.45, (200, 200, 80))

    font   = cv2.FONT_HERSHEY_DUPLEX
    fthick = 4
    fscale = 2.0
    while fscale > 0.7:
        (tw, _), _ = cv2.getTextSize(stable, font, fscale, fthick)
        if tw < bw - 24:
            break
        fscale -= 0.1

    (tw, th), _ = cv2.getTextSize(stable, font, fscale, fthick)
    tx = bx + (bw - tw) // 2
    ty = by + (bh + th) // 2 - 18
    cv2.putText(canvas, stable, (tx + 3, ty + 3), font, fscale,
                (0, 0, 0), fthick + 3, cv2.LINE_AA)
    cv2.putText(canvas, stable, (tx, ty), font, fscale,
                color, fthick, cv2.LINE_AA)

    _pt(canvas, f"Instant: {instant}", bx + 10, by + bh - 22, 0.38, GRAY)


# ── Miniaturas ────────────────────────────────────────────────────────────────

def draw_thumbnails(canvas, frame_bgr, mask, gray_pil, bbox,
                    show_thumbs: bool) -> None:
    if not show_thumbs:
        return
    zy = CH - THUMB_H
    _panel(canvas, 0, zy, CAM_W, CH, (20, 22, 28), BORDER)
    _pt(canvas, "Minis: original | mascara | ROI crop | gray_bg (U3 input)",
        8, zy + 15, 0.32, GRAY)

    n = 4; pad = 6
    tw = (CAM_W - pad * (n + 1)) // n
    th = THUMB_H - 35

    imgs   = [frame_bgr]
    labels = ["Original", "Mascara", "ROI crop", "gray_bg U3"]

    if mask is not None and np.any(mask > 0):
        tinted = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).copy()
        tinted[mask > 0] = [50, 180, 50]
        imgs.append(tinted)
    else:
        imgs.append(np.zeros((50, 50, 3), np.uint8))

    if bbox is not None and frame_bgr is not None:
        x, y, bw, bh = bbox
        x = max(0, x); y = max(0, y)
        bw = min(bw, frame_bgr.shape[1] - x)
        bh = min(bh, frame_bgr.shape[0] - y)
        imgs.append(frame_bgr[y:y + bh, x:x + bw] if bw > 0 and bh > 0
                    else frame_bgr)
    else:
        imgs.append(frame_bgr)

    if gray_pil is not None:
        imgs.append(cv2.cvtColor(np.array(gray_pil), cv2.COLOR_RGB2BGR))
    else:
        imgs.append(np.full((50, 50, 3), 128, np.uint8))

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        tx_ = pad + i * (tw + pad); ty_ = zy + 22
        if img is not None and img.size > 0:
            ih, iw = img.shape[:2]
            s  = min(tw / max(iw, 1), th / max(ih, 1))
            rw = max(1, int(iw * s)); rh = max(1, int(ih * s))
            thumb = cv2.resize(img, (rw, rh))
            ox_ = tx_ + (tw - rw) // 2; oy_ = ty_ + (th - rh) // 2
            canvas[oy_:oy_ + rh, ox_:ox_ + rw] = thumb
        cv2.rectangle(canvas, (tx_, ty_), (tx_ + tw, ty_ + th), BORDER, 1)
        _pt(canvas, lbl, tx_ + 3, ty_ + th + 14, 0.28, GRAY)


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDADO DE EVIDENCIAS V6
# ══════════════════════════════════════════════════════════════════════════════

def _img_save(path: Path, img_bgr: np.ndarray) -> bool:
    suffix = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(suffix, img_bgr)
    if ok:
        path.write_bytes(buf.tobytes())
    return ok


def _build_overlay(frame_bgr, contour, bbox, decision: str) -> np.ndarray:
    ov    = frame_bgr.copy()
    color = DECISION_COLORS.get(decision, GRAY)
    if contour is not None:
        cv2.drawContours(ov, [contour], -1, color, 3)
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(ov, (bx, by), (bx + bw, by + bh), color, 2)
    cv2.putText(ov, decision, (12, 38), cv2.FONT_HERSHEY_DUPLEX,
                1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(ov, decision, (12, 38), cv2.FONT_HERSHEY_DUPLEX,
                1.0, color, 2, cv2.LINE_AA)
    return ov


_CSV_COLS = [
    "timestamp", "frame_id",
    "saved_original", "saved_overlay", "saved_mask", "saved_roi", "saved_snapshot",
    "camera_index", "frame_width", "frame_height", "fps",
    "capture_status", "mask_valid", "border_cut", "bg_calibrated", "strategy_used",
    "u3_blocked", "gate_reason",
    "instant_decision", "stable_decision",
    "u3_pred", "p_good", "p_bad",
    "live_good_accept_threshold", "bad_reject_threshold",
    "pear_area_ratio", "bbox_w_ratio", "bbox_h_ratio",
    "rectangularity", "solidity",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "mask_area", "roi_width", "roi_height",
    "preprocessing_ms", "inference_ms", "total_latency_ms",
    "reason", "notes",
]


class EvidenceSaverV6:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        for sub in ("frames_original", "frames_overlay", "masks",
                    "roi_processed", "snapshots", "metadata"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "live_predictions.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, _CSV_COLS).writeheader()
        self.count     = 0
        self.last_name = "--"

    def save(self, frame_bgr, canvas, mask, gray_pil,
             contour, state: dict,
             cam_idx: int, fw: int, fh_: int, fps: float) -> str:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        fid  = state.get("frame_id", 0)
        base = f"{ts}_f{fid:06d}"
        dec  = state.get("stable_decision", "REVISAR")

        p_orig = self.out_dir / "frames_original" / f"{base}_orig.jpg"
        p_over = self.out_dir / "frames_overlay"  / f"{base}_overlay.jpg"
        p_mask = self.out_dir / "masks"           / f"{base}_mask.jpg"
        p_roi  = self.out_dir / "roi_processed"   / f"{base}_roi.jpg"
        p_snap = self.out_dir / "snapshots"        / f"{base}_snapshot.jpg"
        p_meta = self.out_dir / "metadata"         / f"{base}_data.json"

        _img_save(p_orig, frame_bgr)
        _img_save(p_over, _build_overlay(frame_bgr, contour,
                                          state.get("bbox"), dec))
        if mask is not None:
            _img_save(p_mask, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        if gray_pil is not None:
            _img_save(p_roi, cv2.cvtColor(np.array(gray_pil), cv2.COLOR_RGB2BGR))
        _img_save(p_snap, canvas)

        b = state.get("bbox") or (None, None, None, None)
        row = {
            "timestamp":       ts,       "frame_id":    fid,
            "saved_original":  p_orig.name,
            "saved_overlay":   p_over.name,
            "saved_mask":      p_mask.name if mask is not None else "",
            "saved_roi":       p_roi.name  if gray_pil is not None else "",
            "saved_snapshot":  p_snap.name,
            "camera_index":    cam_idx,  "frame_width": fw,
            "frame_height":    fh_,      "fps": f"{fps:.1f}",
            "capture_status":  state.get("capture_status",  ""),
            "mask_valid":      state.get("mask_valid",  False),
            "border_cut":      state.get("border_cut",  False),
            "bg_calibrated":   state.get("bg_calibrated", False),
            "strategy_used":   state.get("strategy_used", ""),
            "u3_blocked":      state.get("u3_blocked", False),
            "gate_reason":     state.get("gate_reason", ""),
            "instant_decision":state.get("instant_decision", ""),
            "stable_decision": state.get("stable_decision", ""),
            "u3_pred":         state.get("u3_pred", ""),
            "p_good":          f"{state.get('p_good', 0.0):.4f}",
            "p_bad":           f"{state.get('p_bad',  0.0):.4f}",
            "live_good_accept_threshold": LIVE_GOOD_ACCEPT_THRESHOLD,
            "bad_reject_threshold":       BAD_REJECT_THRESHOLD,
            "pear_area_ratio": f"{state.get('pear_area_ratio', 0.0):.4f}",
            "bbox_w_ratio":    f"{state.get('bbox_w_ratio', 0.0):.4f}",
            "bbox_h_ratio":    f"{state.get('bbox_h_ratio', 0.0):.4f}",
            "rectangularity":  f"{state.get('rectangularity', 0.0):.4f}",
            "solidity":        f"{state.get('solidity', 0.0):.4f}",
            "bbox_x": b[0] if b[0] is not None else "",
            "bbox_y": b[1] if b[1] is not None else "",
            "bbox_w": b[2] if b[2] is not None else "",
            "bbox_h": b[3] if b[3] is not None else "",
            "mask_area":        state.get("mask_area", 0),
            "roi_width":  224,  "roi_height": 224,
            "preprocessing_ms": f"{state.get('preproc_ms', 0.0):.1f}",
            "inference_ms":     f"{state.get('infer_ms',   0.0):.1f}",
            "total_latency_ms": f"{state.get('total_ms',   0.0):.1f}",
            "reason":  state.get("reason", ""),
            "notes":   state.get("notes",  ""),
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, _CSV_COLS).writerow(row)

        p_meta.write_text(
            json.dumps(row, indent=2, ensure_ascii=True), encoding="utf-8")

        self.count    += 1
        self.last_name = base
        print(f"SAVED evidence: {p_snap}")
        return base


# ══════════════════════════════════════════════════════════════════════════════
#  MODO CARPETA
# ══════════════════════════════════════════════════════════════════════════════

def run_folder_mode(model, image_folder: Path, out_dir: Path) -> None:
    folder_out = out_dir / "folder_test"
    folder_out.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(image_folder.iterdir())
              if p.suffix.lower() in exts]
    if not images:
        print(f"[WARN] No hay imagenes en: {image_folder}")
        return

    print(f"[folder] Procesando {len(images)} imagenes con politica V6")
    rows: list = []; thumbs: list = []; thumbs_rr: list = []

    for img_path in images:
        raw   = np.fromfile(str(img_path), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [SKIP] {img_path.name}"); continue

        t0  = time.perf_counter()
        det = detect_pear_v6(frame)
        t1  = time.perf_counter()

        gray_pil, _, _ = make_u3_input(frame, det.get("bbox"))
        preproc_ms = (t1 - t0) * 1000

        # Bloqueo pre-U3
        candidate_ok, gate_reason = is_valid_live_pear_candidate(det, frame)

        t2 = time.perf_counter()
        if candidate_ok:
            u3_pred, p_good, p_bad = run_u3(model, gray_pil)
            decision, reason = apply_policy_v6(u3_pred, p_good, p_bad)
        else:
            u3_pred, p_good, p_bad = "--", 0.0, 0.0
            cap_status = det["cap_status"]
            decision = "SIN PERA" if (cap_status == "SIN_PERA" or not det["mask_valid"]) \
                       else "MALA CAPTURA"
            reason   = gate_reason
        t3 = time.perf_counter(); infer_ms = (t3 - t2) * 1000

        color   = DECISION_COLORS.get(decision, GRAY)
        overlay = _build_overlay(frame, det["contour"], det["bbox"], decision)

        out_path = folder_out / f"{img_path.stem}_v6_overlay.jpg"
        ok, buf  = cv2.imencode(".jpg", overlay)
        if ok:
            out_path.write_bytes(buf.tobytes())

        th = cv2.resize(overlay, (200, 150))
        cv2.putText(th, decision[:12], (4, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)
        thumbs.append(th)
        if decision in ("REVISAR", "RECHAZA", "MALA CAPTURA"):
            thumbs_rr.append(th)

        rows.append({
            "image": img_path.name,
            "capture_status": det["cap_status"],
            "mask_valid": det["mask_valid"],
            "u3_blocked": not candidate_ok,
            "gate_reason": gate_reason,
            "decision": decision,
            "u3_pred": u3_pred,
            "p_good": f"{p_good:.4f}",
            "p_bad": f"{p_bad:.4f}",
            "pear_area_ratio": f"{det['pear_area_ratio']:.4f}",
            "strategy": det.get("strategy_used", ""),
            "preprocessing_ms": f"{preproc_ms:.1f}",
            "inference_ms": f"{infer_ms:.1f}",
            "reason": reason,
        })
        print(f"  {img_path.name}: {decision}  "
              f"p_good={p_good:.3f}  blocked={not candidate_ok}  [{det.get('strategy_used','')}]")

    if rows:
        csv_p = folder_out / "predictions.csv"
        with open(csv_p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[folder] CSV: {csv_p}")

    def _sheet(th_list, path):
        if not th_list: return
        cols  = 5
        n_row = (len(th_list) + cols - 1) // cols
        while len(th_list) < n_row * cols:
            th_list.append(np.zeros((150, 200, 3), np.uint8))
        s = np.vstack([np.hstack(th_list[r * cols:(r + 1) * cols])
                       for r in range(n_row)])
        ok, buf = cv2.imencode(".jpg", s)
        if ok: path.write_bytes(buf.tobytes())
        print(f"[folder] Sheet: {path}")

    _sheet(list(thumbs),    folder_out / "contact_sheet_all.jpg")
    _sheet(list(thumbs_rr), folder_out / "contact_sheet_review_reject.jpg")

    counts = Counter(r["decision"] for r in rows)
    lines  = [
        "PearVision QC V6 - Folder Test Summary",
        f"Folder: {image_folder}",
        f"Total : {len(rows)} images",
        f"LIVE_GOOD_ACCEPT_THRESHOLD = {LIVE_GOOD_ACCEPT_THRESHOLD}",
        f"BAD_REJECT_THRESHOLD       = {BAD_REJECT_THRESHOLD}", "",
    ]
    for d in ("PASA", "REVISAR", "RECHAZA", "SIN PERA", "MALA CAPTURA"):
        lines.append(f"  {d}: {counts.get(d, 0)}")
    lines += ["", "--- per image ---"]
    for r in rows:
        lines.append(f"  {r['image']}: {r['decision']}  p_good={r['p_good']}")
    (folder_out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[folder] Summary: {folder_out / 'summary.txt'}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PearVision QC Real-Time Camera App Pro V6")
    parser.add_argument("--camera",       type=int, default=0)
    parser.add_argument("--width",        type=int, default=1280)
    parser.add_argument("--height",       type=int, default=720)
    parser.add_argument("--infer-every",  type=int, default=5)
    parser.add_argument("--smoothing",    type=int, default=7)
    parser.add_argument("--image-folder", type=str, default=None)
    args = parser.parse_args()

    model_path = (ROOT / "outputs" / "fruits360_quality_cls_u3_roi_masked_clean"
                  / "best_model.pt")
    thr_path   = (ROOT / "outputs" / "fruits360_quality_cls_u3_roi_masked_clean"
                  / "selected_thresholds.json")
    model, _   = load_u3(model_path, thr_path)

    out_dir = ROOT / "outputs" / "live_camera_qc_pro_v6"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image_folder:
        img_folder = Path(args.image_folder)
        if not img_folder.is_absolute():
            img_folder = ROOT / img_folder
        run_folder_mode(model, img_folder, out_dir)
        return

    # ── Abrir camara ──────────────────────────────────────────────────────────
    cams    = [args.camera] + ([1] if args.camera == 0 else [0])
    cap     = None
    cam_idx = -1
    for idx in cams:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c; cam_idx = idx; break
        c.release()

    if cap is None:
        print("[ERROR] No se pudo abrir ninguna camara.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camara {cam_idx}: {fw}x{fh}")
    print(f"     LIVE_GOOD_ACCEPT_THRESHOLD = {LIVE_GOOD_ACCEPT_THRESHOLD}")
    print(f"     BAD_REJECT_THRESHOLD       = {BAD_REJECT_THRESHOLD}")
    print("     border_cut=YES -> SIN PERA / MALA CAPTURA (nunca REVISAR)")
    print("     Q/ESC=salir  S=guardar  B=cal.fondo  C=limpiar  "
          "P=pausa  R=reset  H=ayuda  M=minis")

    cv2.namedWindow("PearVision QC V6", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PearVision QC V6", CW, CH)

    saver    = EvidenceSaverV6(out_dir)
    smoother = SmoothingBuffer(args.smoothing)

    state: dict = {
        "capture_status":   "SIN_PERA",
        "mask_valid":       False,
        "border_cut":       False,
        "bg_calibrated":    False,
        "strategy_used":    "--",
        "u3_blocked":       True,
        "gate_reason":      "INIT",
        "instant_decision": "SIN PERA",
        "stable_decision":  "SIN PERA",
        "no_pear_mode":     True,
        "u3_pred":          "--",
        "p_good":           0.0,   "p_bad":          0.0,
        "pear_area_ratio":  0.0,
        "bbox_w_ratio":     0.0,   "bbox_h_ratio":   0.0,
        "rectangularity":   0.0,   "solidity":       0.0,
        "bbox":             None,  "bbox_str":       "--",
        "mask_area":        0,
        "preproc_ms":       0.0,   "infer_ms":       0.0,
        "total_ms":         0.0,
        "smoothing_count":  0,     "smoothing_window": args.smoothing,
        "saved_count":      0,     "last_saved":     "--",
        "notes":            "",    "reason":         "",
        "frame_id":         0,
    }

    bg_frame = None

    last_mask     = None
    last_gray_pil = None
    last_contour  = None
    last_bbox     = None
    last_frame    = np.zeros((fh, fw, 3), np.uint8)
    last_canvas   = _make_canvas()

    fps         = 0.0; fps_count = 0; fps_timer = time.perf_counter()
    paused      = False; show_help = True; show_thumbs = True
    infer_ctr   = 0

    save_msg_timer = 0.0; save_msg_text = ""; save_msg_ok = True

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005); continue

            state["frame_id"] += 1
            infer_ctr         += 1

            fps_count += 1
            now = time.perf_counter(); dt = now - fps_timer
            if dt >= 0.5:
                fps = fps_count / dt; fps_count = 0; fps_timer = now

            # ── Inferencia cada N frames ───────────────────────────────────
            if infer_ctr >= args.infer_every:
                infer_ctr = 0
                t0 = time.perf_counter()

                det        = detect_pear_v6(frame, bg_frame)
                cap_status = det["cap_status"]
                mask_valid = det["mask_valid"]
                border_cut = det["border_cut"]
                contour    = det["contour"]
                bbox       = det["bbox"]
                ratio      = det["pear_area_ratio"]
                metrics    = det["metrics"]
                notes      = det["notes"]
                strategy   = det.get("strategy_used", "--")

                gray_pil, _, _ = make_u3_input(frame, bbox)
                t1 = time.perf_counter(); preproc_ms = (t1 - t0) * 1000

                # ── V6: bloqueo duro pre-U3 ────────────────────────────────
                candidate_ok, gate_reason = is_valid_live_pear_candidate(det, frame)

                t2 = time.perf_counter()
                if candidate_ok:
                    u3_pred, p_good, p_bad = run_u3(model, gray_pil)
                    decision, reason = apply_policy_v6(u3_pred, p_good, p_bad)
                    smoother.add(decision)
                    stable       = smoother.stable()
                    no_pear_mode = False
                    u3_blocked   = False
                else:
                    u3_pred, p_good, p_bad = "--", 0.0, 0.0
                    # border_cut, sin mascara, forma invalida -> nunca REVISAR
                    if not mask_valid or cap_status == "SIN_PERA":
                        decision = "SIN PERA"
                    else:
                        decision = "MALA CAPTURA"
                    reason       = gate_reason
                    smoother.reset()
                    stable       = decision
                    no_pear_mode = True
                    u3_blocked   = True
                t3 = time.perf_counter(); infer_ms = (t3 - t2) * 1000

                mask      = det["mask"]
                mask_area = int(np.sum(mask > 0)) if mask is not None else 0
                bbox_str  = (f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
                             if bbox else "--")

                state.update({
                    "capture_status":   cap_status,
                    "mask_valid":       mask_valid,
                    "border_cut":       border_cut,
                    "bg_calibrated":    bg_frame is not None,
                    "strategy_used":    strategy,
                    "u3_blocked":       u3_blocked,
                    "gate_reason":      gate_reason,
                    "instant_decision": decision,
                    "stable_decision":  stable,
                    "no_pear_mode":     no_pear_mode,
                    "u3_pred":          u3_pred,
                    "p_good":           p_good,
                    "p_bad":            p_bad,
                    "pear_area_ratio":  ratio,
                    "bbox_w_ratio":     metrics.get("bw_r", 0.0),
                    "bbox_h_ratio":     metrics.get("bh_r", 0.0),
                    "rectangularity":   metrics.get("rectangularity", 0.0),
                    "solidity":         metrics.get("solidity", 0.0),
                    "bbox":             bbox,
                    "bbox_str":         bbox_str,
                    "mask_area":        mask_area,
                    "preproc_ms":       preproc_ms,
                    "infer_ms":         infer_ms,
                    "total_ms":         preproc_ms + infer_ms,
                    "smoothing_count":  smoother.count(),
                    "saved_count":      saver.count,
                    "last_saved":       saver.last_name,
                    "notes":            notes,
                    "reason":           reason,
                })

                last_mask     = mask
                last_gray_pil = gray_pil if candidate_ok else last_gray_pil
                last_contour  = contour
                last_bbox     = bbox

            # ── Dibujar dashboard ──────────────────────────────────────────
            stable  = state.get("stable_decision",  "SIN PERA")
            instant = state.get("instant_decision", "SIN PERA")

            canvas = _make_canvas()
            draw_header(canvas, fps, cam_idx, fw, fh,
                        state["frame_id"], show_help, bg_frame is not None)
            draw_camera_zone(canvas, frame, last_mask,
                             last_contour, last_bbox, stable)
            draw_tech_panel(canvas, state)
            draw_result_banner(canvas, stable, instant, paused)
            draw_thumbnails(canvas, frame, last_mask, last_gray_pil,
                            last_bbox, show_thumbs)

            if time.perf_counter() - save_msg_timer < 3.5 and save_msg_text:
                col = PASA_C if save_msg_ok else RECHAZA_C
                cv2.putText(canvas, save_msg_text, (12, CH - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, save_msg_text, (12, CH - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            cv2.imshow("PearVision QC V6", canvas)
            last_canvas = canvas.copy()
            last_frame  = frame.copy()

        # ── Teclado ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        elif key in (ord("s"), ord("S")):
            try:
                saver.save(last_frame, last_canvas, last_mask,
                           last_gray_pil, last_contour, state,
                           cam_idx, fw, fh, fps)
                state["saved_count"] = saver.count
                state["last_saved"]  = saver.last_name
                save_msg_text = f"SAVED [{saver.count}]: {saver.last_name}"
                save_msg_ok   = True
            except Exception as exc:
                save_msg_text = f"SAVE ERROR: {exc}"
                save_msg_ok   = False
                print(f"[ERROR] guardado: {exc}")
            save_msg_timer = time.perf_counter()

        elif key in (ord("b"), ord("B")):
            quick_det = detect_pear_v6(last_frame)
            bg_frame  = last_frame.copy()
            state["bg_calibrated"] = True
            if quick_det["mask_valid"]:
                save_msg_text = "WARNING: pera detectada al calibrar? Repite sin pera."
                save_msg_ok   = False
                print("[B] AVISO: posible pera en el fondo calibrado.")
            else:
                save_msg_text = "Fondo calibrado OK. Coloca la pera."
                save_msg_ok   = True
                print("[B] Fondo calibrado correctamente.")
            save_msg_timer = time.perf_counter()

        elif key in (ord("c"), ord("C")):
            bg_frame = None
            state["bg_calibrated"] = False
            save_msg_text  = "Fondo eliminado (C). Modo sin calibrar activo."
            save_msg_ok    = True
            save_msg_timer = time.perf_counter()
            print("[C] Fondo limpiado.")

        elif key in (ord("p"), ord("P")):
            paused = not paused
            print(f"[P] {'Pausado' if paused else 'Reanudado'}")

        elif key in (ord("r"), ord("R")):
            smoother.reset()
            state["smoothing_count"] = 0
            print("[R] Smoothing reseteado.")

        elif key in (ord("h"), ord("H")):
            show_help = not show_help

        elif key in (ord("m"), ord("M")):
            show_thumbs = not show_thumbs

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] App cerrada. Evidencias en: {out_dir}")
    print(f"     Frames guardados: {saver.count}")


if __name__ == "__main__":
    main()
