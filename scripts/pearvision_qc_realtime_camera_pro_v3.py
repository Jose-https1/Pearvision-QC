#!/usr/bin/env python3
"""
PearVision QC - Real-Time Camera App Pro V3

Dashboard de inspeccion de peras con camara en tiempo real.
V3 corrige sobre V2:
  - Texto ASCII puro (sin caracteres raros)
  - Calibracion de fondo con tecla B; deteccion por diferencia de fondo
  - Gating estricto: bbox > 75% frame o area > 45% -> bloquea U3
  - Estado SIN PERA resetea smoothing despues de varios frames invalidos
  - Guardado robusto con S: original, overlay, mascara, roi, snapshot, JSON, CSV

Uso:
    python scripts/pearvision_qc_realtime_camera_pro_v3.py
    python scripts/pearvision_qc_realtime_camera_pro_v3.py --camera 0 --infer-every 5
    python scripts/pearvision_qc_realtime_camera_pro_v3.py --image-folder RUTA

Controles:
    Q / ESC : salir
    S       : guardar evidencia (frame, overlay, mascara, roi, snapshot, JSON, CSV)
    B       : calibrar fondo (apuntar al fondo vacio y pulsar B)
    C       : limpiar fondo calibrado
    P       : pausar / reanudar
    R       : resetear smoothing
    H       : mostrar/ocultar ayuda
    M       : mostrar/ocultar miniaturas
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

# ── Layout canvas (1600x900) ──────────────────────────────────────────────────
CW, CH     = 1600, 900
HEADER_H   = 55
CAM_W      = 960
PANEL_W    = CW - CAM_W          # 640
THUMB_H    = 180
CAM_ZONE_H = CH - HEADER_H - THUMB_H   # 665
RESULT_H   = 290
TECH_H     = CH - HEADER_H - RESULT_H  # 555

# ── Colores (BGR) ─────────────────────────────────────────────────────────────
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
ERROR_C   = (140,  80, 180)

DECISION_COLORS: dict = {
    "PASA":     PASA_C,
    "REVISAR":  REVISAR_C,
    "RECHAZA":  RECHAZA_C,
    "SIN PERA": SINPERA_C,
    "ERROR":    ERROR_C,
}

# ── Politica de decision ──────────────────────────────────────────────────────
THR_GOOD = 0.85
THR_BAD  = 0.995

# ── Gating V3 (TAREA 8) ──────────────────────────────────────────────────────
MAX_BBOX_W_RATIO   = 0.75   # bbox_w / frame_w
MAX_BBOX_H_RATIO   = 0.75   # bbox_h / frame_h
MAX_PEAR_AREA      = 0.45   # area_contorno / area_frame
MIN_PEAR_AREA      = 0.04
MAX_RECTANGULARITY = 0.92   # area / (bbox_w * bbox_h), alto => fondo rectangular

# Ciclos SIN PERA consecutivos antes de resetear smoothing
SINPERA_RESET_STREAK = 3


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
#  SEGMENTACION V3
# ══════════════════════════════════════════════════════════════════════════════

def _segment_with_background(frame_bgr: np.ndarray,
                              bg_bgr: np.ndarray) -> np.ndarray:
    """Segmenta pera por diferencia absoluta con fondo calibrado (LAB)."""
    h, w = frame_bgr.shape[:2]
    bg_res = cv2.resize(bg_bgr, (w, h))

    lab_f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(bg_res,    cv2.COLOR_BGR2LAB).astype(np.float32)
    diff  = np.linalg.norm(lab_f - lab_b, axis=2)
    mask  = (diff > 18).astype(np.uint8) * 255

    # Eliminar sombras: baja saturacion y mas oscuro que el fondo
    hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    bg_gray = cv2.cvtColor(bg_res,    cv2.COLOR_BGR2GRAY).astype(np.float32)
    fr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    is_shadow = (hsv[:, :, 1] < 25) & (fr_gray < bg_gray - 12)
    mask[is_shadow] = 0

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    return mask


def _segment_lab_corners(frame_bgr: np.ndarray) -> np.ndarray:
    """Segmenta pera por distancia LAB al color de fondo (esquinas)."""
    h, w = frame_bgr.shape[:2]
    scale_dn = 4
    sw, sh = max(1, w // scale_dn), max(1, h // scale_dn)
    small  = cv2.resize(frame_bgr, (sw, sh))

    cs = max(6, int(min(sh, sw) * 0.08))
    corners = np.vstack([
        small[:cs, :cs].reshape(-1, 3),
        small[:cs, -cs:].reshape(-1, 3),
        small[-cs:, :cs].reshape(-1, 3),
        small[-cs:, -cs:].reshape(-1, 3),
    ])
    bg_bgr = np.median(corners, axis=0).astype(np.uint8)

    lab    = cv2.cvtColor(small,  cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_bgr.reshape(1, 1, 3),
                           cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist   = np.linalg.norm(lab - bg_lab, axis=2)
    mask_s = (dist > 25).astype(np.uint8) * 255

    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_CLOSE, k, iterations=2)
    mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN,  k, iterations=1)
    return cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)


def detect_pear_v3(frame_bgr: np.ndarray, bg_frame=None):
    """
    Detecta y valida la presencia de una pera en el frame.

    Devuelve dict con:
        mask, contour, bbox, pear_area_ratio,
        cap_status, mask_valid, border_cut, notes, metrics
    """
    h, w = frame_bgr.shape[:2]
    frame_area = w * h

    if bg_frame is not None:
        mask = _segment_with_background(frame_bgr, bg_frame)
    else:
        mask = _segment_lab_corners(frame_bgr)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    empty = dict(mask=mask, contour=None, bbox=None, pear_area_ratio=0.0,
                 cap_status="SIN_PERA", mask_valid=False, border_cut=False,
                 notes="Sin contornos", metrics={})

    if not cnts:
        return empty

    contour = max(cnts, key=cv2.contourArea)
    area    = cv2.contourArea(contour)
    ratio   = area / frame_area

    if ratio < MIN_PEAR_AREA:
        empty.update(contour=contour, pear_area_ratio=ratio,
                     notes=f"Objeto pequeno ratio={ratio:.3f}")
        return empty

    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_w_r = bw / w
    bbox_h_r = bh / h

    hull      = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1)
    solidity  = area / hull_area
    rect_area = max(bw * bh, 1)
    rectangularity = area / rect_area
    aspect    = bw / max(bh, 1)

    # ── GATING TAREA 8: bbox o ratio demasiado grande ─────────────────────────
    if bbox_w_r > MAX_BBOX_W_RATIO or bbox_h_r > MAX_BBOX_H_RATIO or ratio > MAX_PEAR_AREA:
        return dict(
            mask=mask, contour=contour, bbox=(x, y, bw, bh),
            pear_area_ratio=ratio, cap_status="MALA_CAPTURA",
            mask_valid=False, border_cut=False,
            notes=(f"bbox/frame too large: bw={bbox_w_r:.2f} bh={bbox_h_r:.2f}"
                   f" area={ratio:.3f}"),
            metrics=dict(bbox_w_r=bbox_w_r, bbox_h_r=bbox_h_r,
                         ratio=ratio, rectangularity=rectangularity))

    # ── GATING: bordas ────────────────────────────────────────────────────────
    touches_left   = x <= 2
    touches_right  = (x + bw) >= (w - 2)
    touches_top    = y <= 2
    touches_bottom = (y + bh) >= (h - 2)
    touches_count  = sum([touches_left, touches_right, touches_top, touches_bottom])

    if touches_count >= 3:
        return dict(
            mask=mask, contour=contour, bbox=(x, y, bw, bh),
            pear_area_ratio=ratio, cap_status="MALA_CAPTURA",
            mask_valid=False, border_cut=True,
            notes=f"Toca {touches_count} bordes", metrics={})

    if (touches_left and touches_right) or (touches_top and touches_bottom):
        return dict(
            mask=mask, contour=contour, bbox=(x, y, bw, bh),
            pear_area_ratio=ratio, cap_status="MALA_CAPTURA",
            mask_valid=False, border_cut=True,
            notes="Objeto atraviesa dimensión completa", metrics={})

    # ── GATING: forma rectangular (fondo/mesa) ────────────────────────────────
    if rectangularity > MAX_RECTANGULARITY and ratio > 0.15:
        return dict(
            mask=mask, contour=contour, bbox=(x, y, bw, bh),
            pear_area_ratio=ratio, cap_status="MALA_CAPTURA",
            mask_valid=False, border_cut=False,
            notes=f"Forma rectangular: rect={rectangularity:.2f}", metrics={})

    border_cut = touches_count >= 1

    metrics = dict(
        pear_area_ratio=ratio, bbox_w_r=bbox_w_r, bbox_h_r=bbox_h_r,
        aspect_ratio=aspect, solidity=solidity,
        rectangularity=rectangularity, touches_count=touches_count,
        border_cut=border_cut)

    return dict(
        mask=mask, contour=contour, bbox=(x, y, bw, bh),
        pear_area_ratio=ratio, cap_status="OK",
        mask_valid=True, border_cut=border_cut,
        notes="", metrics=metrics)


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESADO PARA U3 (igual que V2 para compatibilidad con entrenamiento)
# ══════════════════════════════════════════════════════════════════════════════

def make_gray_bg_clean(frame_bgr: np.ndarray, size: int = 224):
    """Replica el preprocesado gray_bg_clean del entrenamiento U3."""
    img = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    cs  = 12
    corners = np.vstack([
        img[:cs, :cs].reshape(-1, 3),
        img[:cs, -cs:].reshape(-1, 3),
        img[-cs:, :cs].reshape(-1, 3),
        img[-cs:, -cs:].reshape(-1, 3),
    ])
    bg_bgr = np.median(corners, axis=0).astype(np.uint8)
    lab    = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_u8  = bg_bgr.reshape(1, 1, 3)
    bg_lab = cv2.cvtColor(bg_u8, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist   = np.linalg.norm(lab - bg_lab, axis=2)
    mask   = (dist > 25).astype(np.uint8) * 255
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb[mask == 0] = [128, 128, 128]
    pil = Image.fromarray(rgb) if _TORCH_OK else None
    return pil, mask, bg_bgr


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
            tT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            probs = torch.softmax(
                model(_tf(gray_pil).unsqueeze(0)), dim=1)[0].cpu().numpy()
        p_bad, p_good = float(probs[0]), float(probs[1])
        pred = "BAD" if p_bad > p_good else "GOOD"
        return pred, p_good, p_bad
    except Exception as exc:
        print(f"[WARN] Error inferencia U3: {exc}")
        return "ERROR", 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  POLITICA DE DECISION V3
# ══════════════════════════════════════════════════════════════════════════════

def apply_policy_v3(cap_status: str, mask_valid: bool, border_cut: bool,
                    u3_pred: str, p_good: float, p_bad: float):
    if cap_status == "SIN_PERA":
        return "SIN PERA", "NO_VALID_PEAR_DETECTED"
    if cap_status == "MALA_CAPTURA" or not mask_valid:
        return "REVISAR", "BAD_CAPTURE_OR_INVALID_MASK"
    if u3_pred == "ERROR":
        return "REVISAR", "U3_INFERENCE_ERROR"
    if border_cut:
        # Pera cortada: no dar PASA aunque p_good sea alto
        if u3_pred == "BAD" and p_bad >= THR_BAD:
            return "RECHAZA", f"U3=BAD p_bad={p_bad:.4f} (borde cortado)"
        return "REVISAR", f"BORDER_CUT p_good={p_good:.4f}"
    if u3_pred == "GOOD" and p_good > THR_GOOD:
        return "PASA", f"U3=GOOD p_good={p_good:.4f}"
    if u3_pred == "BAD" and p_bad >= THR_BAD:
        return "RECHAZA", f"U3=BAD p_bad={p_bad:.4f}"
    if u3_pred == "BAD":
        return "REVISAR", f"U3=BAD p_bad={p_bad:.4f} < {THR_BAD}"
    return "REVISAR", f"Confianza insuficiente p_good={p_good:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTHING BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class SmoothingBuffer:
    def __init__(self, window: int = 7) -> None:
        self.window = window
        self._buf: deque = deque(maxlen=window)

    def add(self, decision: str) -> None:
        if decision != "SIN PERA":
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
#  HELPERS DE DIBUJO (texto ASCII puro)
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
    _pt(canvas, "PearVision QC - Real-Time Inspection V3", 12, 34,
        0.7, (190, 210, 255), 1, True)

    bg_tag = "[BG:OK]" if bg_calibrated else "[BG:---]"
    bg_col = PASA_C if bg_calibrated else REVISAR_C
    _pt(canvas, bg_tag, CW - 360, 34, 0.46, bg_col, 1, True)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = (f"{now_str}   FPS:{fps:5.1f}   CAM {cam_idx}   "
            f"{fw}x{fh}   FRAME {frame_id:06d}   U3 ROI/masked")
    tw, _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    _pt(canvas, info, CW - tw[0] - 370, 32, 0.38, GRAY)

    if show_help:
        _pt(canvas,
            "Q/ESC:Salir  S:Guardar  B:Cal.fondo  C:Limpiar  P:Pausa  R:Reset  H:Ayuda  M:Minis",
            14, 50, 0.30, (100, 140, 180))


# ── Zona camara ───────────────────────────────────────────────────────────────

def draw_camera_zone(canvas, frame_bgr, contour, bbox, decision: str):
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
    cv2.rectangle(canvas, (ox, oy), (ox + dw - 1, oy + dh - 1), color, 3)

    if contour is not None:
        dc = contour.copy().astype(np.float32)
        dc[:, :, 0] = dc[:, :, 0] * scale + ox
        dc[:, :, 1] = dc[:, :, 1] * scale + oy
        cv2.drawContours(canvas, [dc.astype(np.int32)], -1, color, 2)

    if bbox is not None:
        bx, by, bw, bh = bbox
        dx = int(bx * scale + ox); dy = int(by * scale + oy)
        dbw = int(bw * scale);     dbh = int(bh * scale)
        cv2.rectangle(canvas, (dx, dy), (dx + dbw, dy + dbh), color, 1)

    if contour is not None:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"] * scale + ox)
            cy = int(M["m01"] / M["m00"] * scale + oy)
            cv2.circle(canvas, (cx, cy), 5, color, -1)
            cv2.circle(canvas, (cx, cy), 5, WHITE, 1)

    return ox, oy, dw, dh


# ── Panel tecnico ─────────────────────────────────────────────────────────────

def draw_tech_panel(canvas, state: dict) -> None:
    px, py = CAM_W, HEADER_H
    _panel(canvas, px, py, CW, py + TECH_H)
    _pt(canvas, "DATOS TECNICOS", px + 12, py + 20, 0.48, ACCENT, 1, True)
    cv2.line(canvas, (px + 8, py + 28), (CW - 8, py + 28), BORDER, 1)

    sinpera = state.get("sinpera_mode", False)

    def _color(key, val):
        if key in ("instant_decision", "stable_decision"):
            return DECISION_COLORS.get(val, WHITE)
        if key == "u3_pred":
            return RECHAZA_C if val == "BAD" else PASA_C if val == "GOOD" else GRAY
        if key == "capture_status":
            return PASA_C if val == "OK" else SINPERA_C if "SIN" in val else REVISAR_C
        if key == "mask_valid":
            return PASA_C if val == "YES" else REVISAR_C
        if key == "bg_calibrated":
            return PASA_C if val == "YES" else REVISAR_C
        return WHITE

    p_good = state.get("p_good", 0.0)
    p_bad  = state.get("p_bad",  0.0)
    p_g_s  = "N/A" if sinpera else f"{p_good:.4f}"
    p_b_s  = "N/A" if sinpera else f"{p_bad:.4f}"
    u3_s   = "N/A" if sinpera else state.get("u3_pred", "--")

    metrics = [
        ("capture_status",   state.get("capture_status",  "--")),
        ("mask_valid",       "YES" if state.get("mask_valid", False) else "NO"),
        ("bg_calibrated",    "YES" if state.get("bg_calibrated", False) else "NO"),
        ("instant_decision", state.get("instant_decision", "--")),
        ("stable_decision",  state.get("stable_decision",  "--")),
        ("smoothing",        f"{state.get('smoothing_count',0)}/{state.get('smoothing_window',7)}"),
        ("---", ""),
        ("u3_pred",          u3_s),
        ("p_good",           p_g_s),
        ("p_bad",            p_b_s),
        ("thr_good",         f"{THR_GOOD}"),
        ("thr_bad",          f"{THR_BAD}"),
        ("---", ""),
        ("pear_area_ratio",  f"{state.get('pear_area_ratio', 0.0):.4f}"),
        ("bbox_w_ratio",     f"{state.get('bbox_w_ratio', 0.0):.3f}"),
        ("bbox_h_ratio",     f"{state.get('bbox_h_ratio', 0.0):.3f}"),
        ("rectangularity",   f"{state.get('rectangularity', 0.0):.3f}"),
        ("bbox",             state.get("bbox_str", "--")[:28]),
        ("mask_area_px",     str(state.get("mask_area", 0))),
        ("---", ""),
        ("preproc_ms",       f"{state.get('preproc_ms', 0.0):.1f}"),
        ("infer_ms",         f"{state.get('infer_ms',   0.0):.1f}"),
        ("total_ms",         f"{state.get('total_ms',   0.0):.1f}"),
        ("---", ""),
        ("saved_count",      str(state.get("saved_count", 0))),
        ("last_saved",       state.get("last_saved", "--")[:28]),
        ("reason",           state.get("reason", "")[:36]),
    ]

    lx    = px + 10
    row_h = 19
    for i, (k, v) in enumerate(metrics):
        y = py + 38 + i * row_h
        if y > py + TECH_H - 8:
            break
        if k == "---":
            cv2.line(canvas, (px + 8, y - 3), (CW - 8, y - 3), BORDER, 1)
            continue
        _pt(canvas, f"{k}:", lx, y, 0.34, GRAY)
        _pt(canvas, str(v),  lx + 170, y, 0.34, _color(k, v))

    # Barra p_good / p_bad
    bar_y = py + TECH_H - 22
    bar_x = px + 10
    bar_w = PANEL_W - 20
    bar_h = 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), BORDER, -1)
    if not sinpera:
        g_fill = int(bar_w * min(1.0, p_good))
        b_fill = int(bar_w * min(1.0, p_bad))
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + g_fill, bar_y + bar_h), PASA_C, -1)
        cv2.rectangle(canvas, (bar_x + bar_w - b_fill, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), RECHAZA_C, -1)
    _pt(canvas, "p_good", bar_x, bar_y - 3, 0.28, PASA_C)
    _pt(canvas, "p_bad",  bar_x + bar_w - 35, bar_y - 3, 0.28, RECHAZA_C)


# ── Banner resultado ──────────────────────────────────────────────────────────

def draw_result_banner(canvas, stable: str, instant: str, paused: bool) -> None:
    bx = CAM_W; by = CH - RESULT_H; bw = PANEL_W; bh = RESULT_H
    color = DECISION_COLORS.get(stable, GRAY)
    tint  = tuple(int(c * 0.20) for c in color)
    _panel(canvas, bx, by, bx + bw, by + bh, tint, color)
    if paused:
        _pt(canvas, "PAUSADO", bx + 10, by + 20, 0.45, (200, 200, 80))
    font   = cv2.FONT_HERSHEY_DUPLEX
    fscale = 2.2; fthick = 4
    (tw, th), _ = cv2.getTextSize(stable, font, fscale, fthick)
    tx = bx + (bw - tw) // 2
    ty = by + (bh + th) // 2 - 18
    cv2.putText(canvas, stable, (tx + 3, ty + 3), font, fscale,
                (0, 0, 0), fthick + 3, cv2.LINE_AA)
    cv2.putText(canvas, stable, (tx, ty), font, fscale, color, fthick, cv2.LINE_AA)
    _pt(canvas, f"Instant: {instant}", bx + 10, by + bh - 22, 0.38, GRAY)


# ── Miniaturas ────────────────────────────────────────────────────────────────

def draw_thumbnails(canvas, frame_bgr, mask, gray_pil, bbox,
                    show_thumbs: bool) -> None:
    if not show_thumbs:
        return
    zy = CH - THUMB_H
    _panel(canvas, 0, zy, CAM_W, CH, (20, 22, 28), BORDER)
    _pt(canvas, "Miniaturas tecnicas:", 8, zy + 15, 0.36, GRAY)

    n   = 4; pad = 6
    tw  = (CAM_W - pad * (n + 1)) // n
    th  = THUMB_H - 35

    imgs   = [frame_bgr]
    labels = ["Original", "Mascara", "ROI crop", "gray_bg_clean (U3)"]

    if mask is not None:
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
        imgs.append(frame_bgr[y:y + bh, x:x + bw] if bw > 0 and bh > 0 else frame_bgr)
    else:
        imgs.append(frame_bgr)

    if gray_pil is not None:
        imgs.append(cv2.cvtColor(np.array(gray_pil), cv2.COLOR_RGB2BGR))
    else:
        imgs.append(np.full((50, 50, 3), 128, np.uint8))

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        tx_ = pad + i * (tw + pad)
        ty_ = zy + 22
        if img is not None and img.size > 0:
            ih, iw = img.shape[:2]
            s  = min(tw / max(iw, 1), th / max(ih, 1))
            rw = max(1, int(iw * s)); rh = max(1, int(ih * s))
            thumb = cv2.resize(img, (rw, rh))
            ox_ = tx_ + (tw - rw) // 2
            oy_ = ty_ + (th - rh) // 2
            canvas[oy_:oy_ + rh, ox_:ox_ + rw] = thumb
        cv2.rectangle(canvas, (tx_, ty_), (tx_ + tw, ty_ + th), BORDER, 1)
        _pt(canvas, lbl, tx_ + 3, ty_ + th + 14, 0.30, GRAY)


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDADO DE EVIDENCIAS V3
# ══════════════════════════════════════════════════════════════════════════════

def _img_save(path: Path, img_bgr: np.ndarray) -> bool:
    suffix = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(suffix, img_bgr)
    if ok:
        path.write_bytes(buf.tobytes())
    return ok


def _build_overlay(frame_bgr: np.ndarray, contour, bbox, decision: str) -> np.ndarray:
    overlay = frame_bgr.copy()
    color   = DECISION_COLORS.get(decision, GRAY)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, color, 3)
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 2)
    label = decision
    cv2.putText(overlay, label, (12, 38), cv2.FONT_HERSHEY_DUPLEX,
                1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(overlay, label, (12, 38), cv2.FONT_HERSHEY_DUPLEX,
                1.1, color, 2, cv2.LINE_AA)
    return overlay


_CSV_COLS = [
    "timestamp", "frame_id",
    "saved_original", "saved_overlay", "saved_mask", "saved_roi", "saved_snapshot",
    "camera_index", "frame_width", "frame_height", "fps",
    "capture_status", "mask_valid", "border_cut",
    "instant_decision", "stable_decision",
    "u3_pred", "p_good", "p_bad", "threshold_good", "threshold_bad",
    "pear_area_ratio", "bbox_w_ratio", "bbox_h_ratio",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "mask_area", "roi_width", "roi_height",
    "preprocessing_ms", "inference_ms", "total_latency_ms",
    "reason", "notes",
]


class EvidenceSaverV3:
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

    def save(self, frame_bgr: np.ndarray, canvas: np.ndarray,
             mask, gray_pil, contour, state: dict,
             cam_idx: int, fw: int, fh: int, fps: float) -> str:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        fid  = state.get("frame_id", 0)
        base = f"{ts}_f{fid:06d}"

        decision = state.get("stable_decision", "REVISAR")

        p_orig = self.out_dir / "frames_original" / f"{base}_orig.jpg"
        p_over = self.out_dir / "frames_overlay"  / f"{base}_overlay.jpg"
        p_mask = self.out_dir / "masks"           / f"{base}_mask.jpg"
        p_roi  = self.out_dir / "roi_processed"   / f"{base}_roi.jpg"
        p_snap = self.out_dir / "snapshots"        / f"{base}_snapshot.jpg"
        p_meta = self.out_dir / "metadata"         / f"{base}_data.json"

        _img_save(p_orig, frame_bgr)

        bbox    = state.get("bbox")
        overlay = _build_overlay(frame_bgr, contour, bbox, decision)
        _img_save(p_over, overlay)

        if mask is not None:
            _img_save(p_mask, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        if gray_pil is not None:
            _img_save(p_roi, cv2.cvtColor(np.array(gray_pil), cv2.COLOR_RGB2BGR))

        _img_save(p_snap, canvas)

        b = bbox or (None, None, None, None)
        row: dict = {
            "timestamp":       ts,
            "frame_id":        fid,
            "saved_original":  p_orig.name,
            "saved_overlay":   p_over.name,
            "saved_mask":      p_mask.name if mask is not None else "",
            "saved_roi":       p_roi.name  if gray_pil is not None else "",
            "saved_snapshot":  p_snap.name,
            "camera_index":    cam_idx,
            "frame_width":     fw,
            "frame_height":    fh,
            "fps":             f"{fps:.1f}",
            "capture_status":  state.get("capture_status", ""),
            "mask_valid":      state.get("mask_valid", False),
            "border_cut":      state.get("border_cut", False),
            "instant_decision":state.get("instant_decision", ""),
            "stable_decision": state.get("stable_decision", ""),
            "u3_pred":         state.get("u3_pred", ""),
            "p_good":          f"{state.get('p_good', 0.0):.4f}",
            "p_bad":           f"{state.get('p_bad',  0.0):.4f}",
            "threshold_good":  THR_GOOD,
            "threshold_bad":   THR_BAD,
            "pear_area_ratio": f"{state.get('pear_area_ratio', 0.0):.4f}",
            "bbox_w_ratio":    f"{state.get('bbox_w_ratio', 0.0):.4f}",
            "bbox_h_ratio":    f"{state.get('bbox_h_ratio', 0.0):.4f}",
            "bbox_x":          b[0] if b[0] is not None else "",
            "bbox_y":          b[1] if b[1] is not None else "",
            "bbox_w":          b[2] if b[2] is not None else "",
            "bbox_h":          b[3] if b[3] is not None else "",
            "mask_area":       state.get("mask_area", 0),
            "roi_width":       224,
            "roi_height":      224,
            "preprocessing_ms":f"{state.get('preproc_ms', 0.0):.1f}",
            "inference_ms":    f"{state.get('infer_ms',   0.0):.1f}",
            "total_latency_ms":f"{state.get('total_ms',   0.0):.1f}",
            "reason":          state.get("reason", ""),
            "notes":           state.get("notes", ""),
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
#  MODO CARPETA (TAREA 9)
# ══════════════════════════════════════════════════════════════════════════════

def run_folder_mode_v3(model, image_folder: Path, out_dir: Path) -> None:
    folder_out = out_dir / "folder_test"
    folder_out.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(image_folder.iterdir())
              if p.suffix.lower() in exts]
    if not images:
        print(f"[WARN] No hay imagenes en: {image_folder}")
        return

    print(f"[folder] Procesando {len(images)} imagenes de: {image_folder}")

    rows:   list = []
    thumbs: list = []
    thumbs_rr: list = []

    for img_path in images:
        raw   = np.fromfile(str(img_path), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [SKIP] {img_path.name}")
            continue

        t0  = time.perf_counter()
        det = detect_pear_v3(frame)
        gray_pil, _, _ = make_gray_bg_clean(frame)
        t1  = time.perf_counter()
        preproc_ms = (t1 - t0) * 1000

        cap_status = det["cap_status"]
        mask_valid = det["mask_valid"]
        border_cut = det["border_cut"]

        t2 = time.perf_counter()
        if mask_valid:
            u3_pred, p_good, p_bad = run_u3(model, gray_pil)
        else:
            u3_pred, p_good, p_bad = "--", 0.0, 0.0
        t3 = time.perf_counter()
        infer_ms = (t3 - t2) * 1000

        decision, reason = apply_policy_v3(
            cap_status, mask_valid, border_cut, u3_pred, p_good, p_bad)
        color   = DECISION_COLORS.get(decision, GRAY)
        overlay = _build_overlay(frame, det["contour"], det["bbox"], decision)

        ts_fn    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
        out_path = folder_out / f"{ts_fn}_{img_path.stem}_overlay.jpg"
        ok, buf  = cv2.imencode(".jpg", overlay)
        if ok:
            out_path.write_bytes(buf.tobytes())

        th = cv2.resize(overlay, (200, 150))
        cv2.putText(th, decision[:10], (4, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        thumbs.append(th)
        if decision in ("REVISAR", "RECHAZA"):
            thumbs_rr.append(th)

        rows.append({
            "image":           img_path.name,
            "capture_status":  cap_status,
            "mask_valid":      mask_valid,
            "border_cut":      border_cut,
            "decision":        decision,
            "u3_pred":         u3_pred,
            "p_good":          f"{p_good:.4f}",
            "p_bad":           f"{p_bad:.4f}",
            "pear_area_ratio": f"{det['pear_area_ratio']:.4f}",
            "preprocessing_ms":f"{preproc_ms:.1f}",
            "inference_ms":    f"{infer_ms:.1f}",
            "reason":          reason,
        })
        print(f"  {img_path.name}: {decision}  "
              f"p_good={p_good:.3f}  p_bad={p_bad:.3f}")

    if rows:
        csv_path = folder_out / "predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[folder] CSV: {csv_path}")

    def _make_sheet(th_list, path):
        if not th_list:
            return
        cols  = 5
        n_row = (len(th_list) + cols - 1) // cols
        while len(th_list) < n_row * cols:
            th_list.append(np.zeros((150, 200, 3), np.uint8))
        sheet = np.vstack(
            [np.hstack(th_list[r * cols:(r + 1) * cols]) for r in range(n_row)])
        ok, buf = cv2.imencode(".jpg", sheet)
        if ok:
            path.write_bytes(buf.tobytes())
        print(f"[folder] Contact sheet: {path}")

    _make_sheet(list(thumbs),    folder_out / "contact_sheet_all.jpg")
    _make_sheet(list(thumbs_rr), folder_out / "contact_sheet_review_reject.jpg")

    # summary.txt
    counts = Counter(r["decision"] for r in rows)
    summary_lines = [
        f"PearVision QC V3 - Folder Test Summary",
        f"Folder: {image_folder}",
        f"Total images: {len(rows)}",
        "",
    ]
    for dec in ("PASA", "REVISAR", "RECHAZA", "SIN PERA", "ERROR"):
        summary_lines.append(f"  {dec}: {counts.get(dec, 0)}")
    summary_lines += ["", "--- per image ---"]
    for r in rows:
        summary_lines.append(
            f"  {r['image']}: {r['decision']}  "
            f"p_good={r['p_good']}  p_bad={r['p_bad']}")

    (folder_out / "summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8")
    print(f"[folder] summary: {folder_out / 'summary.txt'}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — bucle de camara
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PearVision QC Real-Time Camera App Pro V3")
    parser.add_argument("--camera",      type=int, default=0)
    parser.add_argument("--width",       type=int, default=1280)
    parser.add_argument("--height",      type=int, default=720)
    parser.add_argument("--infer-every", type=int, default=5)
    parser.add_argument("--smoothing",   type=int, default=7)
    parser.add_argument("--image-folder",type=str, default=None)
    args = parser.parse_args()

    model_path = (ROOT / "outputs" / "fruits360_quality_cls_u3_roi_masked_clean"
                  / "best_model.pt")
    thr_path   = (ROOT / "outputs" / "fruits360_quality_cls_u3_roi_masked_clean"
                  / "selected_thresholds.json")
    model, _   = load_u3(model_path, thr_path)

    out_dir = ROOT / "outputs" / "live_camera_qc_pro_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image_folder:
        img_folder = Path(args.image_folder)
        if not img_folder.is_absolute():
            img_folder = ROOT / img_folder
        run_folder_mode_v3(model, img_folder, out_dir)
        return

    # ── Abrir camara ──────────────────────────────────────────────────────────
    candidates = [args.camera] + ([1] if args.camera == 0 else [0])
    cap     = None
    cam_idx = -1
    for idx in candidates:
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
    print("     Q/ESC=salir  S=guardar  B=cal.fondo  C=limpiar  "
          "P=pausa  R=reset  H=ayuda  M=minis")

    cv2.namedWindow("PearVision QC V3", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PearVision QC V3", CW, CH)

    saver    = EvidenceSaverV3(out_dir)
    smoother = SmoothingBuffer(args.smoothing)

    state: dict = {
        "capture_status":   "SIN_PERA",
        "mask_valid":       False,
        "border_cut":       False,
        "instant_decision": "SIN PERA",
        "stable_decision":  "SIN PERA",
        "sinpera_mode":     True,
        "bg_calibrated":    False,
        "u3_pred":          "--",
        "p_good":           0.0,
        "p_bad":            0.0,
        "pear_area_ratio":  0.0,
        "bbox_w_ratio":     0.0,
        "bbox_h_ratio":     0.0,
        "rectangularity":   0.0,
        "bbox":             None,
        "bbox_str":         "--",
        "mask_area":        0,
        "preproc_ms":       0.0,
        "infer_ms":         0.0,
        "total_ms":         0.0,
        "smoothing_count":  0,
        "smoothing_window": args.smoothing,
        "saved_count":      0,
        "last_saved":       "--",
        "notes":            "",
        "reason":           "",
        "frame_id":         0,
    }

    bg_frame    = None   # fondo calibrado (BGR, resolucion original)
    sinpera_streak = 0   # ciclos de inferencia consecutivos sin pera valida

    last_mask     = None
    last_gray_pil = None
    last_contour  = None
    last_bbox     = None
    last_frame    = np.zeros((fh, fw, 3), np.uint8)
    last_canvas   = _make_canvas()

    fps       = 0.0
    fps_count = 0
    fps_timer = time.perf_counter()
    paused      = False
    show_help   = True
    show_thumbs = True
    infer_ctr   = 0

    # Mensaje de guardado en pantalla
    save_msg_timer = 0.0
    save_msg_text  = ""
    save_msg_ok    = True

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            state["frame_id"] += 1
            infer_ctr += 1

            fps_count += 1
            now = time.perf_counter()
            dt  = now - fps_timer
            if dt >= 0.5:
                fps       = fps_count / dt
                fps_count = 0
                fps_timer = now

            # ── Inferencia (cada N frames) ─────────────────────────────────
            if infer_ctr >= args.infer_every:
                infer_ctr = 0
                t0 = time.perf_counter()

                det = detect_pear_v3(frame, bg_frame)
                cap_status = det["cap_status"]
                mask_valid = det["mask_valid"]
                border_cut = det["border_cut"]
                contour    = det["contour"]
                bbox       = det["bbox"]
                ratio      = det["pear_area_ratio"]
                metrics    = det["metrics"]
                notes      = det["notes"]

                gray_pil, _, _ = make_gray_bg_clean(frame)
                t1 = time.perf_counter()
                preproc_ms = (t1 - t0) * 1000

                t2 = time.perf_counter()
                if mask_valid:
                    u3_pred, p_good, p_bad = run_u3(model, gray_pil)
                    sinpera_streak = 0
                else:
                    u3_pred, p_good, p_bad = "--", 0.0, 0.0
                    sinpera_streak += 1
                t3 = time.perf_counter()
                infer_ms = (t3 - t2) * 1000

                decision, reason = apply_policy_v3(
                    cap_status, mask_valid, border_cut, u3_pred, p_good, p_bad)

                smoother.add(decision)
                stable = smoother.stable()

                # Reset smoothing tras varios ciclos sin pera valida
                if sinpera_streak >= SINPERA_RESET_STREAK:
                    smoother.reset()
                    stable   = "SIN PERA"
                    u3_pred  = "--"
                    p_good   = 0.0
                    p_bad    = 0.0
                    reason   = "NO_VALID_PEAR_DETECTED"

                sinpera_mode = (stable == "SIN PERA")

                mask      = det["mask"]
                mask_area = int(np.sum(mask > 0)) if mask is not None else 0
                bbox_str  = (f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
                             if bbox else "--")

                state.update({
                    "capture_status":   cap_status,
                    "mask_valid":       mask_valid,
                    "border_cut":       border_cut,
                    "instant_decision": decision,
                    "stable_decision":  stable,
                    "sinpera_mode":     sinpera_mode,
                    "bg_calibrated":    bg_frame is not None,
                    "u3_pred":          u3_pred,
                    "p_good":           p_good,
                    "p_bad":            p_bad,
                    "pear_area_ratio":  ratio,
                    "bbox_w_ratio":     metrics.get("bbox_w_r", 0.0),
                    "bbox_h_ratio":     metrics.get("bbox_h_r", 0.0),
                    "rectangularity":   metrics.get("rectangularity", 0.0),
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
                last_gray_pil = gray_pil
                last_contour  = contour
                last_bbox     = bbox

            # ── Dibujar dashboard ──────────────────────────────────────────
            stable  = state.get("stable_decision",  "SIN PERA")
            instant = state.get("instant_decision", "SIN PERA")

            canvas = _make_canvas()
            draw_header(canvas, fps, cam_idx, fw, fh,
                        state["frame_id"], show_help, bg_frame is not None)
            draw_camera_zone(canvas, frame, last_contour, last_bbox, stable)
            draw_tech_panel(canvas, state)
            draw_result_banner(canvas, stable, instant, paused)
            draw_thumbnails(canvas, frame, last_mask, last_gray_pil,
                            last_bbox, show_thumbs)

            # Mensaje de guardado en pantalla
            if time.perf_counter() - save_msg_timer < 3.0 and save_msg_text:
                col = PASA_C if save_msg_ok else RECHAZA_C
                cv2.putText(canvas, save_msg_text, (12, CH - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, save_msg_text, (12, CH - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            cv2.imshow("PearVision QC V3", canvas)
            last_canvas = canvas.copy()
            last_frame  = frame.copy()

        # ── Teclado ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        elif key in (ord("s"), ord("S")):
            try:
                base = saver.save(
                    last_frame, last_canvas, last_mask, last_gray_pil,
                    last_contour, state, cam_idx, fw, fh, fps)
                state["saved_count"] = saver.count
                state["last_saved"]  = saver.last_name
                save_msg_text  = f"SAVED [{saver.count}]: {base}"
                save_msg_ok    = True
            except Exception as exc:
                save_msg_text = f"SAVE ERROR: {exc}"
                save_msg_ok   = False
                print(f"[ERROR] guardado: {exc}")
            save_msg_timer = time.perf_counter()

        elif key in (ord("b"), ord("B")):
            bg_frame = last_frame.copy()
            state["bg_calibrated"] = True
            print("[B] Fondo calibrado")
            save_msg_text  = "Fondo calibrado (B). Coloca la pera."
            save_msg_ok    = True
            save_msg_timer = time.perf_counter()

        elif key in (ord("c"), ord("C")):
            bg_frame = None
            state["bg_calibrated"] = False
            print("[C] Fondo limpiado")
            save_msg_text  = "Fondo calibrado eliminado (C)."
            save_msg_ok    = True
            save_msg_timer = time.perf_counter()

        elif key in (ord("p"), ord("P")):
            paused = not paused
            print(f"[P] {'Pausado' if paused else 'Reanudado'}")

        elif key in (ord("r"), ord("R")):
            smoother.reset()
            sinpera_streak = 0
            state["smoothing_count"] = 0
            print("[R] Smoothing reseteado")

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
