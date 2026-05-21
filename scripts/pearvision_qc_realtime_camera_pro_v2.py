#!/usr/bin/env python3
"""
PearVision QC — Real-Time Camera App Pro V2

Dashboard profesional de inspección de peras con cámara en tiempo real.

Uso:
    python scripts/pearvision_qc_realtime_camera_pro_v2.py
    python scripts/pearvision_qc_realtime_camera_pro_v2.py --camera 0 --infer-every 5
    python scripts/pearvision_qc_realtime_camera_pro_v2.py --image-folder data/unseen_quality_eval_input/...

Controles:
    Q / ESC : salir
    S       : guardar evidencia (frame, overlay, máscara, ROI, CSV, JSON)
    P       : pausar / reanudar
    R       : resetear smoothing temporal
    H       : mostrar/ocultar ayuda de teclado
    M       : mostrar/ocultar miniaturas técnicas
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

# ─── Imports opcionales (PyTorch / PIL) ───────────────────────────────────────
try:
    from PIL import Image
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as tT
    _TORCH_OK = True
except ImportError as _e:
    print(f"[WARN] PyTorch/PIL no disponible: {_e}")
    print("       La app abrirá la cámara pero no podrá hacer inferencia.")
    _TORCH_OK = False
    Image = None  # type: ignore

# ─── Layout del canvas (1600×900) ─────────────────────────────────────────────
CW, CH    = 1600, 900
HEADER_H  = 55
CAM_W     = 960
PANEL_W   = CW - CAM_W          # 640
THUMB_H   = 180
CAM_ZONE_H = CH - HEADER_H - THUMB_H   # 665
RESULT_H  = 290
TECH_H    = CH - HEADER_H - RESULT_H   # 555

# ─── Colores (BGR) ────────────────────────────────────────────────────────────
BG        = (15,  15,  20)
PANEL_COL = (25,  28,  35)
BORDER    = (55,  58,  65)
WHITE     = (220, 225, 235)
GRAY      = (120, 125, 135)
ACCENT    = (180, 160,  70)

PASA_C    = ( 60, 200,  70)   # verde
REVISAR_C = (  0, 145, 255)   # naranja (BGR → RGB=255,145,0)
RECHAZA_C = ( 45,  45, 220)   # rojo
SINPERA_C = (180, 130,  70)   # azul-acero
ERROR_C   = (140,  80, 180)   # violeta

DECISION_COLORS: dict[str, tuple[int, int, int]] = {
    "PASA":     PASA_C,
    "REVISAR":  REVISAR_C,
    "RECHAZA":  RECHAZA_C,
    "SIN PERA": SINPERA_C,
    "ERROR":    ERROR_C,
}

# ─── Política de decisión (cámara) ────────────────────────────────────────────
THR_GOOD        = 0.85
THR_BAD         = 0.995
MIN_PEAR_RATIO  = 0.04
MAX_PEAR_RATIO  = 0.90


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE U3
# ══════════════════════════════════════════════════════════════════════════════

def load_u3(model_path: Path, thr_path: Path):
    """Carga MobileNetV3-Small + thresholds. Devuelve (model, thr_dict) o (None, {})."""
    if not _TORCH_OK:
        return None, {}
    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return None, {}
    try:
        model = tvm.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
        model.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True)
        )
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


def make_gray_bg_clean(frame_bgr: np.ndarray, size: int = 224):
    """
    Replica el preprocesado gray_bg_clean usado en el entrenamiento de U3.
    Devuelve: (pil_224, mask_224_uint8, bg_color_bgr)
    """
    img = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)

    # Estimar color de fondo muestreando 4 esquinas de 12×12 px
    cs = 12
    corners = np.vstack([
        img[:cs, :cs].reshape(-1, 3),
        img[:cs, -cs:].reshape(-1, 3),
        img[-cs:, :cs].reshape(-1, 3),
        img[-cs:, -cs:].reshape(-1, 3),
    ])
    bg_bgr = np.median(corners, axis=0).astype(np.uint8)

    # Máscara por distancia LAB (mismo umbral que en entrenamiento: 25)
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_u8 = bg_bgr.reshape(1, 1, 3)
    bg_lab = cv2.cvtColor(bg_u8, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist  = np.linalg.norm(lab - bg_lab, axis=2)
    mask  = (dist > 25).astype(np.uint8) * 255

    # Limpieza morfológica
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    # Reemplazar fondo con gris neutro (128, 128, 128) RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb[mask == 0] = [128, 128, 128]

    if _TORCH_OK:
        pil = Image.fromarray(rgb)
    else:
        pil = None  # type: ignore

    return pil, mask, bg_bgr


def detect_pear_in_frame(frame_bgr: np.ndarray):
    """
    Detecta la pera en el frame original usando LAB distance masking.
    Opera a 1/4 de resolución para velocidad; escala el resultado al tamaño original.

    Devuelve:
        mask_full   : ndarray uint8 (misma resolución que frame_bgr)
        contour     : mayor contorno encontrado, o None
        bbox        : (x, y, w, h) en coords del frame original, o None
        pear_ratio  : área del contorno / área del frame
        cap_status  : "OK" | "SIN_PERA" | "MALA_CAPTURA"
        notes       : cadena explicativa
    """
    h, w = frame_bgr.shape[:2]

    # Reducir para velocidad
    scale_dn = 4
    small_w, small_h = max(1, w // scale_dn), max(1, h // scale_dn)
    small = cv2.resize(frame_bgr, (small_w, small_h))

    # Estimar fondo
    cs = max(6, int(min(small_h, small_w) * 0.08))
    corners = np.vstack([
        small[:cs, :cs].reshape(-1, 3),
        small[:cs, -cs:].reshape(-1, 3),
        small[-cs:, :cs].reshape(-1, 3),
        small[-cs:, -cs:].reshape(-1, 3),
    ])
    bg_bgr = np.median(corners, axis=0).astype(np.uint8)

    # Distancia LAB
    lab    = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist   = np.linalg.norm(lab - bg_lab, axis=2)
    mask_s = (dist > 25).astype(np.uint8) * 255

    # Limpieza morfológica
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_CLOSE, k, iterations=2)
    mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN,  k, iterations=1)

    # Escalar máscara al tamaño original
    mask_full = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)

    # Buscar contornos
    cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask_full, None, None, 0.0, "SIN_PERA", "Sin contornos"

    contour   = max(cnts, key=cv2.contourArea)
    area      = cv2.contourArea(contour)
    pear_ratio = area / (w * h)

    if pear_ratio < MIN_PEAR_RATIO:
        return mask_full, contour, None, pear_ratio, "SIN_PERA", \
               f"Objeto demasiado pequeño (ratio={pear_ratio:.3f})"

    if pear_ratio > MAX_PEAR_RATIO:
        return mask_full, contour, None, pear_ratio, "MALA_CAPTURA", \
               f"Objeto demasiado grande (ratio={pear_ratio:.3f})"

    x, y, bw, bh = cv2.boundingRect(contour)
    return mask_full, contour, (x, y, bw, bh), pear_ratio, "OK", ""


def run_u3(model, gray_pil):
    """
    Inferencia U3. Devuelve (u3_pred, p_good, p_bad).
    Si falla o model es None devuelve ('ERROR', 0.0, 0.0).
    """
    if model is None or gray_pil is None:
        return "ERROR", 0.0, 0.0
    try:
        _tf = tT.Compose([
            tT.Resize((224, 224)),
            tT.ToTensor(),
            tT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            probs = torch.softmax(model(_tf(gray_pil).unsqueeze(0)), dim=1)[0].cpu().numpy()
        p_bad, p_good = float(probs[0]), float(probs[1])
        pred = "BAD" if p_bad > p_good else "GOOD"
        return pred, p_good, p_bad
    except Exception as exc:
        print(f"[WARN] Error en inferencia U3: {exc}")
        return "ERROR", 0.0, 0.0


def apply_policy(cap_status: str, u3_pred: str, p_good: float, p_bad: float):
    """Política de decisión final PASA / REVISAR / RECHAZA / SIN PERA."""
    if cap_status == "SIN_PERA":
        return "SIN PERA", "No se detectó pera en el frame"
    if cap_status == "MALA_CAPTURA":
        return "REVISAR", "Captura dudosa: ratio fuera de rango"
    if u3_pred == "ERROR":
        return "REVISAR", "Error en inferencia U3"
    if u3_pred == "GOOD" and p_good > THR_GOOD:
        return "PASA", f"U3=GOOD  p_good={p_good:.4f} > {THR_GOOD}"
    if u3_pred == "BAD" and p_bad >= THR_BAD:
        return "RECHAZA", f"U3=BAD  p_bad={p_bad:.4f} >= {THR_BAD}"
    if u3_pred == "BAD":
        return "REVISAR", f"U3=BAD  p_bad={p_bad:.4f} < {THR_BAD}"
    return "REVISAR", f"Confianza insuficiente: p_good={p_good:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTHING BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class SmoothingBuffer:
    def __init__(self, window: int = 7) -> None:
        self.window = window
        self._buf: deque[str] = deque(maxlen=window)

    def add(self, decision: str) -> None:
        if decision != "SIN PERA":
            self._buf.append(decision)

    def stable(self) -> str:
        if not self._buf:
            return "SIN PERA"
        counts = Counter(self._buf)
        majority, cnt = counts.most_common(1)[0]
        # RECHAZA requiere mayoría fuerte para evitar falsas alarmas
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

def _pt(img: np.ndarray, text: str, x: int, y: int,
        scale: float = 0.45, color: tuple = WHITE,
        thickness: int = 1, bold: bool = False) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _panel(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
           fill: tuple = PANEL_COL, border: tuple = BORDER) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)


def _make_canvas() -> np.ndarray:
    c = np.full((CH, CW, 3), BG, dtype=np.uint8)
    return c


# ─── Header ───────────────────────────────────────────────────────────────────

def draw_header(canvas: np.ndarray, fps: float, cam_idx: int,
                fw: int, fh: int, frame_id: int, show_help: bool) -> None:
    _panel(canvas, 0, 0, CW, HEADER_H, (18, 22, 32), BORDER)
    _pt(canvas, "PearVision QC — Real-Time Inspection", 12, 34,
        0.7, (190, 210, 255), 1, True)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = (f"{now_str}   FPS: {fps:5.1f}   CAM {cam_idx}   "
            f"{fw}×{fh}   FRAME {frame_id:06d}   "
            f"Modelo: U3 ROI/masked clean")
    tw, _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    _pt(canvas, info, CW - tw[0] - 8, 32, 0.38, GRAY)
    if show_help:
        _pt(canvas, "Q/ESC:Salir  S:Guardar  P:Pausa  R:Reset  H:Ayuda  M:Miniaturas",
            14, 50, 0.32, (100, 140, 180))


# ─── Zona cámara (izquierda) ──────────────────────────────────────────────────

def draw_camera_zone(canvas: np.ndarray, frame_bgr: np.ndarray,
                     contour, bbox, decision: str) -> tuple:
    zx, zy   = 0, HEADER_H
    zw, zh   = CAM_W, CAM_ZONE_H
    h, w     = frame_bgr.shape[:2]
    scale    = min(zw / w, zh / h)
    dw, dh   = int(w * scale), int(h * scale)
    ox       = zx + (zw - dw) // 2
    oy       = zy + (zh - dh) // 2

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
        dx = int(bx * scale + ox)
        dy = int(by * scale + oy)
        dbw = int(bw * scale)
        dbh = int(bh * scale)
        cv2.rectangle(canvas, (dx, dy), (dx + dbw, dy + dbh), color, 1)

    if contour is not None:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"] * scale + ox)
            cy = int(M["m01"] / M["m00"] * scale + oy)
            cv2.circle(canvas, (cx, cy), 5, color, -1)
            cv2.circle(canvas, (cx, cy), 5, WHITE, 1)

    return ox, oy, dw, dh


# ─── Panel técnico (derecha superior) ─────────────────────────────────────────

def draw_tech_panel(canvas: np.ndarray, state: dict) -> None:
    px, py = CAM_W, HEADER_H
    _panel(canvas, px, py, CW, py + TECH_H)

    _pt(canvas, "DATOS TECNICOS", px + 12, py + 20, 0.48, ACCENT, 1, True)
    cv2.line(canvas, (px + 8, py + 28), (CW - 8, py + 28), BORDER, 1)

    def row_color(key: str, val: str) -> tuple:
        if key in ("instant_decision", "stable_decision"):
            return DECISION_COLORS.get(val, WHITE)
        if key == "u3_pred":
            return RECHAZA_C if val == "BAD" else PASA_C if val == "GOOD" else GRAY
        if key == "capture_status":
            return PASA_C if val == "OK" else SINPERA_C if val == "SIN_PERA" else REVISAR_C
        return WHITE

    p_good = state.get("p_good", 0.0)
    p_bad  = state.get("p_bad", 0.0)

    metrics = [
        ("capture_status",   state.get("capture_status",  "—")),
        ("instant_decision", state.get("instant_decision","—")),
        ("stable_decision",  state.get("stable_decision", "—")),
        ("smoothing_count",  f"{state.get('smoothing_count', 0)}/{state.get('smoothing_window',7)}"),
        ("─────────────────",""),
        ("u3_pred",          state.get("u3_pred",         "—")),
        ("p_good",           f"{p_good:.4f}"),
        ("p_bad",            f"{p_bad:.4f}"),
        ("thr_good",         f"{THR_GOOD}"),
        ("thr_bad",          f"{THR_BAD}"),
        ("─────────────────",""),
        ("pear_area_ratio",  f"{state.get('pear_area_ratio', 0.0):.4f}"),
        ("bbox",             state.get("bbox_str", "—")),
        ("mask_area_px",     str(state.get("mask_area", 0))),
        ("roi_size",         "224×224"),
        ("─────────────────",""),
        ("preprocessing_ms", f"{state.get('preproc_ms', 0.0):.1f}"),
        ("inference_ms",     f"{state.get('infer_ms', 0.0):.1f}"),
        ("total_latency_ms", f"{state.get('total_ms', 0.0):.1f}"),
        ("─────────────────",""),
        ("saved_count",      str(state.get("saved_count", 0))),
        ("last_saved",       state.get("last_saved", "—")[:30]),
        ("reason",           state.get("reason", "")[:38]),
    ]

    lx = px + 10
    row_h = 22
    for i, (k, v) in enumerate(metrics):
        y = py + 40 + i * row_h
        if y > py + TECH_H - 8:
            break
        if k.startswith("─"):
            cv2.line(canvas, (px + 8, y - 4), (CW - 8, y - 4), BORDER, 1)
            continue
        _pt(canvas, f"{k}:", lx, y, 0.36, GRAY)
        _pt(canvas, str(v), lx + 175, y, 0.36, row_color(k, v))

    # Mini barra p_good / p_bad
    bar_y = py + TECH_H - 24
    bar_x = px + 10
    bar_w = PANEL_W - 20
    bar_h = 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), BORDER, -1)
    g_fill = int(bar_w * min(1.0, p_good))
    b_fill = int(bar_w * min(1.0, p_bad))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + g_fill, bar_y + bar_h), PASA_C, -1)
    cv2.rectangle(canvas, (bar_x + bar_w - b_fill, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), RECHAZA_C, -1)
    _pt(canvas, "p_good", bar_x, bar_y - 3, 0.30, PASA_C)
    _pt(canvas, "p_bad", bar_x + bar_w - 35, bar_y - 3, 0.30, RECHAZA_C)


# ─── Banner resultado (derecha inferior) ──────────────────────────────────────

def draw_result_banner(canvas: np.ndarray, stable: str,
                       instant: str, paused: bool) -> None:
    bx = CAM_W
    by = CH - RESULT_H
    bw = PANEL_W
    bh = RESULT_H

    color = DECISION_COLORS.get(stable, GRAY)
    tint  = tuple(int(c * 0.20) for c in color)
    _panel(canvas, bx, by, bx + bw, by + bh, tint, color)

    if paused:
        _pt(canvas, "PAUSADO", bx + 10, by + 20, 0.45, (200, 200, 80))

    font  = cv2.FONT_HERSHEY_DUPLEX
    fscale = 2.2
    fthick = 4
    (tw, th), _ = cv2.getTextSize(stable, font, fscale, fthick)
    tx = bx + (bw - tw) // 2
    ty = by + (bh + th) // 2 - 18

    # Sombra
    cv2.putText(canvas, stable, (tx + 3, ty + 3), font, fscale,
                (0, 0, 0), fthick + 3, cv2.LINE_AA)
    cv2.putText(canvas, stable, (tx, ty), font, fscale,
                color, fthick, cv2.LINE_AA)

    _pt(canvas, f"Instantaneo: {instant}", bx + 10, by + bh - 22, 0.38, GRAY)


# ─── Miniaturas técnicas (abajo izquierda) ────────────────────────────────────

def draw_thumbnails(canvas: np.ndarray, frame_bgr: np.ndarray,
                    mask, gray_pil, bbox, show_thumbs: bool) -> None:
    if not show_thumbs:
        return

    zy = CH - THUMB_H
    _panel(canvas, 0, zy, CAM_W, CH, (20, 22, 28), BORDER)
    _pt(canvas, "Miniaturas tecnicas:", 8, zy + 15, 0.36, GRAY)

    n     = 4
    pad   = 6
    tw    = (CAM_W - pad * (n + 1)) // n
    th    = THUMB_H - 35

    # Construir 4 imágenes
    imgs   = []
    labels = ["Original", "Mascara", "ROI crop", "gray_bg_clean (U3)"]

    imgs.append(frame_bgr)

    if mask is not None:
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Tinte verde sobre la máscara
        tinted = mask_color.copy()
        tinted[mask > 0] = [50, 180, 50]
        imgs.append(tinted)
    else:
        imgs.append(np.zeros((50, 50, 3), np.uint8))

    if bbox is not None and frame_bgr is not None:
        x, y, bw, bh = bbox
        x = max(0, x); y = max(0, y)
        bw = min(bw, frame_bgr.shape[1] - x)
        bh = min(bh, frame_bgr.shape[0] - y)
        if bw > 0 and bh > 0:
            imgs.append(frame_bgr[y:y+bh, x:x+bw])
        else:
            imgs.append(frame_bgr)
    else:
        imgs.append(frame_bgr)

    if gray_pil is not None:
        import numpy as _np
        imgs.append(cv2.cvtColor(_np.array(gray_pil), cv2.COLOR_RGB2BGR))
    else:
        imgs.append(np.full((50, 50, 3), 128, np.uint8))

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        tx_ = pad + i * (tw + pad)
        ty_ = zy + 22

        if img is not None and img.size > 0:
            ih, iw = img.shape[:2]
            s = min(tw / max(iw, 1), th / max(ih, 1))
            rw, rh = max(1, int(iw * s)), max(1, int(ih * s))
            thumb = cv2.resize(img, (rw, rh))
            ox_  = tx_ + (tw - rw) // 2
            oy_  = ty_ + (th - rh) // 2
            canvas[oy_:oy_ + rh, ox_:ox_ + rw] = thumb

        cv2.rectangle(canvas, (tx_, ty_), (tx_ + tw, ty_ + th), BORDER, 1)
        _pt(canvas, lbl, tx_ + 3, ty_ + th + 14, 0.31, GRAY)


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDADO DE EVIDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def _img_save(path: Path, img_bgr: np.ndarray) -> bool:
    """Guarda imagen evitando problemas con rutas unicode."""
    ok, buf = cv2.imencode(path.suffix.lower() or ".jpg", img_bgr)
    if ok:
        path.write_bytes(buf.tobytes())
    return ok


_CSV_COLS = [
    "timestamp", "frame_id", "saved_original", "saved_overlay",
    "saved_mask", "saved_roi", "camera_index", "frame_width", "frame_height",
    "fps", "capture_status", "instant_decision", "stable_decision",
    "u3_pred", "p_good", "p_bad", "threshold_good", "threshold_bad",
    "pear_area_ratio", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "mask_area", "roi_width", "roi_height",
    "preprocessing_ms", "inference_ms", "total_latency_ms",
    "reason", "notes",
]


class EvidenceSaver:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        for sub in ("frames_original", "frames_overlay", "masks", "roi_processed", "snapshots"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

        self.csv_path = out_dir / "live_predictions.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, _CSV_COLS).writeheader()

        self.count    = 0
        self.last_name = "—"

    def save(self, frame_bgr: np.ndarray, overlay: np.ndarray,
             mask, gray_pil, state: dict,
             cam_idx: int, fw: int, fh: int, fps: float) -> str:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        fid  = state.get("frame_id", 0)
        base = f"{ts}_f{fid:06d}"

        p_orig = self.out_dir / "frames_original" / f"{base}_orig.jpg"
        p_over = self.out_dir / "frames_overlay"  / f"{base}_overlay.jpg"
        p_mask = self.out_dir / "masks"           / f"{base}_mask.jpg"
        p_roi  = self.out_dir / "roi_processed"   / f"{base}_roi.jpg"

        _img_save(p_orig, frame_bgr)
        _img_save(p_over, overlay)

        if mask is not None:
            _img_save(p_mask, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        if gray_pil is not None:
            import numpy as _np
            _img_save(p_roi, cv2.cvtColor(_np.array(gray_pil), cv2.COLOR_RGB2BGR))

        bbox = state.get("bbox") or (None, None, None, None)

        row: dict = {
            "timestamp":        ts,
            "frame_id":         fid,
            "saved_original":   p_orig.name,
            "saved_overlay":    p_over.name,
            "saved_mask":       p_mask.name if mask is not None else "",
            "saved_roi":        p_roi.name  if gray_pil is not None else "",
            "camera_index":     cam_idx,
            "frame_width":      fw,
            "frame_height":     fh,
            "fps":              f"{fps:.1f}",
            "capture_status":   state.get("capture_status",  ""),
            "instant_decision": state.get("instant_decision",""),
            "stable_decision":  state.get("stable_decision", ""),
            "u3_pred":          state.get("u3_pred",         ""),
            "p_good":           f"{state.get('p_good', 0.0):.4f}",
            "p_bad":            f"{state.get('p_bad', 0.0):.4f}",
            "threshold_good":   THR_GOOD,
            "threshold_bad":    THR_BAD,
            "pear_area_ratio":  f"{state.get('pear_area_ratio', 0.0):.4f}",
            "bbox_x":           bbox[0] if bbox[0] is not None else "",
            "bbox_y":           bbox[1] if bbox[1] is not None else "",
            "bbox_w":           bbox[2] if bbox[2] is not None else "",
            "bbox_h":           bbox[3] if bbox[3] is not None else "",
            "mask_area":        state.get("mask_area", 0),
            "roi_width":        224,
            "roi_height":       224,
            "preprocessing_ms": f"{state.get('preproc_ms', 0.0):.1f}",
            "inference_ms":     f"{state.get('infer_ms', 0.0):.1f}",
            "total_latency_ms": f"{state.get('total_ms', 0.0):.1f}",
            "reason":           state.get("reason", ""),
            "notes":            state.get("notes", ""),
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, _CSV_COLS).writerow(row)

        snap = self.out_dir / "snapshots" / f"{base}_data.json"
        snap.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

        self.count    += 1
        self.last_name = base
        print(f"[S] Guardado: {base}  (total={self.count})")
        return base


# ══════════════════════════════════════════════════════════════════════════════
#  MODO CARPETA (test sin cámara)
# ══════════════════════════════════════════════════════════════════════════════

def run_folder_mode(model, image_folder: Path, out_dir: Path) -> None:
    folder_out = out_dir / "folder_test"
    folder_out.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(image_folder.iterdir())
              if p.suffix.lower() in exts]

    if not images:
        print(f"[WARN] No hay imágenes en: {image_folder}")
        return

    print(f"[folder] Procesando {len(images)} imágenes de: {image_folder}")

    rows: list[dict] = []
    thumbs: list[np.ndarray] = []

    for img_path in images:
        raw   = np.fromfile(str(img_path), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [SKIP] No se pudo leer: {img_path.name}")
            continue

        t0 = time.perf_counter()
        mask, contour, bbox, ratio, cap_status, notes = detect_pear_in_frame(frame)
        gray_pil, _, _  = make_gray_bg_clean(frame)
        t1 = time.perf_counter()
        preproc_ms = (t1 - t0) * 1000

        t2 = time.perf_counter()
        u3_pred, p_good, p_bad = run_u3(model, gray_pil)
        t3 = time.perf_counter()
        infer_ms = (t3 - t2) * 1000

        decision, reason = apply_policy(cap_status, u3_pred, p_good, p_bad)
        color = DECISION_COLORS.get(decision, GRAY)

        overlay = frame.copy()
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, color, 3)
        if bbox is not None:
            x, y, bw, bh = bbox
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
        label = f"{decision} | p_g={p_good:.2f} p_b={p_bad:.2f}"
        cv2.putText(overlay, label, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, label, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        ts_fn    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
        out_path = folder_out / f"{ts_fn}_{img_path.stem}_overlay.jpg"
        ok, buf  = cv2.imencode(".jpg", overlay)
        if ok:
            out_path.write_bytes(buf.tobytes())

        th = cv2.resize(overlay, (200, 150))
        cv2.putText(th, decision[:8], (4, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        thumbs.append(th)

        rows.append({
            "image": img_path.name, "capture_status": cap_status,
            "decision": decision, "u3_pred": u3_pred,
            "p_good": f"{p_good:.4f}", "p_bad": f"{p_bad:.4f}",
            "pear_area_ratio": f"{ratio:.4f}",
            "preprocessing_ms": f"{preproc_ms:.1f}",
            "inference_ms": f"{infer_ms:.1f}", "reason": reason,
        })
        print(f"  {img_path.name}: {decision}  p_good={p_good:.3f}  p_bad={p_bad:.3f}")

    if rows:
        csv_path = folder_out / "folder_predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[folder] CSV: {csv_path}")

    if thumbs:
        cols  = 5
        n_row = (len(thumbs) + cols - 1) // cols
        while len(thumbs) < n_row * cols:
            thumbs.append(np.zeros((150, 200, 3), np.uint8))
        sheet_rows = [np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(n_row)]
        sheet = np.vstack(sheet_rows)
        cs_path = folder_out / "contact_sheet.jpg"
        ok, buf = cv2.imencode(".jpg", sheet)
        if ok:
            cs_path.write_bytes(buf.tobytes())
        print(f"[folder] Contact sheet: {cs_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — bucle de cámara
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PearVision QC Real-Time Camera App Pro V2"
    )
    parser.add_argument("--camera",       type=int,   default=0,
                        help="Índice de cámara (default: 0)")
    parser.add_argument("--width",        type=int,   default=1280,
                        help="Anchura solicitada (default: 1280)")
    parser.add_argument("--height",       type=int,   default=720,
                        help="Altura solicitada (default: 720)")
    parser.add_argument("--infer-every",  type=int,   default=5,
                        help="Inferencia cada N frames (default: 5)")
    parser.add_argument("--smoothing",    type=int,   default=7,
                        help="Ventana de smoothing (default: 7)")
    parser.add_argument("--image-folder", type=str,   default=None,
                        help="Modo carpeta: procesa imágenes sin cámara")
    args = parser.parse_args()

    # Cargar modelo U3
    model_path = ROOT / "outputs" / "fruits360_quality_cls_u3_roi_masked_clean" / "best_model.pt"
    thr_path   = ROOT / "outputs" / "fruits360_quality_cls_u3_roi_masked_clean" / "selected_thresholds.json"
    model, _thr = load_u3(model_path, thr_path)

    out_dir = ROOT / "outputs" / "live_camera_qc_pro_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Modo carpeta ──────────────────────────────────────────────────────────
    if args.image_folder:
        img_folder = Path(args.image_folder)
        if not img_folder.is_absolute():
            img_folder = ROOT / img_folder
        run_folder_mode(model, img_folder, out_dir)
        return

    # ── Abrir cámara ──────────────────────────────────────────────────────────
    candidates = [args.camera]
    if args.camera == 0:
        candidates.append(1)
    else:
        candidates.insert(0, 0)

    cap     = None
    cam_idx = -1
    for idx in candidates:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap     = c
            cam_idx = idx
            break
        c.release()

    if cap is None:
        print("[ERROR] No se pudo abrir ninguna cámara.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Cámara {cam_idx}: {fw}×{fh}")
    print("     Q/ESC=salir  S=guardar  P=pausa  R=reset  H=ayuda  M=miniaturas")

    cv2.namedWindow("PearVision QC", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PearVision QC", CW, CH)

    saver    = EvidenceSaver(out_dir)
    smoother = SmoothingBuffer(args.smoothing)

    state: dict = {
        "capture_status":   "SIN_PERA",
        "instant_decision": "SIN PERA",
        "stable_decision":  "SIN PERA",
        "u3_pred":          "—",
        "p_good":           0.0,
        "p_bad":            0.0,
        "pear_area_ratio":  0.0,
        "bbox":             None,
        "bbox_str":         "—",
        "mask_area":        0,
        "preproc_ms":       0.0,
        "infer_ms":         0.0,
        "total_ms":         0.0,
        "smoothing_count":  0,
        "smoothing_window": args.smoothing,
        "saved_count":      0,
        "last_saved":       "—",
        "notes":            "",
        "reason":           "",
        "frame_id":         0,
    }

    # Cache de thumbnails y overlay entre frames de inferencia
    last_mask    : np.ndarray | None = None
    last_gray_pil                    = None
    last_contour                     = None
    last_bbox                        = None

    # Buffers para guardado con S
    last_frame  = np.zeros((fh, fw, 3), np.uint8)
    last_canvas = _make_canvas()

    fps       = 0.0
    fps_count = 0
    fps_timer = time.perf_counter()

    paused      = False
    show_help   = True
    show_thumbs = True
    infer_ctr   = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            state["frame_id"] += 1
            infer_ctr         += 1

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

                mask, contour, bbox, ratio, cap_status, notes = detect_pear_in_frame(frame)
                gray_pil, _, _ = make_gray_bg_clean(frame)

                t1         = time.perf_counter()
                preproc_ms = (t1 - t0) * 1000

                t2 = time.perf_counter()
                if cap_status == "OK":
                    u3_pred, p_good, p_bad = run_u3(model, gray_pil)
                else:
                    u3_pred, p_good, p_bad = "—", 0.0, 0.0
                t3       = time.perf_counter()
                infer_ms = (t3 - t2) * 1000

                decision, reason = apply_policy(cap_status, u3_pred, p_good, p_bad)
                smoother.add(decision)
                stable = smoother.stable()

                mask_area = int(np.sum(mask > 0)) if mask is not None else 0
                bbox_str  = (f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
                             if bbox else "—")

                state.update({
                    "capture_status":   cap_status,
                    "instant_decision": decision,
                    "stable_decision":  stable,
                    "u3_pred":          u3_pred,
                    "p_good":           p_good,
                    "p_bad":            p_bad,
                    "pear_area_ratio":  ratio,
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
                        state["frame_id"], show_help)
            draw_camera_zone(canvas, frame, last_contour, last_bbox, stable)
            draw_tech_panel(canvas, state)
            draw_result_banner(canvas, stable, instant, paused)
            draw_thumbnails(canvas, frame, last_mask, last_gray_pil,
                            last_bbox, show_thumbs)

            cv2.imshow("PearVision QC", canvas)
            last_canvas = canvas
            last_frame  = frame

        # ── Teclado ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("s"), ord("S")):
            state["saved_count"] = saver.count
            saver.save(last_frame, last_canvas, last_mask, last_gray_pil,
                       state, cam_idx, fw, fh, fps)
            state["saved_count"] = saver.count
            state["last_saved"]  = saver.last_name
        elif key in (ord("p"), ord("P")):
            paused = not paused
            print(f"[P] {'Pausado' if paused else 'Reanudado'}")
        elif key in (ord("r"), ord("R")):
            smoother.reset()
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
