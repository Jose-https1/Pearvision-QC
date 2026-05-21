#!/usr/bin/env python3
"""
PearVision QC - Web Local Dashboard V1

Interfaz web accesible desde cualquier dispositivo en la misma red local.
Reutiliza el pipeline completo de V6 (segmentacion, bloqueo pre-U3, U3, politica).

Uso:
    .venv\\Scripts\\python.exe scripts\\pearvision_qc_web_local_v1.py
    .venv\\Scripts\\python.exe scripts\\pearvision_qc_web_local_v1.py --camera 0 --host 0.0.0.0 --port 8000 --infer-every 5

Desde movil: abrir http://<IP_LAN>:8000 en el navegador.
"""

import argparse
import asyncio
import csv
import datetime
import io
import json
import socket
import sys
import threading
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

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
import uvicorn


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES V6 (copiadas de V6 para no modificar el original)
# ══════════════════════════════════════════════════════════════════════════════

LIVE_GOOD_ACCEPT_THRESHOLD = 0.60
BAD_REJECT_THRESHOLD       = 0.995

MIN_AREA_RATIO     = 0.01
MAX_AREA_RATIO     = 0.45
MAX_BBOX_W_RATIO   = 0.80
MAX_BBOX_H_RATIO   = 0.80
MAX_RECTANGULARITY = 0.93
MAX_BORDER_TOUCHES = 3
MIN_CANDIDATE_SAT  = 18

PRE_U3_MIN_AREA      = 0.004
PRE_U3_MAX_AREA      = 0.45
PRE_U3_MAX_BW        = 0.80
PRE_U3_MAX_BH        = 0.80
PRE_U3_MAX_RECT      = 0.93
PRE_U3_MAX_RECT_AREA = 0.10

DECISION_COLORS_BGR = {
    "PASA":         ( 60, 200,  70),
    "REVISAR":      (  0, 145, 255),
    "RECHAZA":      ( 45,  45, 220),
    "SIN PERA":     (180, 130,  70),
    "MALA CAPTURA": ( 20, 120, 230),
    "ERROR":        (140,  80, 180),
}

DECISION_COLORS_CSS = {
    "PASA":         "#28c83c",
    "REVISAR":      "#ff8c00",
    "RECHAZA":      "#e02020",
    "SIN PERA":     "#3a8fff",
    "MALA CAPTURA": "#e0a020",
    "ERROR":        "#9050c0",
}


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE V6 (funciones copiadas de pearvision_qc_realtime_camera_pro_v6.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_u3(model_path: Path, thr_path: Path):
    if not _TORCH_OK:
        return None, {}
    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        return None, {}
    try:
        model = tvm.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
        model.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True))
        model.eval()
        thr = {}
        if thr_path.exists():
            with open(str(thr_path), encoding="utf-8") as fh:
                thr = json.load(fh)
        print(f"[OK] U3 cargado: {model_path.name}")
        return model, thr
    except Exception as exc:
        print(f"[ERROR] Cargando U3: {exc}")
        return None, {}


def _seg_by_saturation(frame_bgr):
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 35), (180, 255, 242))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    return mask


def _seg_by_pear_color(frame_bgr):
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


def _is_candidate_valid(contour, frame_bgr):
    h, w  = frame_bgr.shape[:2]
    area  = cv2.contourArea(contour)
    ratio = area / (w * h)
    if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
        return False, {}
    x, y, bw, bh = cv2.boundingRect(contour)
    bw_r = bw / w; bh_r = bh / h
    if bw_r > MAX_BBOX_W_RATIO or bh_r > MAX_BBOX_H_RATIO:
        return False, {}
    hull      = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    solidity  = area / hull_area
    rect_val  = area / max(bw * bh, 1)
    aspect    = bw / max(bh, 1)
    if rect_val > MAX_RECTANGULARITY and ratio > 0.12:
        return False, {}
    touches = sum([x <= 2, y <= 2, x + bw >= w - 2, y + bh >= h - 2])
    if touches >= MAX_BORDER_TOUCHES:
        return False, {}
    hsv     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    roi_sat = float(np.mean(hsv[y:y + bh, x:x + bw, 1]))
    if roi_sat < MIN_CANDIDATE_SAT:
        return False, {}
    perimeter = cv2.arcLength(contour, True)
    compact   = 4 * np.pi * area / max(perimeter ** 2, 1)
    return True, dict(
        area=area, ratio=ratio, bw_r=bw_r, bh_r=bh_r,
        bbox=(x, y, bw, bh), solidity=solidity,
        rectangularity=rect_val, aspect=aspect,
        touches=touches, compactness=compact, roi_sat=roi_sat)


def _score_candidate(metrics, frame_bgr):
    h, w = frame_bgr.shape[:2]
    ratio  = metrics["ratio"]; solidity = metrics["solidity"]
    rect_v = metrics["rectangularity"]; compact = metrics["compactness"]
    aspect = metrics["aspect"]; touches = metrics["touches"]
    roi_sat = metrics["roi_sat"]
    x, y, bw, bh = metrics["bbox"]
    area_score   = max(0.0, 1.0 - abs(ratio - 0.15) * 4.0)
    cx = x + bw / 2; cy = y + bh / 2
    cd = np.sqrt(((cx - w/2)/w)**2 + ((cy - h/2)/h)**2)
    center_score = max(0.0, 1.0 - cd * 2.0)
    shape_score  = solidity * max(0.0, 1.0 - max(0, rect_v - 0.75) * 2)
    compact_s    = max(0.0, 1.0 - abs(compact - 0.70) * 2.0)
    aspect_s     = max(0.0, 1.0 - max(0, abs(aspect - 0.9) - 0.7) * 1.5)
    sat_s        = min(1.0, roi_sat / 70.0)
    return (area_score*0.25 + center_score*0.15 + shape_score*0.25 +
            compact_s*0.10 + aspect_s*0.10 + sat_s*0.15 - touches*0.25)


def detect_pear_v6(frame_bgr):
    h, w = frame_bgr.shape[:2]
    _no = dict(mask=np.zeros((h, w), np.uint8), contour=None, bbox=None,
               pear_area_ratio=0.0, cap_status="SIN_PERA", mask_valid=False,
               border_cut=False, notes="Sin candidatos validos", metrics={},
               strategy_used="none")
    mask_sat   = _seg_by_saturation(frame_bgr)
    mask_color = _seg_by_pear_color(frame_bgr)
    cands = []
    for mask, tag in ((mask_sat, "sat"), (mask_color, "color")):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            ok, m = _is_candidate_valid(c, frame_bgr)
            if ok:
                cands.append((c, m, _score_candidate(m, frame_bgr), tag))
    if not cands:
        _no["mask"] = cv2.bitwise_or(mask_sat, mask_color)
        return _no
    best_c, best_m, best_s, best_tag = max(cands, key=lambda t: t[2])
    if best_s < 0:
        _no["mask"]  = cv2.bitwise_or(mask_sat, mask_color)
        _no["notes"] = f"Score negativo ({best_s:.2f})"
        return _no
    best_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(best_mask, [best_c], -1, 255, -1)
    bbox      = best_m["bbox"]
    touches   = best_m["touches"]
    notes     = f"strategy={best_tag} score={best_s:.2f} cands={len(cands)}"
    return dict(mask=best_mask, contour=best_c, bbox=bbox,
                pear_area_ratio=best_m["ratio"], cap_status="OK",
                mask_valid=True, border_cut=touches >= 1,
                notes=notes, metrics=best_m, strategy_used=best_tag)


def is_valid_live_pear_candidate(det, frame_bgr):
    if not det.get("mask_valid", False):
        return False, "NO_VALID_MASK"
    bbox = det.get("bbox")
    if bbox is None:
        return False, "NO_BBOX"
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
    if frame_bgr is not None:
        fh, fw = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        cx = (x + bw/2)/fw; cy = (y + bh/2)/fh
        corner_dist = min(
            np.sqrt(cx**2 + cy**2), np.sqrt((1-cx)**2 + cy**2),
            np.sqrt(cx**2 + (1-cy)**2), np.sqrt((1-cx)**2 + (1-cy)**2))
        if corner_dist < 0.12:
            return False, f"CANDIDATE_IN_CORNER dist={corner_dist:.3f}"
    return True, "OK"


def make_gray_bg_clean(frame_bgr, size=224):
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
    pil    = Image.fromarray(rgb) if _TORCH_OK else None
    return pil, mask, bg_bgr


def make_u3_input(frame_bgr, bbox=None, size=224):
    if bbox is not None:
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        margin = max(int(max(bw, bh)*0.25), 15)
        x1 = max(0, x-margin); y1 = max(0, y-margin)
        x2 = min(w, x+bw+margin); y2 = min(h, y+bh+margin)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
            return make_gray_bg_clean(crop, size)
    return make_gray_bg_clean(frame_bgr, size)


def run_u3(model, gray_pil):
    if model is None or gray_pil is None:
        return "ERROR", 0.0, 0.0
    try:
        _tf = tT.Compose([
            tT.Resize((224, 224)), tT.ToTensor(),
            tT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            probs = torch.softmax(model(_tf(gray_pil).unsqueeze(0)), dim=1)[0].cpu().numpy()
        p_bad, p_good = float(probs[0]), float(probs[1])
        return ("BAD" if p_bad > p_good else "GOOD"), p_good, p_bad
    except Exception as exc:
        print(f"[WARN] Error inferencia U3: {exc}")
        return "ERROR", 0.0, 0.0


def apply_policy_v6(u3_pred, p_good, p_bad):
    if u3_pred == "ERROR":
        return "REVISAR", "U3_INFERENCE_ERROR"
    if u3_pred == "GOOD" and p_good >= LIVE_GOOD_ACCEPT_THRESHOLD:
        return "PASA", f"U3=GOOD p_good={p_good:.4f} >= {LIVE_GOOD_ACCEPT_THRESHOLD}"
    if u3_pred == "BAD" and p_bad >= BAD_REJECT_THRESHOLD:
        return "RECHAZA", f"U3=BAD p_bad={p_bad:.4f} >= {BAD_REJECT_THRESHOLD}"
    if u3_pred == "BAD":
        return "REVISAR", f"U3=BAD p_bad={p_bad:.4f} < {BAD_REJECT_THRESHOLD}"
    return "REVISAR", f"p_good={p_good:.4f} < {LIVE_GOOD_ACCEPT_THRESHOLD}"


class SmoothingBuffer:
    def __init__(self, window=7):
        self.window = window
        self._buf = deque(maxlen=window)

    def add(self, decision):
        if decision not in ("SIN PERA", "MALA CAPTURA"):
            self._buf.append(decision)

    def stable(self):
        if not self._buf:
            return "SIN PERA"
        counts = Counter(self._buf)
        majority, cnt = counts.most_common(1)[0]
        if majority == "RECHAZA" and cnt < max(3, self.window // 2):
            return "REVISAR"
        if majority == "PASA" and cnt < max(2, self.window // 3):
            return "REVISAR"
        return majority

    def reset(self):
        self._buf.clear()

    def count(self):
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDADO DE EVIDENCIAS (mismo formato V6)
# ══════════════════════════════════════════════════════════════════════════════

_CSV_COLS = [
    "timestamp", "frame_id",
    "saved_original", "saved_overlay", "saved_mask", "saved_roi", "saved_snapshot",
    "camera_index", "frame_width", "frame_height", "fps",
    "capture_status", "mask_valid", "border_cut", "bg_calibrated", "strategy_used",
    "u3_blocked", "gate_reason", "instant_decision", "stable_decision",
    "u3_pred", "p_good", "p_bad",
    "live_good_accept_threshold", "bad_reject_threshold",
    "pear_area_ratio", "bbox_w_ratio", "bbox_h_ratio",
    "rectangularity", "solidity",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "mask_area", "roi_width", "roi_height",
    "preprocessing_ms", "inference_ms", "total_latency_ms",
    "reason", "notes",
]

OUT_DIR = ROOT / "outputs" / "live_camera_qc_web_v1"


def _build_overlay_bgr(frame_bgr, contour, bbox, decision):
    ov    = frame_bgr.copy()
    color = DECISION_COLORS_BGR.get(decision, (120, 120, 120))
    if contour is not None:
        cv2.drawContours(ov, [contour], -1, color, 3)
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), color, 2)
    cv2.putText(ov, decision, (12, 38), cv2.FONT_HERSHEY_DUPLEX,
                1.0, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(ov, decision, (12, 38), cv2.FONT_HERSHEY_DUPLEX,
                1.0, color, 2, cv2.LINE_AA)
    return ov


def _img_save(path: Path, img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if ok:
        path.write_bytes(buf.tobytes())
    return ok


def save_evidence(frame_bgr, gray_pil, contour, state: dict,
                  cam_idx: int, fw: int, fh_: int, fps: float) -> str:
    for sub in ("frames_original", "frames_overlay", "masks",
                "roi_processed", "snapshots", "metadata"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "live_predictions.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, _CSV_COLS).writeheader()

    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    fid  = state.get("frame_id", 0)
    base = f"{ts}_f{fid:06d}"
    dec  = state.get("stable_decision", "REVISAR")
    mask = state.get("_mask")

    p_orig = OUT_DIR / "frames_original" / f"{base}_orig.jpg"
    p_over = OUT_DIR / "frames_overlay"  / f"{base}_overlay.jpg"
    p_mask = OUT_DIR / "masks"           / f"{base}_mask.jpg"
    p_roi  = OUT_DIR / "roi_processed"   / f"{base}_roi.jpg"
    p_meta = OUT_DIR / "metadata"        / f"{base}_data.json"

    _img_save(p_orig, frame_bgr)
    _img_save(p_over, _build_overlay_bgr(frame_bgr,
              state.get("_contour"), state.get("bbox"), dec))
    if mask is not None:
        _img_save(p_mask, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    if gray_pil is not None:
        _img_save(p_roi, cv2.cvtColor(np.array(gray_pil), cv2.COLOR_RGB2BGR))

    b = state.get("bbox") or (None, None, None, None)
    row = {
        "timestamp": ts, "frame_id": fid,
        "saved_original": p_orig.name, "saved_overlay": p_over.name,
        "saved_mask": p_mask.name if mask is not None else "",
        "saved_roi":  p_roi.name  if gray_pil is not None else "",
        "saved_snapshot": "",
        "camera_index": cam_idx, "frame_width": fw, "frame_height": fh_,
        "fps": f"{fps:.1f}",
        "capture_status":  state.get("capture_status", ""),
        "mask_valid":      state.get("mask_valid", False),
        "border_cut":      state.get("border_cut", False),
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
        "roi_width": 224, "roi_height": 224,
        "preprocessing_ms": f"{state.get('preproc_ms', 0.0):.1f}",
        "inference_ms":     f"{state.get('infer_ms',   0.0):.1f}",
        "total_latency_ms": f"{state.get('total_ms',   0.0):.1f}",
        "reason": state.get("reason", ""),
        "notes":  state.get("notes",  ""),
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, _CSV_COLS).writerow(row)
    p_meta.write_text(json.dumps(row, indent=2, ensure_ascii=True), encoding="utf-8")
    return base


# ══════════════════════════════════════════════════════════════════════════════
#  ESTADO GLOBAL COMPARTIDO
# ══════════════════════════════════════════════════════════════════════════════

_lock    = threading.Lock()
_state: dict = {
    "capture_status":   "SIN_PERA",
    "mask_valid":       False,
    "border_cut":       False,
    "bg_calibrated":    False,
    "strategy_used":    "--",
    "u3_blocked":       True,
    "gate_reason":      "INIT",
    "instant_decision": "SIN PERA",
    "stable_decision":  "SIN PERA",
    "u3_pred":          "--",
    "p_good":           0.0,
    "p_bad":            0.0,
    "pear_area_ratio":  0.0,
    "bbox_w_ratio":     0.0,
    "bbox_h_ratio":     0.0,
    "rectangularity":   0.0,
    "solidity":         0.0,
    "bbox":             None,
    "bbox_str":         "--",
    "mask_area":        0,
    "preproc_ms":       0.0,
    "infer_ms":         0.0,
    "total_ms":         0.0,
    "fps":              0.0,
    "frame_id":         0,
    "saved_count":      0,
    "last_saved":       "--",
    "reason":           "",
    "notes":            "",
    "_mask":            None,
    "_contour":         None,
    "cam_idx":          0,
    "fw":               1280,
    "fh":               720,
}

_last_frame_bgr: list = [None]   # [np.ndarray | None]
_last_gray_pil:  list = [None]
_save_trigger:   list = [False]


# ══════════════════════════════════════════════════════════════════════════════
#  HILO DE CAPTURA + INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def camera_loop(camera_idx: int, infer_every: int, model) -> None:
    cams = [camera_idx] + ([1] if camera_idx == 0 else [0])
    cap  = None; cam_idx = -1
    for idx in cams:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c; cam_idx = idx; break
        c.release()
    if cap is None:
        print("[ERROR] No se pudo abrir ninguna camara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camara {cam_idx}: {fw}x{fh}")

    smoother = SmoothingBuffer(7)
    fps_count = 0; fps_timer = time.perf_counter(); fps = 0.0
    infer_ctr = 0

    with _lock:
        _state["cam_idx"] = cam_idx
        _state["fw"]      = fw
        _state["fh"]      = fh

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        with _lock:
            _state["frame_id"] += 1
            fid = _state["frame_id"]

        fps_count += 1
        now = time.perf_counter()
        if now - fps_timer >= 0.5:
            fps = fps_count / (now - fps_timer)
            fps_count = 0; fps_timer = now

        infer_ctr += 1
        if infer_ctr >= infer_every:
            infer_ctr = 0
            t0  = time.perf_counter()
            det = detect_pear_v6(frame)
            gray_pil, _, _ = make_u3_input(frame, det.get("bbox"))
            t1 = time.perf_counter(); preproc_ms = (t1 - t0) * 1000

            candidate_ok, gate_reason = is_valid_live_pear_candidate(det, frame)

            t2 = time.perf_counter()
            if candidate_ok:
                u3_pred, p_good, p_bad = run_u3(model, gray_pil)
                decision, reason = apply_policy_v6(u3_pred, p_good, p_bad)
                smoother.add(decision)
                stable       = smoother.stable()
                u3_blocked   = False
            else:
                u3_pred, p_good, p_bad = "--", 0.0, 0.0
                decision = "SIN PERA" if (not det["mask_valid"]
                           or det["cap_status"] == "SIN_PERA") else "MALA CAPTURA"
                reason   = gate_reason
                smoother.reset()
                stable     = decision
                u3_blocked = True
            t3 = time.perf_counter(); infer_ms = (t3 - t2) * 1000

            metrics   = det.get("metrics", {})
            mask_area = int(np.sum(det["mask"] > 0)) if det["mask"] is not None else 0
            bbox      = det.get("bbox")
            bbox_str  = (f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})" if bbox else "--")

            with _lock:
                _state.update({
                    "capture_status":   det["cap_status"],
                    "mask_valid":       det["mask_valid"],
                    "border_cut":       det["border_cut"],
                    "strategy_used":    det.get("strategy_used", "--"),
                    "u3_blocked":       u3_blocked,
                    "gate_reason":      gate_reason,
                    "instant_decision": decision,
                    "stable_decision":  stable,
                    "u3_pred":          u3_pred,
                    "p_good":           p_good,
                    "p_bad":            p_bad,
                    "pear_area_ratio":  det["pear_area_ratio"],
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
                    "fps":              fps,
                    "notes":            det["notes"],
                    "reason":           reason,
                    "_mask":            det["mask"],
                    "_contour":         det.get("contour"),
                })
                _last_frame_bgr[0] = frame.copy()
                _last_gray_pil[0]  = gray_pil if candidate_ok else _last_gray_pil[0]

                if _save_trigger[0]:
                    _save_trigger[0] = False
                    try:
                        base = save_evidence(
                            frame, gray_pil if candidate_ok else _last_gray_pil[0],
                            det.get("contour"), dict(_state),
                            cam_idx, fw, fh, fps)
                        _state["saved_count"] += 1
                        _state["last_saved"]   = base
                        print(f"[SAVE] {base}")
                    except Exception as exc:
                        print(f"[SAVE ERROR] {exc}")
        else:
            with _lock:
                _state["fps"] = fps
                _last_frame_bgr[0] = frame.copy()

    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  MJPEG: frame con overlay dibujado
# ══════════════════════════════════════════════════════════════════════════════

def _encode_overlay_frame() -> bytes:
    with _lock:
        frame = _last_frame_bgr[0]
        s     = dict(_state)
    if frame is None:
        blank = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(blank, "Iniciando camara...", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
        _, buf = cv2.imencode(".jpg", blank)
        return buf.tobytes()

    decision = s.get("stable_decision", "SIN PERA")
    contour  = s.get("_contour")
    bbox     = s.get("bbox")
    color    = DECISION_COLORS_BGR.get(decision, (120, 120, 120))
    ov       = frame.copy()

    if contour is not None:
        cv2.drawContours(ov, [contour], -1, color, 3)
    if bbox is not None:
        bx, by, bw, bh = bbox
        cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), color, 2)

    cv2.putText(ov, decision, (12, 44), cv2.FONT_HERSHEY_DUPLEX,
                1.4, (0,0,0), 6, cv2.LINE_AA)
    cv2.putText(ov, decision, (12, 44), cv2.FONT_HERSHEY_DUPLEX,
                1.4, color, 3, cv2.LINE_AA)

    fps  = s.get("fps", 0.0)
    info = (f"FPS:{fps:.1f}  p_good:{s.get('p_good',0):.3f}  "
            f"p_bad:{s.get('p_bad',0):.3f}  {s.get('gate_reason','')}")
    cv2.putText(ov, info, (8, ov.shape[0]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(ov, info, (8, ov.shape[0]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


def _mjpeg_generator():
    while True:
        data = _encode_overlay_frame()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.04)  # ~25 fps max al cliente


# ══════════════════════════════════════════════════════════════════════════════
#  HTML DE LA INTERFAZ
# ══════════════════════════════════════════════════════════════════════════════

_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PearVision QC - Remote Local Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d0f14;color:#c8d0e0;font-family:'Segoe UI',sans-serif;min-height:100vh}
  header{background:#131720;border-bottom:1px solid #2a2d3a;padding:12px 20px;display:flex;align-items:center;justify-content:space-between}
  header h1{font-size:1.15rem;letter-spacing:.05em;color:#8ab4f8}
  header .badge{font-size:.7rem;background:#1e2533;border:1px solid #3a4060;border-radius:4px;padding:3px 8px;color:#7090c0}
  .main{display:flex;gap:0;height:calc(100vh - 52px)}
  .video-col{flex:1 1 0;min-width:0;background:#060709;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:10px}
  .video-col img{max-width:100%;max-height:calc(100vh - 150px);border:2px solid #2a2d3a;border-radius:6px}
  .panel{width:300px;flex-shrink:0;background:#111420;border-left:1px solid #1e2230;display:flex;flex-direction:column;overflow:hidden}
  .result-box{padding:20px 16px;text-align:center;border-bottom:1px solid #1e2230;transition:background .3s}
  .result-label{font-size:2.4rem;font-weight:700;letter-spacing:.04em;line-height:1.1;transition:color .3s}
  .result-sub{font-size:.72rem;color:#606880;margin-top:6px}
  .btn-save{margin:10px 16px;padding:10px;background:#1e2d4a;border:1px solid #3a6090;border-radius:6px;color:#8ab4f8;cursor:pointer;font-size:.85rem;width:calc(100% - 32px);transition:background .2s}
  .btn-save:hover{background:#25395e}
  .btn-save:active{background:#0d1f38}
  .save-msg{font-size:.7rem;color:#60c080;text-align:center;padding:2px 16px 6px;min-height:18px}
  .tech{flex:1;overflow-y:auto;padding:10px 14px}
  .tech h3{font-size:.7rem;letter-spacing:.1em;color:#4a6080;text-transform:uppercase;margin-bottom:8px;border-bottom:1px solid #1e2230;padding-bottom:4px}
  .row{display:flex;justify-content:space-between;padding:3px 0;font-size:.72rem;border-bottom:1px solid #13161f}
  .row .k{color:#5a6880}.row .v{font-weight:600;text-align:right;max-width:55%}
  .sep{border:none;border-top:1px solid #1e2536;margin:6px 0}
  .bar-wrap{margin:10px 0 4px;position:relative;height:10px;background:#1a1d28;border-radius:4px;overflow:hidden}
  .bar-good{height:100%;border-radius:4px 0 0 4px;background:#28c83c;transition:width .3s}
  .bar-bad{position:absolute;right:0;top:0;height:100%;border-radius:0 4px 4px 0;background:#e02020;transition:width .3s}
  .bar-thr{position:absolute;top:-2px;width:2px;height:14px;background:#e0d020}
  .bar-labels{display:flex;justify-content:space-between;font-size:.62rem;color:#4a6080}
  @media(max-width:600px){.panel{width:100%}.main{flex-direction:column}.video-col img{max-height:55vw}}
</style>
</head>
<body>
<header>
  <h1>&#127; PearVision QC &mdash; Remote Local Dashboard</h1>
  <span class="badge" id="hdr-fps">FPS: --</span>
</header>
<div class="main">
  <div class="video-col">
    <img id="stream" src="/video_feed" alt="Camara">
  </div>
  <div class="panel">
    <div class="result-box" id="result-box">
      <div class="result-label" id="result-label">--</div>
      <div class="result-sub" id="result-sub">Esperando datos...</div>
    </div>
    <button class="btn-save" onclick="saveEvidence()">&#128190; Guardar evidencia</button>
    <div class="save-msg" id="save-msg"></div>
    <div class="tech">
      <h3>Datos tecnicos</h3>
      <div class="row"><span class="k">capture_status</span><span class="v" id="t-cs">--</span></div>
      <div class="row"><span class="k">mask_valid</span><span class="v" id="t-mv">--</span></div>
      <div class="row"><span class="k">stable_decision</span><span class="v" id="t-sd">--</span></div>
      <div class="row"><span class="k">instant_decision</span><span class="v" id="t-id">--</span></div>
      <hr class="sep">
      <div class="row"><span class="k">u3_pred</span><span class="v" id="t-u3">--</span></div>
      <div class="row"><span class="k">p_good</span><span class="v" id="t-pg" style="color:#28c83c">--</span></div>
      <div class="row"><span class="k">p_bad</span><span class="v" id="t-pb" style="color:#e02020">--</span></div>
      <div class="row"><span class="k">threshold_good</span><span class="v" id="t-tg">--</span></div>
      <div class="row"><span class="k">threshold_bad</span><span class="v" id="t-tb">--</span></div>
      <div class="bar-wrap">
        <div class="bar-good" id="bar-good" style="width:0%"></div>
        <div class="bar-bad"  id="bar-bad"  style="width:0%"></div>
        <div class="bar-thr"  id="bar-thr"  style="left:60%"></div>
      </div>
      <div class="bar-labels"><span>p_good</span><span>p_bad</span></div>
      <hr class="sep">
      <div class="row"><span class="k">pear_area_ratio</span><span class="v" id="t-ar">--</span></div>
      <div class="row"><span class="k">bbox</span><span class="v" id="t-bb">--</span></div>
      <div class="row"><span class="k">gate_reason</span><span class="v" id="t-gr">--</span></div>
      <hr class="sep">
      <div class="row"><span class="k">FPS</span><span class="v" id="t-fps">--</span></div>
      <div class="row"><span class="k">latencia total</span><span class="v" id="t-lat">--</span></div>
      <div class="row"><span class="k">preproc_ms</span><span class="v" id="t-pre">--</span></div>
      <div class="row"><span class="k">infer_ms</span><span class="v" id="t-inf">--</span></div>
      <hr class="sep">
      <div class="row"><span class="k">saved_count</span><span class="v" id="t-sc">0</span></div>
      <div class="row"><span class="k">last_saved</span><span class="v" id="t-ls" style="font-size:.62rem">--</span></div>
    </div>
  </div>
</div>
<script>
const COLORS={
  "PASA":"#28c83c","REVISAR":"#ff8c00","RECHAZA":"#e02020",
  "SIN PERA":"#3a8fff","MALA CAPTURA":"#e0a020","ERROR":"#9050c0"};
function g(id){return document.getElementById(id)}
function update(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    const dec=d.stable_decision||'--';
    const col=COLORS[dec]||'#8090a0';
    g('result-label').textContent=dec;
    g('result-label').style.color=col;
    g('result-box').style.background=col+'18';
    g('result-sub').textContent='instant: '+(d.instant_decision||'--');
    g('hdr-fps').textContent='FPS: '+(d.fps||0).toFixed(1);
    g('t-cs').textContent=d.capture_status||'--';
    g('t-mv').textContent=d.mask_valid?'YES':'NO';
    g('t-sd').textContent=dec; g('t-sd').style.color=col;
    g('t-id').textContent=d.instant_decision||'--';
    g('t-u3').textContent=d.u3_pred||'--';
    g('t-pg').textContent=(d.p_good||0).toFixed(4);
    g('t-pb').textContent=(d.p_bad||0).toFixed(4);
    g('t-tg').textContent=d.threshold_good;
    g('t-tb').textContent=d.threshold_bad;
    g('bar-good').style.width=Math.min(100,d.p_good*100).toFixed(1)+'%';
    g('bar-bad').style.width=Math.min(100,d.p_bad*100).toFixed(1)+'%';
    g('bar-thr').style.left=(d.threshold_good*100).toFixed(1)+'%';
    g('t-ar').textContent=(d.pear_area_ratio||0).toFixed(4);
    g('t-bb').textContent=d.bbox_str||'--';
    g('t-gr').textContent=d.gate_reason||'--';
    g('t-fps').textContent=(d.fps||0).toFixed(1);
    g('t-lat').textContent=(d.total_ms||0).toFixed(1)+' ms';
    g('t-pre').textContent=(d.preproc_ms||0).toFixed(1)+' ms';
    g('t-inf').textContent=(d.infer_ms||0).toFixed(1)+' ms';
    g('t-sc').textContent=d.saved_count||0;
    g('t-ls').textContent=d.last_saved||'--';
  }).catch(()=>{});
}
function saveEvidence(){
  fetch('/api/save',{method:'POST'}).then(r=>r.json()).then(d=>{
    const m=g('save-msg');
    m.textContent=d.ok?'Guardado: '+d.name:'Error al guardar';
    m.style.color=d.ok?'#60c080':'#e06050';
    setTimeout(()=>{m.textContent=''},4000);
  });
}
setInterval(update,400);
update();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="PearVision QC Web Local V1", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_HTML)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
async def api_status():
    with _lock:
        s = {k: v for k, v in _state.items() if not k.startswith("_")}
    # serializar bbox
    if s.get("bbox") is not None:
        bx, by, bw, bh = s["bbox"]
        s["bbox"] = {"x": bx, "y": by, "w": bw, "h": bh}
    s["threshold_good"] = LIVE_GOOD_ACCEPT_THRESHOLD
    s["threshold_bad"]  = BAD_REJECT_THRESHOLD
    return JSONResponse(s)


@app.get("/api/health")
async def api_health():
    with _lock:
        alive = _last_frame_bgr[0] is not None
        fid   = _state.get("frame_id", 0)
    return JSONResponse({"status": "ok", "camera_alive": alive, "frame_id": fid})


@app.post("/api/save")
async def api_save():
    with _lock:
        _save_trigger[0] = True
    # esperar hasta 2s a que el hilo lo procese
    for _ in range(40):
        await asyncio.sleep(0.05)
        with _lock:
            if not _save_trigger[0]:
                name = _state.get("last_saved", "--")
                return JSONResponse({"ok": True, "name": name})
    return JSONResponse({"ok": False, "name": "--"})


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "TU_IP_LOCAL"


def main():
    parser = argparse.ArgumentParser(description="PearVision QC Web Local V1")
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--host",       type=str,   default="0.0.0.0")
    parser.add_argument("--port",       type=int,   default=8000)
    parser.add_argument("--infer-every",type=int,   default=5)
    args = parser.parse_args()

    model_path = (ROOT / "outputs" / "fruits360_quality_cls_u3_roi_masked_clean"
                  / "best_model.pt")
    thr_path   = (ROOT / "outputs" / "fruits360_quality_cls_u3_roi_masked_clean"
                  / "selected_thresholds.json")
    model, _   = load_u3(model_path, thr_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lan_ip = _get_lan_ip()
    print()
    print("=" * 60)
    print("  PearVision QC - Web Local Dashboard V1")
    print("=" * 60)
    print(f"  URL local (portátil):  http://127.0.0.1:{args.port}")
    print(f"  URL LAN (móvil):       http://{lan_ip}:{args.port}")
    print()
    print("  Para obtener tu IP LAN: ejecuta 'ipconfig' en PowerShell")
    print("  y busca 'Dirección IPv4' en el adaptador WiFi.")
    print(f"  Ejemplo para móvil:    http://TU_IP_LOCAL:{args.port}")
    print()
    print("  El móvil debe estar en la misma red WiFi que el portátil.")
    print("  Si no conecta: revisar firewall de Windows, puerto 8000.")
    print("=" * 60)
    print()

    cam_thread = threading.Thread(
        target=camera_loop,
        args=(args.camera, args.infer_every, model),
        daemon=True)
    cam_thread.start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
