#!/usr/bin/env python3
"""
PearVision QC — Camera Smoke Test V2

Verifica que la cámara funciona antes de abrir la app completa.

Uso:
    python scripts/camera_smoke_test_v2.py
    python scripts/camera_smoke_test_v2.py --camera 0 --width 1280 --height 720

Controles:
    S       : guardar frame en data/live_camera_smoke_test/
    Q / ESC : salir
"""

import argparse
import datetime
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="PearVision QC Camera Smoke Test V2")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara (default: 0)")
    parser.add_argument("--width",  type=int, default=1280, help="Anchura solicitada (default: 1280)")
    parser.add_argument("--height", type=int, default=720,  help="Altura solicitada (default: 720)")
    args = parser.parse_args()

    out_dir = ROOT / "data" / "live_camera_smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intentar cámara solicitada; si falla, probar la otra
    candidates = [args.camera]
    if args.camera == 0:
        candidates.append(1)
    else:
        candidates.append(0)

    cap = None
    cam_idx = -1
    for idx in candidates:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c
            cam_idx = idx
            break
        c.release()

    if cap is None:
        print("[ERROR] No se pudo abrir ninguna cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Cámara {cam_idx} abierta: {fw}x{fh}")
    print("     Controles: S=guardar frame  Q/ESC=salir")

    fps = 0.0
    fps_count = 0
    fps_timer = time.perf_counter()
    saved = 0

    cv2.namedWindow("PearVision QC — Smoke Test V2", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        fps_count += 1
        now = time.perf_counter()
        dt = now - fps_timer
        if dt >= 1.0:
            fps = fps_count / dt
            fps_count = 0
            fps_timer = now

        overlay = frame.copy()
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "PearVision QC — Camera Smoke Test V2",
            f"Camara: {cam_idx}   Resolucion: {fw}x{fh}   FPS: {fps:.1f}",
            f"Hora: {ts_str}",
            f"Frames guardados: {saved}   |   S=guardar   Q/ESC=salir",
        ]

        for i, line in enumerate(lines):
            y = 30 + i * 28
            cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (220, 240, 255), 1, cv2.LINE_AA)

        cv2.imshow("PearVision QC — Smoke Test V2", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("s"), ord("S")):
            ts_fn = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            path = out_dir / f"smoke_{ts_fn}.jpg"
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                path.write_bytes(buf.tobytes())
                saved += 1
                print(f"[S] Guardado: {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] Cerrado. {saved} frames en: {out_dir}")


if __name__ == "__main__":
    main()
