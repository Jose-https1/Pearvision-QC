"""
make_fruits360_contact_sheet.py
Genera contact sheet de auditoría del pipeline sobre Fruits-360 good eval.

Orden: RECHAZA primero, luego REVISAR (muestra), luego PASA (muestra).
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_CSV = PROJECT_ROOT / "outputs" / "quality_analysis_fruits360_good_eval" / "resultados_calidad.csv"
IMGS_DIR    = PROJECT_ROOT / "outputs" / "quality_analysis_fruits360_good_eval"
AUDIT_DIR   = PROJECT_ROOT / "outputs" / "quality_audit_fruits360_good_eval"
OUT_SHEET   = AUDIT_DIR / "contact_sheet_fruits360_good_eval.jpg"

MAX_REVISAR = 20
MAX_PASA    = 10


def _load_panel(path: Path, w=180, h=160):
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError
        scale = min(w / bgr.shape[1], h / bgr.shape[0])
        nw = max(1, int(bgr.shape[1] * scale))
        nh = max(1, int(bgr.shape[0] * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((h, w, 3), 45, dtype=np.uint8)
        yo, xo = (h - nh) // 2, (w - nw) // 2
        canvas[yo:yo+nh, xo:xo+nw] = bgr
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    except Exception:
        return Image.fromarray(np.full((h, w, 3), 60, dtype=np.uint8))


def _get_font(size):
    for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def build_sheet(rows):
    CELL_W  = 200   # texto
    IMG_W   = 180   # cada panel de imagen
    IMG_H   = 160
    N_IMGS  = 2     # overlay + crop
    ROW_W   = CELL_W + IMG_W * N_IMGS
    ROW_H   = IMG_H + 8
    HDR_H   = 32

    rechaza_rows = [r for r in rows if r["decision"] == "RECHAZA"]
    revisar_rows = [r for r in rows if r["decision"] == "REVISAR"][:MAX_REVISAR]
    pasa_rows    = [r for r in rows if r["decision"] == "PASA"][:MAX_PASA]

    all_rows = rechaza_rows + revisar_rows + pasa_rows
    if not all_rows:
        print("No hay filas para mostrar.")
        return

    total_h = HDR_H + ROW_H * len(all_rows)
    sheet = Image.new("RGB", (ROW_W, total_h), (22, 22, 22))
    draw  = ImageDraw.Draw(sheet)

    f_sm = _get_font(10)
    f_md = _get_font(12)
    f_lg = _get_font(14)

    draw.text((6, 6),
              f"PearVision QC -- Fruits-360 Good Eval Audit  "
              f"(RECHAZA={len(rechaza_rows)}  REVISAR={len(revisar_rows)}  PASA={len(pasa_rows)})",
              fill=(220, 220, 220), font=f_lg)

    section_labels = {}
    for i, row in enumerate(all_rows):
        dec = row["decision"]
        if i == 0 or all_rows[i-1]["decision"] != dec:
            section_labels[i] = dec

    for ri, row in enumerate(all_rows):
        y0 = HDR_H + ri * ROW_H
        decision = row["decision"]

        bg = {"RECHAZA": (55, 22, 22), "REVISAR": (44, 40, 22), "PASA": (22, 44, 22)}.get(decision, (33, 33, 33))
        draw.rectangle([(0, y0), (ROW_W, y0 + ROW_H)], fill=bg)

        dec_color = {"PASA": (0, 210, 80), "REVISAR": (230, 185, 0), "RECHAZA": (230, 60, 60)}.get(decision, (180, 180, 180))

        # Sección label
        if ri in section_labels:
            draw.rectangle([(0, y0), (ROW_W, y0 + 14)], fill=(60, 40, 40) if decision == "RECHAZA" else (50, 50, 30) if decision == "REVISAR" else (30, 55, 30))
            draw.text((4, y0 + 1), f"--- {decision} ---", fill=dec_color, font=f_md)
            y0 += 14

        # Texto
        stem = Path(row["image"]).stem
        short_name = row["image"].replace("Training__", "Tr__").replace("Validation__", "Va__").replace("Test__", "Te__")
        cap_ok = row.get("capture_valid", "true").lower() in ("true", "1")
        false_rej = decision == "RECHAZA"

        ty = y0 + 2
        line_h = 13

        def put(txt, color, font=f_sm):
            nonlocal ty
            draw.text((4, ty), txt, fill=color, font=font)
            ty += line_h

        put(short_name[:36], (200, 200, 200), f_md)
        put(f"{decision}  {row.get('estimated_category','')}", dec_color, f_md)
        put(f"def={row.get('defect_pct','?')}%  rot={row.get('dark_rot_pct','?')}%  max={row.get('max_region_pct','?')}%", (175, 175, 175))
        put(f"brown={row.get('brown_dark_pct','?')}%  L={row.get('body_l_mean','?')}", (155, 155, 155))
        put(f"cls={row.get('quality_cls_pred','?')} bad={row.get('quality_cls_bad_conf','?')}", (160, 210, 255))
        if not cap_ok:
            put(">> CAPTURA INVALIDA", (200, 120, 50))
        if false_rej:
            put("!! FALSO RECHAZO", (255, 80, 80))

        # Paneles de imagen
        for pi, suffix in enumerate(["", "_crop"]):
            p = IMGS_DIR / f"{stem}{suffix}.jpg"
            x0 = CELL_W + pi * IMG_W
            if p.exists():
                panel = _load_panel(p, IMG_W, IMG_H - 8)
                sheet.paste(panel, (x0, y0 + 4))
            else:
                draw.rectangle([(x0 + 2, y0 + 4), (x0 + IMG_W - 2, y0 + IMG_H)], fill=(55, 55, 55))
                draw.text((x0 + IMG_W // 2 - 12, y0 + IMG_H // 2), "N/A", fill=(90, 90, 90), font=f_sm)
        # etiquetas de columna (solo primera fila)
        if ri == 0:
            for pi, lbl in enumerate(["OVERLAY", "CROP"]):
                draw.text((CELL_W + pi * IMG_W + 4, y0 + 4), lbl, fill=(120, 120, 120), font=f_sm)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(str(OUT_SHEET), quality=90)
    print(f"  Contact sheet guardado: {OUT_SHEET}")


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} no encontrado.")
        return

    with open(RESULTS_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"  Registros: {len(rows)}")
    build_sheet(rows)


if __name__ == "__main__":
    main()
