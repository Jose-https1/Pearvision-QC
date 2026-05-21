"""
prepare_quality_roi_masked_previews.py

Audita como entra la imagen al clasificador V2 y genera previews ROI/masked
para preparar el futuro entrenamiento U3.

Pipeline por imagen:
  1. Detectar pera con YOLO (eclpod_v1).
  2. Si no hay deteccion, fallback con rectangulo central.
  3. GrabCut inicializado con bbox → mascara binaria de la pera.
  4. Morfologia (closing + opening + fill holes).
  5. Generar: crop original, fondo negro, fondo gris neutro, fondo blanco.
  6. Guardar metadatos por imagen.

NO entrena ningun modelo.
NO modifica V2 ni datasets anteriores.
"""

import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Rutas ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = (PROJECT_ROOT / "runs" / "detect" / "runs" / "pear_detector"
                / "eclpod_v1" / "weights" / "best.pt")
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "quality_roi_masked_previews"
CROPS_DIR    = OUTPUT_DIR / "crops"

SOURCE_FOLDERS = [
    PROJECT_ROOT / "data" / "unseen_quality_eval_input",
    PROJECT_ROOT / "data" / "unseen_quality_eval_input" / "supermarket_good_batch_v2",
    PROJECT_ROOT / "data" / "unseen_quality_eval_input" / "supermarket_valid_conditions_batch_v3",
    PROJECT_ROOT / "data" / "supermarket_good_hard_examples_v1" / "images",
    PROJECT_ROOT / "data" / "supermarket_good_hard_examples_v2" / "images",
]

PROBLEM_FILES = [
    "1000060792.jpg", "1000060802.jpg", "1000060811.jpg",  # batch V3 false BAD
    "1000060770.jpg", "1000060771.jpg", "1000060773.jpg",  # batch V2 false BAD (azul)
    "1000060774.jpg", "1000060775.jpg",
    "1000060779.jpg", "1000060781.jpg",                    # batch V2 false BAD (negro)
    "1000060747.jpg",                                      # batch V1 false BAD
]

EXTS         = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CROP_SIZE    = 224      # tamaño final de cada crop
NEUTRAL_GRAY = (128, 128, 128)
NEUTRAL_WHITE= (255, 255, 255)
NEUTRAL_BLACK= (0, 0, 0)
MARGIN_RATIO = 0.12     # margen extra alrededor del bbox detectado

# Contact sheet config
THUMB        = 120
PAD          = 5
TEXT_H       = 18
BG_CS        = (20, 20, 20)


# ── Colección de imágenes (deduplicar por filename) ────────────────────────────
def collect_images():
    seen     = {}
    for folder in SOURCE_FOLDERS:
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS and p.name not in seen:
                seen[p.name] = p
    return list(seen.values())


# ── YOLO ───────────────────────────────────────────────────────────────────────
_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
    return _yolo_model


def detect_pear_yolo(img_bgr):
    """Devuelve (x1,y1,x2,y2,conf) en píxeles o None."""
    try:
        model  = get_yolo()
        result = model(img_bgr, verbose=False, conf=0.10)[0]
        if result.boxes and len(result.boxes):
            box  = result.boxes[0]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            return x1, y1, x2, y2, conf
    except Exception as e:
        print(f"    YOLO error: {e}")
    return None


# ── GrabCut ────────────────────────────────────────────────────────────────────
def grabcut_mask(img_bgr, x1, y1, x2, y2):
    """Aplica GrabCut con el rect dado. Devuelve mascara uint8 (0=bg,1=fg)."""
    h, w = img_bgr.shape[:2]
    x1c  = max(0, x1)
    y1c  = max(0, y1)
    x2c  = min(w - 1, x2)
    y2c  = min(h - 1, y2)
    rw   = x2c - x1c
    rh   = y2c - y1c
    if rw < 10 or rh < 10:
        return None
    mask     = np.zeros((h, w), np.uint8)
    bgd      = np.zeros((1, 65), np.float64)
    fgd      = np.zeros((1, 65), np.float64)
    rect     = (x1c, y1c, rw, rh)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    final = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    return final


def clean_mask(mask):
    """Cierre + apertura morfologica + fill de agujeros."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel, iterations=3)
    m = cv2.morphologyEx(m,    cv2.MORPH_OPEN,   kernel, iterations=1)
    # fill holes via flood-fill invertido
    filled = m.copy()
    h, w   = m.shape
    flood  = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 1)
    holes = (filled == 0)
    m[holes] = 1
    return m


def apply_background(img_bgr, mask, color_bgr):
    """Reemplaza el fondo (mask==0) con color_bgr."""
    out = img_bgr.copy()
    out[mask == 0] = color_bgr
    return out


def crop_and_resize(img_bgr, x1, y1, x2, y2, size=CROP_SIZE):
    """Recorta con margen y redimensiona a size×size con padding."""
    h, w   = img_bgr.shape[:2]
    mx     = int((x2 - x1) * MARGIN_RATIO)
    my     = int((y2 - y1) * MARGIN_RATIO)
    cx1    = max(0, x1 - mx)
    cy1    = max(0, y1 - my)
    cx2    = min(w, x2 + mx)
    cy2    = min(h, y2 + my)
    crop   = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return cv2.resize(img_bgr, (size, size))
    # letterbox resize
    ch, cw = crop.shape[:2]
    scale  = size / max(ch, cw)
    nh, nw = int(ch * scale), int(cw * scale)
    resized = cv2.resize(crop, (nw, nh))
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    yoff   = (size - nh) // 2
    xoff   = (size - nw) // 2
    canvas[yoff:yoff+nh, xoff:xoff+nw] = resized
    return canvas


# ── Procesar una imagen ────────────────────────────────────────────────────────
def process_image(img_path: Path):
    """
    Devuelve un dict con todos los metadatos y crops.
    Los crops son ndarrays BGR 224×224.
    """
    result = {
        "filename":            img_path.name,
        "source_path":         str(img_path),
        "found_image":         False,
        "bbox_x1": None, "bbox_y1": None, "bbox_x2": None, "bbox_y2": None,
        "detector_conf":       None,
        "method_used":         "none",
        "mask_ok":             False,
        "mask_area_pct":       None,
        "output_crop_original": None,
        "output_mask":         None,
        "output_neutral_gray": None,
        "output_neutral_white": None,
        "warning":             "",
        # crops como arrays (no se guardan en CSV)
        "_crop_orig":  None,
        "_crop_black": None,
        "_crop_gray":  None,
        "_crop_white": None,
        "_mask_vis":   None,
    }

    if not img_path.exists():
        result["warning"] = "file not found"
        return result

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        result["warning"] = "cv2.imread failed"
        return result

    result["found_image"] = True
    h, w = img_bgr.shape[:2]

    # ── Detección ──────────────────────────────────────────────────────────────
    det = detect_pear_yolo(img_bgr)
    if det:
        x1, y1, x2, y2, conf = det
        result["method_used"]   = "yolo+grabcut"
        result["detector_conf"] = round(conf, 4)
    else:
        # fallback: rectángulo central 70% de la imagen
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x1, y1, x2, y2 = margin_x, margin_y, w - margin_x, h - margin_y
        result["method_used"]   = "grabcut_center_init"
        result["detector_conf"] = None
        result["warning"]       = "no YOLO detection — center rect fallback"

    result["bbox_x1"] = x1
    result["bbox_y1"] = y1
    result["bbox_x2"] = x2
    result["bbox_y2"] = y2

    # ── GrabCut ────────────────────────────────────────────────────────────────
    raw_mask = grabcut_mask(img_bgr, x1, y1, x2, y2)
    if raw_mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 1
        result["warning"] += " grabcut_failed_bbox_mask_used"
    else:
        mask = clean_mask(raw_mask)

    fg_pct = float(mask.sum()) / (h * w) * 100
    result["mask_ok"]       = bool(fg_pct > 5.0)
    result["mask_area_pct"] = round(fg_pct, 2)

    if fg_pct < 5.0:
        result["warning"] += " mask_too_small"

    # ── Generar versiones con fondo neutralizado ───────────────────────────────
    img_black = apply_background(img_bgr, mask, NEUTRAL_BLACK)
    img_gray  = apply_background(img_bgr, mask, NEUTRAL_GRAY)
    img_white = apply_background(img_bgr, mask, NEUTRAL_WHITE)

    # ── Crop y resize ─────────────────────────────────────────────────────────
    crop_orig  = crop_and_resize(img_bgr,   x1, y1, x2, y2)
    crop_black = crop_and_resize(img_black, x1, y1, x2, y2)
    crop_gray  = crop_and_resize(img_gray,  x1, y1, x2, y2)
    crop_white = crop_and_resize(img_white, x1, y1, x2, y2)

    # Máscara visual
    mask_vis = cv2.resize((mask * 255).astype(np.uint8), (CROP_SIZE, CROP_SIZE))
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # ── Guardar archivos individuales ──────────────────────────────────────────
    stem = img_path.stem
    p_orig  = CROPS_DIR / f"{stem}_original.jpg"
    p_mask  = CROPS_DIR / f"{stem}_mask.jpg"
    p_gray  = CROPS_DIR / f"{stem}_gray_bg.jpg"
    p_white = CROPS_DIR / f"{stem}_white_bg.jpg"

    cv2.imwrite(str(p_orig),  crop_orig)
    cv2.imwrite(str(p_mask),  mask_vis)
    cv2.imwrite(str(p_gray),  crop_gray)
    cv2.imwrite(str(p_white), crop_white)

    result["output_crop_original"] = str(p_orig)
    result["output_mask"]          = str(p_mask)
    result["output_neutral_gray"]  = str(p_gray)
    result["output_neutral_white"] = str(p_white)
    result["_crop_orig"]           = crop_orig
    result["_crop_black"]          = crop_black
    result["_crop_gray"]           = crop_gray
    result["_crop_white"]          = crop_white
    result["_mask_vis"]            = mask_vis

    return result


# ── Contact sheet original vs masked ──────────────────────────────────────────
def bgr_to_pil(arr):
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def make_thumb_from_arr(arr, label, color=(200, 200, 200)):
    pil  = bgr_to_pil(arr).resize((THUMB, THUMB))
    cell = Image.new("RGB", (THUMB, THUMB + TEXT_H), BG_CS)
    cell.paste(pil, (0, 0))
    draw = ImageDraw.Draw(cell)
    try:
        font = ImageFont.truetype("arial.ttf", 8)
    except Exception:
        font = ImageFont.load_default()
    draw.text((2, THUMB + 2), label, fill=color, font=font)
    return cell


def make_contact_sheet_rows(records, out_path):
    """
    Una fila por imagen, 5 columnas:
    original full | crop original | mask | crop gray bg | crop white bg
    """
    COLS   = 5
    hdr_h  = 26
    cell_w = THUMB + PAD
    cell_h = THUMB + TEXT_H + PAD
    W = COLS * cell_w + PAD
    H = len(records) * cell_h + PAD + hdr_h

    sheet = Image.new("RGB", (W, H), BG_CS)
    draw  = ImageDraw.Draw(sheet)
    try:
        font_hdr = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font_hdr = ImageFont.load_default()

    col_labels = ["ORIGINAL", "CROP", "MASK", "GRAY BG", "WHITE BG"]
    for ci, lbl in enumerate(col_labels):
        draw.text((PAD + ci * cell_w + 2, 6), lbl, fill=(200, 200, 200), font=font_hdr)

    for ri, rec in enumerate(records):
        y = hdr_h + PAD + ri * cell_h
        fname = rec["filename"][:16]
        method_color = (80, 200, 80) if rec["method_used"].startswith("yolo") else (200, 140, 40)
        mask_color   = (80, 200, 80) if rec["mask_ok"] else (200, 60, 60)

        # Columna 0: original (carga la imagen directamente)
        try:
            orig_full = cv2.imread(str(rec["source_path"]))
            orig_full = cv2.resize(orig_full, (THUMB, THUMB))
            c0 = make_thumb_from_arr(orig_full, fname, color=(180, 180, 180))
        except Exception:
            c0 = Image.new("RGB", (THUMB, THUMB + TEXT_H), (60, 60, 60))

        c1 = make_thumb_from_arr(rec["_crop_orig"],  rec["method_used"][:14], color=method_color)
        c2 = make_thumb_from_arr(rec["_mask_vis"],   f"{rec['mask_area_pct']:.0f}%", color=mask_color)
        c3 = make_thumb_from_arr(rec["_crop_gray"],  "gray_bg", color=(150, 200, 220))
        c4 = make_thumb_from_arr(rec["_crop_white"], "white_bg", color=(220, 220, 180))

        for ci, cell in enumerate([c0, c1, c2, c3, c4]):
            sheet.paste(cell, (PAD + ci * cell_w, y))

    sheet.save(out_path, quality=88)
    print(f"  [OK] {out_path.name}  ({len(records)} filas)")


# ── Problem cases grid ─────────────────────────────────────────────────────────
def make_problem_grid(records_by_name, out_path):
    """
    Una fila por caso problema, 5 columnas como contact sheet principal.
    """
    rows = []
    for fn in PROBLEM_FILES:
        if fn in records_by_name:
            rows.append(records_by_name[fn])
        else:
            print(f"  AVISO problem grid: {fn} no encontrado — se omite")

    if not rows:
        print("  AVISO: ningún caso problema encontrado")
        return

    COLS   = 5
    hdr_h  = 26
    cell_w = THUMB + PAD
    cell_h = THUMB + TEXT_H + PAD
    W = COLS * cell_w + PAD
    H = len(rows) * cell_h + PAD + hdr_h

    sheet = Image.new("RGB", (W, H), BG_CS)
    draw  = ImageDraw.Draw(sheet)
    try:
        font_hdr = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font_hdr = ImageFont.load_default()

    col_labels = ["ORIGINAL", "CROP", "MASK", "GRAY BG", "WHITE BG"]
    for ci, lbl in enumerate(col_labels):
        draw.text((PAD + ci * cell_w + 2, 6), lbl, fill=(220, 100, 80), font=font_hdr)

    for ri, rec in enumerate(rows):
        y = hdr_h + PAD + ri * cell_h
        fname = rec["filename"][:16]
        method_color = (80, 200, 80) if rec["method_used"].startswith("yolo") else (200, 140, 40)
        mask_color   = (80, 200, 80) if rec["mask_ok"] else (200, 60, 60)

        try:
            orig_full = cv2.imread(str(rec["source_path"]))
            orig_full = cv2.resize(orig_full, (THUMB, THUMB))
            c0 = make_thumb_from_arr(orig_full, fname, color=(220, 100, 80))
        except Exception:
            c0 = Image.new("RGB", (THUMB, THUMB + TEXT_H), (60, 60, 60))

        c1 = make_thumb_from_arr(rec["_crop_orig"],  rec["method_used"][:14], color=method_color)
        c2 = make_thumb_from_arr(rec["_mask_vis"],   f"{rec['mask_area_pct']:.0f}%", color=mask_color)
        c3 = make_thumb_from_arr(rec["_crop_gray"],  "gray_bg", color=(150, 200, 220))
        c4 = make_thumb_from_arr(rec["_crop_white"], "white_bg", color=(220, 220, 180))

        for ci, cell in enumerate([c0, c1, c2, c3, c4]):
            sheet.paste(cell, (PAD + ci * cell_w, y))

    sheet.save(out_path, quality=88)
    print(f"  [OK] {out_path.name}  ({len(rows)} casos problema)")


# ── CSV de diagnóstico ────────────────────────────────────────────────────────
def write_diagnostics_csv(records):
    out = OUTPUT_DIR / "roi_masked_diagnostics.csv"
    fields = [
        "filename", "source_path", "found_image",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "detector_conf", "method_used",
        "mask_ok", "mask_area_pct",
        "output_crop_original", "output_mask",
        "output_neutral_gray", "output_neutral_white",
        "warning",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  [OK] {out.name}  ({len(records)} filas)")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = collect_images()
    print("=== prepare_quality_roi_masked_previews ===")
    print(f"  Imagenes unicas recopiladas: {len(img_paths)}")
    print(f"  YOLO weights: {YOLO_WEIGHTS}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Pre-cargar YOLO
    print("  Cargando modelo YOLO ...")
    get_yolo()
    print("  YOLO listo.")
    print()

    records      = []
    records_by_name = {}
    for i, p in enumerate(img_paths, start=1):
        print(f"  [{i:>3}/{len(img_paths)}] {p.name:<32}", end="  ")
        rec = process_image(p)
        tag = rec["method_used"][:12]
        msk = f"mask={rec['mask_area_pct']:.0f}%" if rec["mask_area_pct"] is not None else "mask=?"
        wrn = f"  AVISO: {rec['warning']}" if rec["warning"] else ""
        print(f"{tag:<14}  {msk}{wrn}")
        records.append(rec)
        records_by_name[p.name] = rec

    valid_records = [r for r in records if r["_crop_orig"] is not None]
    print(f"\n  Procesadas OK: {len(valid_records)}/{len(records)}")
    print()

    print("  Generando contact_sheet_original_vs_masked.jpg ...")
    make_contact_sheet_rows(valid_records,
                            OUTPUT_DIR / "contact_sheet_original_vs_masked.jpg")

    print("  Generando problem_cases_grid.jpg ...")
    make_problem_grid(records_by_name,
                      OUTPUT_DIR / "problem_cases_grid.jpg")

    print("  Generando roi_masked_diagnostics.csv ...")
    write_diagnostics_csv(records)

    # Estadísticas rápidas
    n_yolo   = sum(1 for r in records if "yolo" in r["method_used"])
    n_center = sum(1 for r in records if "center" in r["method_used"])
    n_mask_ok= sum(1 for r in records if r["mask_ok"])
    areas    = [r["mask_area_pct"] for r in records if r["mask_area_pct"] is not None]
    mean_area= sum(areas) / len(areas) if areas else 0

    print()
    print("=" * 60)
    print("ROI MASKED QUALITY PIPELINE PREP COMPLETADO")
    print()
    print(f"  Imagenes procesadas:     {len(records)}")
    print(f"  Detectadas con YOLO:     {n_yolo}")
    print(f"  Fallback centro:         {n_center}")
    print(f"  Mascara OK (>5% fg):     {n_mask_ok}")
    print(f"  Area media mascara:      {mean_area:.1f}%")
    print()
    print("No se entrenó ningún modelo.")
    print("No se modificó V2.")
    print("No se modificó analyze_quality.py.")
    print("No se modificó quality_rules.yaml.")
    print()
    print("Archivos principales:")
    print(f"- reports/quality_v2_input_audit_report.md")
    print(f"- {OUTPUT_DIR / 'contact_sheet_original_vs_masked.jpg'}")
    print(f"- {OUTPUT_DIR / 'problem_cases_grid.jpg'}")
    print(f"- {OUTPUT_DIR / 'roi_masked_diagnostics.csv'}")
    print(f"- reports/quality_roi_masked_u3_plan.md")
    print()
    print("Siguiente paso:")
    print("José debe abrir contact_sheet_original_vs_masked.jpg y problem_cases_grid.jpg")
    print("para confirmar si la máscara/contorno de la pera es correcto.")
    print("Si las máscaras son buenas, el siguiente paso será entrenar U3 usando")
    print("imágenes ROI/masked y hard examples GOOD.")


if __name__ == "__main__":
    main()
