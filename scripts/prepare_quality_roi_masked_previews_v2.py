"""
prepare_quality_roi_masked_previews_v2.py

Mejora el pipeline ROI/masked V1 con:
  Metodo A: GrabCut + LCC + fill holes + closing suave
  Metodo B: A + eliminacion de pixeles borde similares al fondo (sombras/halos)
  Metodo C: B + erosion conservadora + limpieza de componentes pequeños

Estrategia para evitar re-proceso pesado:
  - Para las 47 imagenes que ya tienen _original.jpg en crops/ V1,
    usa ese crop 224x224 directamente como entrada (pera ya centrada).
  - Para las 17 imagenes restantes (batch_v3 sin procesar),
    carga desde fuente original usando PIL (maneja rutas acentuadas).

NO entrena ningun modelo.
NO modifica V2.
NO borra outputs anteriores.
"""

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Rutas ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_CROPS     = PROJECT_ROOT / "outputs" / "quality_roi_masked_previews"  / "crops"
OUT_DIR      = PROJECT_ROOT / "outputs" / "quality_roi_masked_previews_v2"
CROPS_V2     = OUT_DIR / "crops"
REPORTS_DIR  = PROJECT_ROOT / "reports"

SOURCE_FOLDERS = [
    PROJECT_ROOT / "data" / "unseen_quality_eval_input",
    PROJECT_ROOT / "data" / "unseen_quality_eval_input" / "supermarket_good_batch_v2",
    PROJECT_ROOT / "data" / "unseen_quality_eval_input" / "supermarket_valid_conditions_batch_v3",
    PROJECT_ROOT / "data" / "supermarket_good_hard_examples_v1" / "images",
    PROJECT_ROOT / "data" / "supermarket_good_hard_examples_v2" / "images",
]

PROBLEM_IDS = [
    "1000060747",
    "1000060792", "1000060802", "1000060811",
    "1000060770", "1000060771", "1000060773",
    "1000060774", "1000060775",
    "1000060779", "1000060781",
]

EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CROP_SIZE = 224
NEUTRAL_GRAY  = (128, 128, 128)
NEUTRAL_WHITE = (255, 255, 255)

# Contact sheet config
THUMB   = 130
PAD     = 5
TEXT_H  = 18
BG_DARK = (18, 18, 18)


# ── Imagen I/O (siempre PIL para rutas acentuadas) ────────────────────────────
def pil_load_bgr(path: Path):
    """Carga imagen con PIL y devuelve numpy BGR (seguro con rutas acentuadas)."""
    try:
        pil = Image.open(str(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def bgr_save(path: Path, arr: np.ndarray, quality=92):
    """Guarda numpy BGR con PIL (seguro con rutas acentuadas)."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(str(path), quality=quality)


def rgba_save(path: Path, bgr: np.ndarray, mask: np.ndarray):
    """Guarda PNG RGBA: fondo transparente donde mask==0."""
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb).convert("RGBA")
    alpha = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    pil.putalpha(alpha)
    pil.save(str(path))


# ── Colectar imágenes ─────────────────────────────────────────────────────────
def collect_all_ids():
    """
    Devuelve dict {base_id: source_path}.
    Prioriza el _original.jpg de V1 como fuente si existe.
    Para los que no tienen V1 crop, busca en fuentes originales.
    """
    ids = {}
    # Imágenes con crop V1
    for f in sorted(V1_CROPS.glob("*_original.jpg")):
        base = f.stem.replace("_original", "")
        ids[base] = ("v1_crop", f)
    # Buscar imágenes sin crop V1
    for folder in SOURCE_FOLDERS:
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS:
                base = p.stem
                if base not in ids:
                    ids[base] = ("source", p)
    return ids


# ── Segmentación: Método A ────────────────────────────────────────────────────
def grabcut_center(img_bgr, margin=0.18):
    """GrabCut con rect central. Devuelve mascara uint8 binaria."""
    h, w = img_bgr.shape[:2]
    x1 = int(w * margin);  y1 = int(h * margin)
    x2 = int(w * (1 - margin)); y2 = int(h * (1 - margin))
    rw = max(10, x2 - x1);  rh = max(10, y2 - y1)
    mask = np.zeros((h, w), np.uint8)
    bgd  = np.zeros((1, 65), np.float64)
    fgd  = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, (x1, y1, rw, rh), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # Fallback: todo el interior del rect
        m = np.zeros((h, w), np.uint8)
        m[y1:y2, x1:x2] = 1
        return m
    return np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)


def keep_largest_component(mask):
    """Conserva solo el componente conectado principal."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    sizes = stats[1:, cv2.CC_STAT_AREA]
    main  = int(np.argmax(sizes)) + 1
    return (labels == main).astype(np.uint8)


def fill_holes(mask):
    """Rellena huecos internos via flood-fill invertido."""
    h, w  = mask.shape
    flooded = mask.copy()
    border = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flooded, border, (0, 0), 1)
    holes = (flooded == 0)
    filled = mask.copy()
    filled[holes] = 1
    return filled


def method_a(img_bgr):
    """GrabCut + LCC + fill holes + closing suave."""
    raw = grabcut_center(img_bgr)
    m   = keep_largest_component(raw)
    m   = fill_holes(m)
    k7  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m   = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7, iterations=2)
    k3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m   = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k3, iterations=1)
    return m


# ── Método B: limpieza de sombras/halos ───────────────────────────────────────
def estimate_bg_color_bgr(img_bgr, corner=12):
    """Color medio del fondo muestreando 4 esquinas."""
    h, w = img_bgr.shape[:2]
    s = min(corner, h // 6, w // 6)
    corners = [
        img_bgr[:s, :s], img_bgr[:s, w-s:],
        img_bgr[h-s:, :s], img_bgr[h-s:, w-s:],
    ]
    pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return pixels.mean(axis=0).astype(np.float32)


def color_dist_from_bg(img_bgr, bg_bgr):
    """Distancia euclídea en espacio BGR de cada pixel al color de fondo."""
    diff = img_bgr.astype(np.float32) - bg_bgr.astype(np.float32)
    return np.sqrt((diff ** 2).sum(axis=2))


def method_b(img_bgr, mask_a, tol=38, border_px=6):
    """
    Elimina del borde de mask_a los píxeles demasiado parecidos al fondo.
    Preserva el interior de la pera intacto.
    """
    bg   = estimate_bg_color_bgr(img_bgr)
    dist = color_dist_from_bg(img_bgr, bg)

    # Zona borde de la máscara (dentro de la máscara pero cerca del contorno)
    k_b  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_px * 2 + 1,) * 2)
    eroded = cv2.erode(mask_a, k_b, iterations=1)
    border = (mask_a > 0) & (eroded == 0)

    cleaned = mask_a.copy()
    cleaned[border & (dist < tol)] = 0

    # Re-aplicar LCC y fill holes tras limpieza
    cleaned = keep_largest_component(cleaned)
    cleaned = fill_holes(cleaned)
    return cleaned


# ── Método C: máscara conservadora ────────────────────────────────────────────
def method_c(mask_b, erode_px=2):
    """Erosión ligera + limpieza de componentes aislados + re-dilatación mínima."""
    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
    m   = cv2.erode(mask_b, k_e, iterations=1)
    # Eliminar componentes pequeños
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep  = np.where(areas >= max(100, areas.max() * 0.05))[0] + 1
        clean = np.zeros_like(m)
        for lbl in keep:
            clean[labels == lbl] = 1
        m = clean
    # Dilatación leve para no perder borde real
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m   = cv2.dilate(m, k_d, iterations=1)
    m   = fill_holes(m)
    return m


# ── Aplicar fondo neutro y crop ───────────────────────────────────────────────
def apply_bg(img_bgr, mask, color_bgr):
    out = img_bgr.copy()
    out[mask == 0] = color_bgr
    return out


def letterbox_resize(img_bgr, size=CROP_SIZE, pad_color=(128, 128, 128)):
    h, w   = img_bgr.shape[:2]
    scale  = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    yo = (size - nh) // 2;  xo = (size - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = r
    return canvas


def mask_resize(mask, size=CROP_SIZE):
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


# ── Procesar una imagen ────────────────────────────────────────────────────────
def process_image(base_id, src_type, src_path):
    rec = {
        "filename_base": base_id,
        "source_path":   str(src_path),
        "found_image":   False,
        "bbox_method":   "none",
        "detector_conf": "",
        "mask_v1_area_pct":   "",
        "mask_clean_area_pct": "",
        "area_change_pct":    "",
        "clean_status":  "FAIL",
        "warning":       "",
        "output_gray_bg_clean":  "",
        "output_white_bg_clean": "",
    }

    # ── Cargar imagen ──────────────────────────────────────────────────────────
    img_bgr = pil_load_bgr(src_path)
    if img_bgr is None:
        rec["warning"] = "load_failed"
        return rec, None

    rec["found_image"] = True

    # Redimensionar si es necesario (V1 crops ya son 224x224)
    if img_bgr.shape[0] != CROP_SIZE or img_bgr.shape[1] != CROP_SIZE:
        img_bgr = letterbox_resize(img_bgr)
        rec["bbox_method"] = "source_letterbox+grabcut_center"
    else:
        rec["bbox_method"] = "v1_crop_reuse+grabcut_center"

    # ── Métodos A → B → C ─────────────────────────────────────────────────────
    mask_a = method_a(img_bgr)
    mask_b = method_b(img_bgr, mask_a)
    mask_c = method_c(mask_b)

    h, w = img_bgr.shape[:2]
    total = h * w

    area_a = float(mask_a.sum()) / total * 100
    area_c = float(mask_c.sum()) / total * 100
    change = area_a - area_c

    rec["mask_v1_area_pct"]    = round(area_a, 2)
    rec["mask_clean_area_pct"] = round(area_c, 2)
    rec["area_change_pct"]     = round(change, 2)

    if area_c < 5.0:
        rec["clean_status"] = "FAIL"
        rec["warning"]      = "mask_too_small_after_cleanup"
    elif change > 35.0:
        rec["clean_status"] = "REVIEW"
        rec["warning"]      = f"large_area_reduction_{change:.1f}pct"
    else:
        rec["clean_status"] = "OK"

    # ── Versiones con fondo ────────────────────────────────────────────────────
    gray_bg  = apply_bg(img_bgr, mask_c, NEUTRAL_GRAY)
    white_bg = apply_bg(img_bgr, mask_c, NEUTRAL_WHITE)

    # Máscara visual
    mask_a_vis = cv2.cvtColor((mask_a * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_c_vis = cv2.cvtColor((mask_c * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Comparación horizontal: original | mask_a | mask_c | gray | white
    strip = np.concatenate([img_bgr, mask_a_vis, mask_c_vis, gray_bg, white_bg], axis=1)

    # Recuperar máscara V1 si existe (para referencia en contact sheet)
    v1_mask_path = V1_CROPS / f"{base_id}_mask.jpg"
    if v1_mask_path.exists():
        v1_mask_arr = pil_load_bgr(v1_mask_path)
    else:
        v1_mask_arr = np.zeros_like(img_bgr)

    # ── Guardar ───────────────────────────────────────────────────────────────
    stem = base_id
    paths = {
        "original":     CROPS_V2 / f"{stem}_original.jpg",
        "mask_v1_like": CROPS_V2 / f"{stem}_mask_v1_like.jpg",
        "mask_clean":   CROPS_V2 / f"{stem}_mask_clean.jpg",
        "gray_bg_clean":  CROPS_V2 / f"{stem}_gray_bg_clean.jpg",
        "white_bg_clean": CROPS_V2 / f"{stem}_white_bg_clean.jpg",
        "transparent":    CROPS_V2 / f"{stem}_transparent_debug.png",
        "comparison":     CROPS_V2 / f"{stem}_comparison.jpg",
    }

    bgr_save(paths["original"],     img_bgr)
    bgr_save(paths["mask_v1_like"], v1_mask_arr)
    bgr_save(paths["mask_clean"],   mask_c_vis)
    bgr_save(paths["gray_bg_clean"],  gray_bg)
    bgr_save(paths["white_bg_clean"], white_bg)
    bgr_save(paths["comparison"],   strip)
    try:
        rgba_save(paths["transparent"], img_bgr, mask_resize(mask_c))
    except Exception:
        paths["transparent"] = None

    rec["output_gray_bg_clean"]  = str(paths["gray_bg_clean"])
    rec["output_white_bg_clean"] = str(paths["white_bg_clean"])

    return rec, {
        "base_id":    base_id,
        "img":        img_bgr,
        "mask_a_vis": mask_a_vis,
        "mask_c_vis": mask_c_vis,
        "gray_bg":    gray_bg,
        "white_bg":   white_bg,
        "v1_mask":    v1_mask_arr,
        "paths":      paths,
        "status":     rec["clean_status"],
    }


# ── Contact sheet comparativo ─────────────────────────────────────────────────
def get_font(size=8):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def arr_to_thumb(arr, size=THUMB):
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).resize((size, size))


def pil_thumb(path: Path, size=THUMB):
    try:
        return Image.open(str(path)).convert("RGB").resize((size, size))
    except Exception:
        ph = Image.new("RGB", (size, size), (50, 30, 30))
        return ph


def build_contact_sheet_v2(data_rows, out_path, title="ROI Masked V2"):
    """
    6 columnas: ID | original | mask_v1_like | mask_clean | gray_bg_clean | white_bg_clean
    """
    LABEL_W  = 88
    hdr_h    = 28
    cell_w   = THUMB + PAD
    cell_h   = THUMB + TEXT_H + PAD
    N_COLS   = 5
    W = LABEL_W + PAD + N_COLS * cell_w + PAD
    H = hdr_h + PAD + len(data_rows) * cell_h + PAD

    sheet = Image.new("RGB", (W, H), BG_DARK)
    draw  = ImageDraw.Draw(sheet)
    hdr   = get_font(9)
    draw.text((PAD, 7), title, fill=(210, 210, 210), font=hdr)

    col_labels = ["ID", "ORIGINAL", "MASK_V1", "MASK_CLEAN", "GRAY_BG", "WHITE_BG"]
    col_colors = [(160,160,160),(180,180,180),(160,200,160),(100,180,220),(130,200,220),(220,210,160)]
    col_x = [PAD] + [PAD + LABEL_W + PAD + i * cell_w for i in range(N_COLS)]
    for cx, lbl, cc in zip(col_x, col_labels, col_colors):
        draw.text((cx + 2, hdr_h - 14), lbl, fill=cc, font=get_font(7))

    for ri, r in enumerate(data_rows):
        y     = hdr_h + PAD + ri * cell_h
        bid   = r["base_id"]
        status_color = (80,200,80) if r["status"] == "OK" else (200,140,40) if r["status"] == "REVIEW" else (200,60,60)

        # Label
        label_cell = Image.new("RGB", (LABEL_W, THUMB + TEXT_H), BG_DARK)
        ld = ImageDraw.Draw(label_cell)
        ld.text((2, 4),  bid[:14], fill=status_color, font=get_font(7))
        ld.text((2, 14), r["status"][:6], fill=status_color, font=get_font(6))
        sheet.paste(label_cell, (PAD, y))

        thumbs = [
            arr_to_thumb(r["img"]),
            arr_to_thumb(r["v1_mask"]),
            arr_to_thumb(r["mask_c_vis"]),
            arr_to_thumb(r["gray_bg"]),
            arr_to_thumb(r["white_bg"]),
        ]
        labels_t = ["original","mask_v1","mask_clean","gray_bg","white_bg"]
        for ci, (th, lt) in enumerate(zip(thumbs, labels_t)):
            x    = PAD + LABEL_W + PAD + ci * cell_w
            cell = Image.new("RGB", (THUMB, THUMB + TEXT_H), BG_DARK)
            cell.paste(th, (0, 0))
            ImageDraw.Draw(cell).text((2, THUMB + 2), lt[:10], fill=(160,160,160), font=get_font(6))
            sheet.paste(cell, (x, y))

    sheet.save(str(out_path), quality=88)
    print(f"  [OK] {out_path.name}  ({len(data_rows)} filas)")


def build_problem_grid(data_by_id, out_path):
    rows = []
    for pid in PROBLEM_IDS:
        if pid in data_by_id:
            rows.append(data_by_id[pid])
        else:
            print(f"  AVISO problem grid: {pid} no encontrado — se omite")
    if not rows:
        return
    build_contact_sheet_v2(rows, out_path,
                            title="Problem Cases — False BAD Diagnosed (V2 masks)")
    print(f"  [OK] {out_path.name}  ({len(rows)}/{len(PROBLEM_IDS)} casos)")


# ── CSV diagnóstico ───────────────────────────────────────────────────────────
def write_csv(records):
    out = OUT_DIR / "roi_masked_v2_diagnostics.csv"
    fields = [
        "filename_base", "source_path", "found_image",
        "bbox_method", "detector_conf",
        "mask_v1_area_pct", "mask_clean_area_pct", "area_change_pct",
        "clean_status", "warning",
        "output_gray_bg_clean", "output_white_bg_clean",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    ok  = sum(1 for r in records if r["clean_status"] == "OK")
    rev = sum(1 for r in records if r["clean_status"] == "REVIEW")
    fail= sum(1 for r in records if r["clean_status"] == "FAIL")
    print(f"  [OK] {out.name}  (OK={ok}, REVIEW={rev}, FAIL={fail})")
    return ok, rev, fail


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_V2.mkdir(parents=True, exist_ok=True)

    all_ids = collect_all_ids()
    print("=== prepare_quality_roi_masked_previews_v2 ===")
    print(f"  Imagenes a procesar: {len(all_ids)}")
    v1_reuse = sum(1 for t, _ in all_ids.values() if t == "v1_crop")
    print(f"  Desde V1 crop (224×224 ya): {v1_reuse}")
    print(f"  Desde fuente original:       {len(all_ids) - v1_reuse}")
    print()

    records  = []
    data_rows = []
    data_by_id = {}

    for i, (base_id, (src_type, src_path)) in enumerate(sorted(all_ids.items()), 1):
        print(f"  [{i:>3}/{len(all_ids)}] {base_id:<14}", end="  ")
        rec, data = process_image(base_id, src_type, src_path)
        tag = rec["clean_status"]
        area_c = rec["mask_clean_area_pct"]
        chg    = rec["area_change_pct"]
        wrn    = f"  ⚠ {rec['warning']}" if rec["warning"] else ""
        print(f"{tag:<6}  area={area_c}%  chg={chg}%{wrn}")
        records.append(rec)
        if data:
            data_rows.append(data)
            data_by_id[base_id] = data

    print(f"\n  Procesadas: {len(data_rows)}/{len(all_ids)}")
    print()

    print("  Generando contact_sheet_v1_vs_v2.jpg ...")
    build_contact_sheet_v2(data_rows,
                            OUT_DIR / "contact_sheet_v1_vs_v2.jpg",
                            title="ROI Masked V1 vs V2 — comparativa de mascaras")

    print("  Generando problem_cases_v2_grid.jpg ...")
    build_problem_grid(data_by_id, OUT_DIR / "problem_cases_v2_grid.jpg")

    print("  Generando roi_masked_v2_diagnostics.csv ...")
    ok, rev, fail = write_csv(records)

    print()
    print(f"  OK={ok}  REVIEW={rev}  FAIL={fail}")
    print()
    print("=== DONE (V2 preprocessor) ===")
    return len(data_rows), ok, rev, fail


if __name__ == "__main__":
    main()
