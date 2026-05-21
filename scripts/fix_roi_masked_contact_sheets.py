"""
fix_roi_masked_contact_sheets.py

Regenera contact_sheet_original_vs_masked.jpg, problem_cases_grid.jpg y
roi_masked_diagnostics.csv usando los crops ya existentes en crops/.

Para el problem_cases_grid, si faltan crops de casos problema, los genera
con GrabCut ligero (sin YOLO) para no repetir el trabajo pesado completo.

NO entrena ningun modelo.
NO modifica V2 ni datasets anteriores.
NO borra crops existentes.
"""

import csv
import shutil
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
PREVIEWS_DIR = PROJECT_ROOT / "outputs" / "quality_roi_masked_previews"
CROPS_DIR    = PREVIEWS_DIR / "crops"
REPORTS_DIR  = PROJECT_ROOT / "reports"

# Carpetas donde buscar originales de los casos problema
SOURCE_FOLDERS = [
    PROJECT_ROOT / "data" / "unseen_quality_eval_input",
    PROJECT_ROOT / "data" / "unseen_quality_eval_input" / "supermarket_good_batch_v2",
    PROJECT_ROOT / "data" / "unseen_quality_eval_input" / "supermarket_valid_conditions_batch_v3",
    PROJECT_ROOT / "data" / "supermarket_good_hard_examples_v1" / "images",
    PROJECT_ROOT / "data" / "supermarket_good_hard_examples_v2" / "images",
]

PROBLEM_IDS = [
    "1000060792", "1000060802", "1000060811",   # batch V3 falsos BAD
    "1000060770", "1000060771", "1000060773",   # batch V2 falsos BAD (azul)
    "1000060774", "1000060775",
    "1000060779", "1000060781",                 # batch V2 falsos BAD (negro)
    "1000060747",                               # batch V1 falso BAD
]

# Contact sheet config
THUMB   = 128
PAD     = 6
TEXT_H  = 20
BG_DARK = (20, 20, 20)
NEUTRAL_GRAY  = (128, 128, 128)
NEUTRAL_WHITE = (255, 255, 255)
CROP_SIZE = 224


# ── TAREA 1+2: Inspeccionar y agrupar crops existentes ───────────────────────
def collect_existing_groups():
    """Lee crops/ y agrupa por stem base."""
    VARIANTS = {"_original": "original", "_mask": "mask",
                "_gray_bg": "gray_bg", "_white_bg": "white_bg"}
    groups = defaultdict(dict)

    if not CROPS_DIR.exists():
        print("  AVISO: crops/ no existe")
        return {}

    for f in sorted(CROPS_DIR.glob("*.jpg")):
        stem = f.stem
        for suffix, key in VARIANTS.items():
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                groups[base][key] = f
                break

    return dict(groups)


def find_original_image(base_id):
    """Busca el JPG original en las carpetas fuente."""
    for folder in SOURCE_FOLDERS:
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            p = folder / f"{base_id}{ext}"
            if p.exists():
                return p
    return None


# ── GrabCut ligero (sin YOLO) para casos problema faltantes ───────────────────
def grabcut_center(img_bgr, margin=0.15):
    """GrabCut inicializado con rectángulo central (sin detector)."""
    h, w = img_bgr.shape[:2]
    x1 = int(w * margin);  y1 = int(h * margin)
    x2 = int(w * (1 - margin)); y2 = int(h * (1 - margin))
    rw, rh = x2 - x1, y2 - y1
    if rw < 10 or rh < 10:
        return None, x1, y1, x2, y2
    mask = np.zeros((h, w), np.uint8)
    bgd  = np.zeros((1, 65), np.float64)
    fgd  = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, (x1, y1, rw, rh), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, x1, y1, x2, y2
    final = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    # morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final  = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel, iterations=3)
    final  = cv2.morphologyEx(final, cv2.MORPH_OPEN,  kernel, iterations=1)
    return final, x1, y1, x2, y2


def crop_resize(img_bgr, x1, y1, x2, y2, size=CROP_SIZE, margin=0.10):
    h, w = img_bgr.shape[:2]
    mx = int((x2 - x1) * margin); my = int((y2 - y1) * margin)
    cx1 = max(0, x1 - mx); cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx); cy2 = min(h, y2 + my)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return cv2.resize(img_bgr, (size, size))
    ch, cw = crop.shape[:2]
    scale  = size / max(ch, cw)
    nh, nw = int(ch * scale), int(cw * scale)
    r = cv2.resize(crop, (nw, nh))
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    yo = (size - nh) // 2; xo = (size - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = r
    return canvas


def load_image_bgr(path: Path):
    """Carga imagen como numpy BGR usando PIL para evitar problemas con rutas acentuadas."""
    try:
        pil = Image.open(str(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None


def generate_problem_crops(base_id):
    """
    Genera los 4 crops para un caso problema usando GrabCut-solo (sin YOLO).
    Guarda en crops/ y devuelve dict con paths.
    """
    src = find_original_image(base_id)
    if src is None:
        return None, f"original image not found for {base_id}"

    img_bgr = load_image_bgr(src)
    if img_bgr is None:
        return None, f"load failed for {src}"

    mask, x1, y1, x2, y2 = grabcut_center(img_bgr)

    if mask is None:
        mask = np.ones(img_bgr.shape[:2], np.uint8)

    img_gray  = img_bgr.copy(); img_gray[mask == 0]  = NEUTRAL_GRAY
    img_white = img_bgr.copy(); img_white[mask == 0] = NEUTRAL_WHITE

    crop_orig  = crop_resize(img_bgr,   x1, y1, x2, y2)
    crop_gray  = crop_resize(img_gray,  x1, y1, x2, y2)
    crop_white = crop_resize(img_white, x1, y1, x2, y2)
    mask_vis   = cv2.resize((mask * 255).astype(np.uint8), (CROP_SIZE, CROP_SIZE))
    mask_vis   = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    p_orig  = CROPS_DIR / f"{base_id}_original.jpg"
    p_mask  = CROPS_DIR / f"{base_id}_mask.jpg"
    p_gray  = CROPS_DIR / f"{base_id}_gray_bg.jpg"
    p_white = CROPS_DIR / f"{base_id}_white_bg.jpg"

    cv2.imwrite(str(p_orig),  crop_orig)
    cv2.imwrite(str(p_mask),  mask_vis)
    cv2.imwrite(str(p_gray),  crop_gray)
    cv2.imwrite(str(p_white), crop_white)

    return {
        "original": p_orig, "mask": p_mask,
        "gray_bg": p_gray,  "white_bg": p_white,
    }, ""


# ── Helpers PIL ────────────────────────────────────────────────────────────────
def load_thumb(path, size=THUMB):
    """Carga JPG como PIL thumb."""
    try:
        return Image.open(path).convert("RGB").resize((size, size))
    except Exception:
        ph = Image.new("RGB", (size, size), (60, 60, 60))
        ImageDraw.Draw(ph).text((4, size // 2 - 5), "ERR", fill=(200, 60, 60))
        return ph


def get_font(size=8):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_label_cell(text, w, h, color=(180, 180, 180)):
    cell = Image.new("RGB", (w, h), BG_DARK)
    draw = ImageDraw.Draw(cell)
    draw.text((3, h // 2 - 5), text[:18], fill=color, font=get_font(8))
    return cell


# ── TAREA 4: contact sheet general ────────────────────────────────────────────
def build_contact_sheet(groups, out_path):
    """
    5 columnas: ID-label | ORIGINAL | MASK | GRAY_BG | WHITE_BG
    Una fila por grupo de imagen.
    """
    LABEL_W = 90
    COLS    = 4          # 4 imágenes + 1 label = 5 celdas visibles
    hdr_h   = 28
    cell_w  = THUMB + PAD
    cell_h  = THUMB + TEXT_H + PAD
    W = LABEL_W + PAD + COLS * cell_w + PAD
    H = hdr_h + PAD + len(groups) * cell_h + PAD

    sheet = Image.new("RGB", (W, H), BG_DARK)
    draw  = ImageDraw.Draw(sheet)
    font_hdr = get_font(10)
    draw.text((PAD, 8), "ROI / Masked Quality Previews", fill=(220, 220, 220), font=font_hdr)
    # cabeceras columnas
    col_labels = ["ID", "ORIGINAL", "MASK", "GRAY_BG", "WHITE_BG"]
    col_x = [PAD,
              PAD + LABEL_W + PAD,
              PAD + LABEL_W + PAD + cell_w,
              PAD + LABEL_W + PAD + 2 * cell_w,
              PAD + LABEL_W + PAD + 3 * cell_w]
    for cx, lbl in zip(col_x, col_labels):
        draw.text((cx, hdr_h - 14), lbl, fill=(160, 160, 200), font=get_font(8))

    for ri, (base_id, variants) in enumerate(sorted(groups.items())):
        y = hdr_h + PAD + ri * cell_h

        # label
        label_cell = make_label_cell(base_id, LABEL_W, THUMB + TEXT_H, (160, 200, 160))
        sheet.paste(label_cell, (PAD, y))

        # 4 imágenes
        keys = ["original", "mask", "gray_bg", "white_bg"]
        colors = [(180, 180, 180), (140, 200, 140), (130, 190, 220), (220, 210, 160)]
        for ci, (key, color) in enumerate(zip(keys, colors)):
            x = PAD + LABEL_W + PAD + ci * cell_w
            if key in variants:
                thumb = load_thumb(variants[key])
            else:
                thumb = Image.new("RGB", (THUMB, THUMB), (50, 30, 30))
            cell = Image.new("RGB", (THUMB, THUMB + TEXT_H), BG_DARK)
            cell.paste(thumb, (0, 0))
            ImageDraw.Draw(cell).text((2, THUMB + 2), key[:10], fill=color, font=get_font(7))
            sheet.paste(cell, (x, y))

    sheet.save(out_path, quality=90)
    print(f"  [OK] {out_path.name}  ({len(groups)} filas)")


# ── TAREA 5: problem cases grid ───────────────────────────────────────────────
def build_problem_grid(groups, out_path):
    """
    Muestra solo los casos problema. Si faltan crops, los genera con GrabCut ligero.
    """
    LABEL_W = 90
    hdr_h   = 28
    cell_w  = THUMB + PAD
    cell_h  = THUMB + TEXT_H + PAD

    rows        = []   # (base_id, variants_dict, warning)
    missing_gen = []
    skipped     = []

    for pid in PROBLEM_IDS:
        if pid in groups:
            rows.append((pid, groups[pid], ""))
        else:
            print(f"  Generando crops ligeros (sin YOLO) para caso problema: {pid} ...")
            new_crops, warn = generate_problem_crops(pid)
            if new_crops:
                groups[pid] = new_crops          # añadir al dict global
                rows.append((pid, new_crops, "grabcut_center_fallback"))
                missing_gen.append(pid)
            else:
                skipped.append(pid)
                print(f"    AVISO: {pid} no encontrado — se omite del grid")

    if not rows:
        print("  AVISO: ningún caso problema disponible para el grid")
        return missing_gen, skipped

    W = LABEL_W + PAD + 4 * cell_w + PAD
    H = hdr_h + PAD + len(rows) * cell_h + PAD

    sheet = Image.new("RGB", (W, H), BG_DARK)
    draw  = ImageDraw.Draw(sheet)
    draw.text((PAD, 8), "Problem Cases — False BAD Diagnosed", fill=(220, 100, 80), font=get_font(10))
    col_labels = ["ID", "ORIGINAL", "MASK", "GRAY_BG", "WHITE_BG"]
    col_x = [PAD,
              PAD + LABEL_W + PAD,
              PAD + LABEL_W + PAD + cell_w,
              PAD + LABEL_W + PAD + 2 * cell_w,
              PAD + LABEL_W + PAD + 3 * cell_w]
    for cx, lbl in zip(col_x, col_labels):
        draw.text((cx, hdr_h - 14), lbl, fill=(220, 120, 100), font=get_font(8))

    for ri, (base_id, variants, warn) in enumerate(rows):
        y     = hdr_h + PAD + ri * cell_h
        lcolor = (220, 120, 80) if warn else (220, 100, 80)
        label_cell = make_label_cell(base_id, LABEL_W, THUMB + TEXT_H, lcolor)
        sheet.paste(label_cell, (PAD, y))
        keys   = ["original", "mask", "gray_bg", "white_bg"]
        colors = [(200, 160, 140), (160, 200, 140), (130, 190, 220), (220, 210, 160)]
        for ci, (key, color) in enumerate(zip(keys, colors)):
            x = PAD + LABEL_W + PAD + ci * cell_w
            if key in variants:
                thumb = load_thumb(variants[key])
            else:
                thumb = Image.new("RGB", (THUMB, THUMB), (50, 30, 30))
            cell = Image.new("RGB", (THUMB, THUMB + TEXT_H), BG_DARK)
            cell.paste(thumb, (0, 0))
            tag = key[:8] + ("*" if warn else "")
            ImageDraw.Draw(cell).text((2, THUMB + 2), tag, fill=color, font=get_font(7))
            sheet.paste(cell, (x, y))

    sheet.save(out_path, quality=90)
    print(f"  [OK] {out_path.name}  ({len(rows)} filas, {len(missing_gen)} generados ligeros, {len(skipped)} omitidos)")
    return missing_gen, skipped


# ── TAREA 3: CSV diagnóstico ──────────────────────────────────────────────────
def write_diagnostics_csv(groups):
    out    = PREVIEWS_DIR / "roi_masked_diagnostics.csv"
    fields = ["filename_base", "original_path", "mask_path", "gray_bg_path",
              "white_bg_path", "has_original", "has_mask", "has_gray_bg",
              "has_white_bg", "status", "warning"]
    n_ok = 0; n_inc = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for base_id, variants in sorted(groups.items()):
            ho = "original" in variants
            hm = "mask"     in variants
            hg = "gray_bg"  in variants
            hw = "white_bg" in variants
            status = "OK" if (ho and hm and hg and hw) else "INCOMPLETE"
            if status == "OK": n_ok += 1
            else: n_inc += 1
            missing = [k for k, v in [("original", ho), ("mask", hm),
                                       ("gray_bg", hg), ("white_bg", hw)] if not v]
            w.writerow({
                "filename_base": base_id,
                "original_path": str(variants.get("original", "")),
                "mask_path":     str(variants.get("mask", "")),
                "gray_bg_path":  str(variants.get("gray_bg", "")),
                "white_bg_path": str(variants.get("white_bg", "")),
                "has_original":  ho, "has_mask": hm,
                "has_gray_bg":   hg, "has_white_bg": hw,
                "status":        status,
                "warning":       f"missing: {missing}" if missing else "",
            })
    print(f"  [OK] {out.name}  ({n_ok} OK, {n_inc} INCOMPLETE)")
    return n_ok, n_inc


# ── TAREA 6: README ───────────────────────────────────────────────────────────
def write_readme(groups, n_ok, n_inc, skipped):
    out = PREVIEWS_DIR / "README_LOOK_HERE.txt"
    lines = [
        "=== ROI Masked Quality Previews — Dónde mirar ===",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "ARCHIVOS PRINCIPALES:",
        "-" * 50,
        "",
        "1. CONTACT SHEET GENERAL (todas las imágenes procesadas):",
        f"   outputs/quality_roi_masked_previews/contact_sheet_original_vs_masked.jpg",
        f"   -> {len(groups)} grupos de imágenes, 5 columnas por fila",
        f"      (ID | ORIGINAL | MASK | GRAY_BG | WHITE_BG)",
        "",
        "2. GRID DE CASOS PROBLEMÁTICOS (falsos BAD confirmados):",
        f"   outputs/quality_roi_masked_previews/problem_cases_grid.jpg",
        f"   -> 11 imágenes que el modelo V2 clasificó mal",
        f"   -> Las marcadas con * usan GrabCut-solo (sin YOLO)",
        "",
        "3. CSV DIAGNÓSTICO:",
        f"   outputs/quality_roi_masked_previews/roi_masked_diagnostics.csv",
        f"   -> {n_ok} grupos completos (OK), {n_inc} incompletos",
        "",
        "4. CROPS INDIVIDUALES:",
        f"   outputs/quality_roi_masked_previews/crops/",
        f"   -> {len(list(CROPS_DIR.glob('*.jpg')))} archivos JPG",
        f"   -> Formato: {{ID}}_original.jpg, {{ID}}_mask.jpg,",
        f"               {{ID}}_gray_bg.jpg, {{ID}}_white_bg.jpg",
        "",
        "QUÉ COMPROBAR:",
        "-" * 50,
        "  - ¿La máscara (MASK) recorta bien la pera o incluye demasiado fondo?",
        "  - ¿El crop GRAY_BG aísla bien la fruta sobre gris neutro?",
        "  - ¿Los casos problema (problem_cases_grid.jpg) muestran peras sanas",
        "    sin defectos visibles? → confirma que el error era el fondo.",
        "",
    ]
    if skipped:
        lines += [
            "CASOS PROBLEMA NO ENCONTRADOS (imagen original ausente):",
            *[f"  - {pid}" for pid in skipped],
            "",
        ]
    lines += [
        "CONFIRMACIONES:",
        "  NO se entrenó ningún modelo.",
        "  NO se modificó V2.",
        "  NO se modificó analyze_quality.py.",
        "  NO se modificó quality_rules.yaml.",
        "  NO se borraron crops existentes.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {out.name}")


# ── TAREA 7: reporte ──────────────────────────────────────────────────────────
def write_report(groups, n_ok, n_inc, missing_gen, skipped):
    n_problem_shown    = len(PROBLEM_IDS) - len(skipped)
    n_problem_gen      = len(missing_gen)
    n_problem_existing = n_problem_shown - n_problem_gen

    out = REPORTS_DIR / "fix_roi_masked_contact_sheets_report.md"
    content = f"""# Fix ROI Masked Contact Sheets — Report

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Estado de la carpeta crops/

| Métrica | Valor |
|---|---|
| Grupos de imagen encontrados | {len(groups)} |
| Grupos completos (OK) | {n_ok} |
| Grupos incompletos | {n_inc} |
| Total archivos JPG en crops/ | {len(list(CROPS_DIR.glob('*.jpg')))} |

---

## 2. Contact sheets regenerados

| Archivo | Estado |
|---|---|
| `contact_sheet_original_vs_masked.jpg` | ✅ Generado ({len(groups)} filas) |
| `problem_cases_grid.jpg` | ✅ Generado ({n_problem_shown} de {len(PROBLEM_IDS)} casos) |
| `roi_masked_diagnostics.csv` | ✅ Generado |
| `README_LOOK_HERE.txt` | ✅ Generado |

---

## 3. Casos problema

| ID | Crops previos | Método | Estado |
|---|---|---|---|
""" + "\n".join(
        f"| `{pid}` | "
        + ("Sí" if pid not in missing_gen and pid not in skipped else "No") + " | "
        + ("—" if pid not in missing_gen and pid not in skipped
           else ("grabcut_center" if pid not in skipped else "—")) + " | "
        + ("✅ Incluido" if pid not in skipped else "❌ Omitido (imagen no encontrada)")
        + " |"
        for pid in PROBLEM_IDS
    ) + f"""

- **{n_problem_existing}** casos ya tenían crops existentes.
- **{n_problem_gen}** casos se generaron con GrabCut ligero (sin YOLO) para completar el grid.
- **{len(skipped)}** casos omitidos por imagen original no encontrada.

> Nota: Los crops generados con GrabCut-solo (sin YOLO) usan un rectángulo central como
> inicialización. Pueden ser menos precisos que los generados con YOLO, pero son suficientes
> para diagnóstico visual.

---

## 4. Contexto

El script anterior (`prepare_quality_roi_masked_previews.py`) procesó 22 de 64 imágenes
antes de finalizar. Los crops de batch_v1 (20 imgs) y las primeras 2 de batch_v2 están
disponibles. Los 42 restantes (batch_v2 completo y batch_v3) no tienen crops todavía.

Este script solo regeneró los archivos de presentación usando lo que existía,
y completó los casos problema con GrabCut ligero.

---

## 5. Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.
- **NO** se borraron crops existentes.

---

## 6. Siguiente paso

José debe abrir:
1. `outputs/quality_roi_masked_previews/contact_sheet_original_vs_masked.jpg`
2. `outputs/quality_roi_masked_previews/problem_cases_grid.jpg`

Y comprobar si la máscara de la pera está bien recortada.
Si las máscaras son aceptables, el siguiente paso es entrenar U3 con el pipeline ROI/masked.
"""
    out.write_text(content, encoding="utf-8")
    print(f"  [OK] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== fix_roi_masked_contact_sheets ===")

    # TAREA 1+2: recoger grupos existentes
    print("  TAREA 1+2: Inspeccionando crops/ ...")
    groups = collect_existing_groups()
    print(f"  Grupos encontrados: {len(groups)}")
    n_complete = sum(1 for v in groups.values()
                     if all(k in v for k in ("original", "mask", "gray_bg", "white_bg")))
    print(f"  Grupos completos:   {n_complete}")
    print()

    # TAREA 3: CSV
    print("  TAREA 3: CSV diagnóstico ...")
    n_ok, n_inc = write_diagnostics_csv(groups)

    # TAREA 4: contact sheet general (solo grupos con al menos original)
    print("  TAREA 4: Contact sheet general ...")
    usable = {bid: v for bid, v in groups.items() if "original" in v}
    build_contact_sheet(usable,
                        PREVIEWS_DIR / "contact_sheet_original_vs_masked.jpg")

    # TAREA 5: problem grid (genera ligero si faltan)
    print("  TAREA 5: Problem cases grid ...")
    missing_gen, skipped = build_problem_grid(groups,
                                              PREVIEWS_DIR / "problem_cases_grid.jpg")

    # Actualizar CSV con los nuevos grupos añadidos
    n_ok, n_inc = write_diagnostics_csv(groups)

    # TAREA 6: README
    print("  TAREA 6: README_LOOK_HERE.txt ...")
    write_readme(groups, n_ok, n_inc, skipped)

    # TAREA 7: reporte
    print("  TAREA 7: Reporte final ...")
    write_report(groups, n_ok, n_inc, missing_gen, skipped)

    # TAREA 8: validación
    print()
    required = [
        PREVIEWS_DIR / "contact_sheet_original_vs_masked.jpg",
        PREVIEWS_DIR / "problem_cases_grid.jpg",
        PREVIEWS_DIR / "roi_masked_diagnostics.csv",
        PREVIEWS_DIR / "README_LOOK_HERE.txt",
        REPORTS_DIR  / "fix_roi_masked_contact_sheets_report.md",
    ]
    print("  TAREA 8: Validación ...")
    all_ok = True
    for p in required:
        mark = "✅" if p.exists() else "❌"
        print(f"    {mark} {p.relative_to(PROJECT_ROOT)}")
        if not p.exists():
            all_ok = False

    print()
    print("=" * 60)
    print("ROI MASKED CONTACT SHEETS REGENERADOS")
    print()
    print("Archivos principales:")
    print("- outputs/quality_roi_masked_previews/contact_sheet_original_vs_masked.jpg")
    print("- outputs/quality_roi_masked_previews/problem_cases_grid.jpg")
    print("- outputs/quality_roi_masked_previews/roi_masked_diagnostics.csv")
    print("- outputs/quality_roi_masked_previews/README_LOOK_HERE.txt")
    print("- reports/fix_roi_masked_contact_sheets_report.md")
    print()
    print("NO se entrenó ningún modelo.")
    print("NO se modificó V2.")
    print("NO se modificó analyze_quality.py.")
    print("NO se modificó quality_rules.yaml.")
    print()
    print("Siguiente paso:")
    print("José debe abrir contact_sheet_original_vs_masked.jpg y problem_cases_grid.jpg")
    print("para comprobar si la máscara de la pera está bien.")


if __name__ == "__main__":
    main()
