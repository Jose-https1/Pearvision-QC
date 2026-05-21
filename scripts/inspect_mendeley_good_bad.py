"""
Inspección visual y estadística del dataset Mendeley Good/Bad Pear.

Uso:
    python scripts/inspect_mendeley_good_bad.py
"""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOOD_DIR = PROJECT_ROOT / "data_external" / "mendeley_good_bad_pear" / "raw_clean" / "good"
BAD_DIR  = PROJECT_ROOT / "data_external" / "mendeley_good_bad_pear" / "raw_clean" / "bad"

OUT_DIR     = PROJECT_ROOT / "outputs" / "mendeley_good_bad_preview"
REPORT_DIR  = PROJECT_ROOT / "reports"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

GRID_COLS   = 6
GRID_ROWS   = 5          # 30 thumbs per sheet
THUMB_W     = 200
THUMB_H     = 200
GRID_GAP    = 4
GRID_BG     = (40, 40, 40)


# ── helpers ──────────────────────────────────────────────────────────────────

def _imread_unicode(path: Path):
    """cv2.imread seguro para rutas con caracteres Unicode en Windows."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _collect_paths(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _orientation(w, h) -> str:
    ratio = w / h
    if ratio > 1.05:
        return "horizontal"
    if ratio < 0.95:
        return "vertical"
    return "square"


def _inspect_split(paths: list[Path], label: str) -> list[dict]:
    records = []
    for p in paths:
        img = _imread_unicode(p)
        if img is None:
            records.append({"file": p.name, "label": label, "valid": False,
                             "w": None, "h": None, "orientation": None})
            continue
        h, w = img.shape[:2]
        records.append({"file": p.name, "label": label, "valid": True,
                         "w": w, "h": h, "orientation": _orientation(w, h)})
    return records


def _make_grid(paths: list[Path], out_path: Path, title: str,
               n_cols: int = GRID_COLS, n_rows: int = GRID_ROWS):
    n = n_cols * n_rows
    sample = random.sample(paths, min(n, len(paths)))

    cell_w = THUMB_W
    cell_h = THUMB_H
    gap    = GRID_GAP

    grid_w = n_cols * cell_w + (n_cols + 1) * gap
    grid_h = n_rows * cell_h + (n_rows + 1) * gap + 28   # +28 for title bar

    canvas = np.full((grid_h, grid_w, 3), GRID_BG, dtype=np.uint8)

    # title
    cv2.putText(canvas, title, (gap, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    for idx, p in enumerate(sample):
        row = idx // n_cols
        col = idx % n_cols
        x0 = gap + col * (cell_w + gap)
        y0 = 28 + gap + row * (cell_h + gap)

        img = _imread_unicode(p)
        if img is None:
            continue
        thumb = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(str(out_path))
    print(f"  Preview guardado: {out_path.relative_to(PROJECT_ROOT)}")


def _stats(records: list[dict], label: str) -> dict:
    valid = [r for r in records if r["valid"]]
    corrupt = [r for r in records if not r["valid"]]

    widths  = [r["w"] for r in valid]
    heights = [r["h"] for r in valid]

    orientations = {"horizontal": 0, "vertical": 0, "square": 0}
    for r in valid:
        orientations[r["orientation"]] += 1

    return {
        "label":      label,
        "total":      len(records),
        "valid":      len(valid),
        "corrupt":    len(corrupt),
        "min_w":      min(widths)  if widths else None,
        "max_w":      max(widths)  if widths else None,
        "mean_w":     int(np.mean(widths))  if widths else None,
        "min_h":      min(heights) if heights else None,
        "max_h":      max(heights) if heights else None,
        "mean_h":     int(np.mean(heights)) if heights else None,
        "horizontal": orientations["horizontal"],
        "vertical":   orientations["vertical"],
        "square":     orientations["square"],
    }


def _save_csv(all_records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "label", "valid", "w", "h", "orientation"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"  CSV guardado: {out_path.relative_to(PROJECT_ROOT)}")


def _save_markdown(good_st: dict, bad_st: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = good_st["total"] + bad_st["total"]
    corrupt = good_st["corrupt"] + bad_st["corrupt"]

    lines = [
        "# Inspección Dataset Mendeley Good/Bad Pear",
        "",
        f"**Fecha:** 2026-05-17",
        "",
        "## Conteos",
        "",
        f"| Split | Total | Válidas | Corruptas |",
        f"|-------|-------|---------|-----------|",
        f"| good  | {good_st['total']} | {good_st['valid']} | {good_st['corrupt']} |",
        f"| bad   | {bad_st['total']}  | {bad_st['valid']}  | {bad_st['corrupt']}  |",
        f"| **Total** | **{total}** | **{good_st['valid'] + bad_st['valid']}** | **{corrupt}** |",
        "",
        "## Resoluciones",
        "",
        "| Split | Min (WxH) | Max (WxH) | Media (WxH) |",
        "|-------|-----------|-----------|-------------|",
        f"| good | {good_st['min_w']}x{good_st['min_h']} | {good_st['max_w']}x{good_st['max_h']} | {good_st['mean_w']}x{good_st['mean_h']} |",
        f"| bad  | {bad_st['min_w']}x{bad_st['min_h']}  | {bad_st['max_w']}x{bad_st['max_h']}  | {bad_st['mean_w']}x{bad_st['mean_h']}  |",
        "",
        "## Orientación",
        "",
        "| Split | Horizontal | Vertical | Cuadrada |",
        "|-------|-----------|---------|---------|",
        f"| good | {good_st['horizontal']} | {good_st['vertical']} | {good_st['square']} |",
        f"| bad  | {bad_st['horizontal']}  | {bad_st['vertical']}  | {bad_st['square']}  |",
        "",
        "## Previews generados",
        "",
        "- `outputs/mendeley_good_bad_preview/good_grid.jpg`",
        "- `outputs/mendeley_good_bad_preview/bad_grid.jpg`",
        "- `outputs/mendeley_good_bad_preview/mixed_grid.jpg`",
        "",
        "## Utilidad estimada del dataset",
        "",
        "| Uso | Valoración |",
        "|-----|-----------|",
        "| Clasificación good/bad | **Alta** — etiquetas binarias directas |",
        "| Detección de defectos  | **Baja** — sin bounding boxes |",
        "| Segmentación           | **Baja** — sin máscaras de instancia |",
        "| Apoyo al sistema actual (pre-filtro / clasificador) | **Media-Alta** — útil como clasificador binario auxiliar |",
        "",
        "## Siguiente paso recomendado",
        "",
        "Entrenar un clasificador binario ligero (YOLOv8n-cls o MobileNet)  ",
        "sobre este dataset para usarlo como pre-filtro que descarte peras  ",
        "claramente dañadas antes del pipeline principal de detección de defectos.",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown guardado: {out_path.relative_to(PROJECT_ROOT)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    print("\n=== Inspección Mendeley Good/Bad Pear ===\n")

    for folder, name in [(GOOD_DIR, "good"), (BAD_DIR, "bad")]:
        if not folder.exists():
            print(f"ERROR: carpeta no encontrada: {folder}", file=sys.stderr)
            sys.exit(1)

    good_paths = _collect_paths(GOOD_DIR)
    bad_paths  = _collect_paths(BAD_DIR)
    print(f"Encontradas: {len(good_paths)} good, {len(bad_paths)} bad")

    print("\nInspeccionando imágenes good ...")
    good_records = _inspect_split(good_paths, "good")

    print("Inspeccionando imágenes bad ...")
    bad_records = _inspect_split(bad_paths, "bad")

    good_st = _stats(good_records, "good")
    bad_st  = _stats(bad_records,  "bad")

    print("\n--- Estadísticas ---")
    for st in (good_st, bad_st):
        print(f"  [{st['label']}] total={st['total']}  válidas={st['valid']}  "
              f"corruptas={st['corrupt']}  "
              f"res_media={st['mean_w']}x{st['mean_h']}  "
              f"H={st['horizontal']} V={st['vertical']} S={st['square']}")

    print("\nGenerando previews ...")
    valid_good = [p for p, r in zip(good_paths, good_records) if r["valid"]]
    valid_bad  = [p for p, r in zip(bad_paths,  bad_records)  if r["valid"]]
    mixed      = random.sample(valid_good, min(15, len(valid_good))) + \
                 random.sample(valid_bad,  min(15, len(valid_bad)))
    random.shuffle(mixed)

    _make_grid(valid_good, OUT_DIR / "good_grid.jpg",  "good pears")
    _make_grid(valid_bad,  OUT_DIR / "bad_grid.jpg",   "bad pears")
    _make_grid(mixed,      OUT_DIR / "mixed_grid.jpg", "mixed (good + bad)",
               n_rows=5, n_cols=6)

    print("\nGuardando CSV ...")
    _save_csv(good_records + bad_records,
              OUT_DIR / "mendeley_good_bad_summary.csv")

    print("\nGuardando reporte markdown ...")
    _save_markdown(good_st, bad_st,
                   REPORT_DIR / "mendeley_good_bad_inspection.md")

    print("\n=== Listo ===")


if __name__ == "__main__":
    main()
