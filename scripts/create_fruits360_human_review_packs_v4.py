"""
create_fruits360_human_review_packs_v4.py
Genera lotes de revisión humana para imágenes de peras de Fruits-360.

Crea:
  data/fruits360_human_review/images/              (imágenes con ID)
  data/fruits360_human_review/fruits360_human_review_master.csv
  data/fruits360_human_review/human_labels_template.csv
  data/fruits360_human_review/README_HUMAN_LABELING.md
  outputs/fruits360_human_review_packs/review_pack_001.jpg ... review_pack_NNN.jpg
  outputs/fruits360_human_review_packs/review_pack_overview.jpg
  reports/fruits360_human_review_packs_v4_report.md
"""

import csv
import math
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_BASE   = PROJECT_ROOT / "data_external" / "fruits360_original_size"
SPLITS     = ["Training", "Validation", "Test"]
IMG_EXTS   = {".jpg", ".jpeg", ".png"}

REVIEW_DIR  = PROJECT_ROOT / "data" / "fruits360_human_review"
IMAGES_DIR  = REVIEW_DIR / "images"
MASTER_CSV  = REVIEW_DIR / "fruits360_human_review_master.csv"
TEMPLATE_CSV = REVIEW_DIR / "human_labels_template.csv"
README_MD   = REVIEW_DIR / "README_HUMAN_LABELING.md"

PACKS_DIR   = PROJECT_ROOT / "outputs" / "fruits360_human_review_packs"
REPORT_MD   = PROJECT_ROOT / "reports" / "fruits360_human_review_packs_v4_report.md"

MAX_TOTAL   = 300
IMGS_PER_PACK = 30   # 5 cols × 6 rows
SEED        = 42

VALID_LABELS = ["GOOD", "BAD", "INVALID", "REVIEW"]

# Colores por clase Pear para el ID visual
CLASS_PALETTE = [
    (100, 200, 100), (200, 150,  60), ( 80, 160, 220), (200,  80, 100),
    (160, 220, 100), (220, 180,  60), (100, 180, 200), (180,  80, 220),
    (220, 130,  80), ( 80, 220, 160), (200, 200,  80), (130, 100, 200),
    (200, 100, 140),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_font(size: int):
    for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
              "C:/Windows/Fonts/verdana.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _load_thumb(path: Path, w: int, h: int) -> Image.Image:
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError
        scale = min(w / bgr.shape[1], h / bgr.shape[0])
        nw = max(1, int(bgr.shape[1] * scale))
        nh = max(1, int(bgr.shape[0] * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((h, w, 3), 38, dtype=np.uint8)
        yo, xo = (h - nh) // 2, (w - nw) // 2
        canvas[yo:yo + nh, xo:xo + nw] = bgr
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    except Exception:
        return Image.fromarray(np.full((h, w, 3), 55, dtype=np.uint8))


# ---------------------------------------------------------------------------
# 1. Muestreo equilibrado
# ---------------------------------------------------------------------------

def collect_pear_images() -> list[dict]:
    """Recorre Training/Validation/Test y recoge todas las imágenes Pear*."""
    all_imgs = []
    for split in SPLITS:
        d = SRC_BASE / split
        if not d.exists():
            continue
        for cls_dir in sorted(d.iterdir()):
            if not cls_dir.is_dir():
                continue
            if "pear" not in cls_dir.name.lower():
                continue
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in IMG_EXTS:
                    all_imgs.append({
                        "original_split": split,
                        "original_class": cls_dir.name,
                        "filename":       img.name,
                        "source_path":    str(img),
                    })
    return all_imgs


def balanced_sample(all_imgs: list[dict], max_total: int, seed: int) -> list[dict]:
    """Muestreo estratificado por clase, equilibrado."""
    rng = random.Random(seed)

    by_class: dict[str, list[dict]] = {}
    for img in all_imgs:
        by_class.setdefault(img["original_class"], []).append(img)

    n_classes = len(by_class)
    per_class = max_total // n_classes
    remainder = max_total - per_class * n_classes

    selected = []
    classes_sorted = sorted(by_class.keys())

    for i, cls in enumerate(classes_sorted):
        pool = list(by_class[cls])
        rng.shuffle(pool)
        quota = per_class + (1 if i < remainder else 0)
        selected.extend(pool[:quota])

    rng.shuffle(selected)
    return selected[:max_total]


# ---------------------------------------------------------------------------
# 2. Copiar imágenes con ID único
# ---------------------------------------------------------------------------

def copy_with_ids(sample: list[dict]) -> list[dict]:
    """Copia imágenes a images/ con nombre F360_NNNN.jpg y devuelve registros."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for i, entry in enumerate(sample):
        review_id = f"F360_{i+1:04d}"
        dst_name  = f"{review_id}.jpg"
        dst       = IMAGES_DIR / dst_name
        src       = Path(entry["source_path"])

        # Copiar convirtiendo a JPG si es PNG
        if src.suffix.lower() == ".jpg":
            shutil.copy2(src, dst)
        else:
            buf = np.frombuffer(src.read_bytes(), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is not None:
                cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                shutil.copy2(src, dst)

        records.append({
            "review_id":      review_id,
            "original_split": entry["original_split"],
            "original_class": entry["original_class"],
            "filename":       dst_name,
            "source_path":    entry["source_path"],
            "human_label":    "",
            "notes":          "",
        })
    return records


# ---------------------------------------------------------------------------
# 3. CSVs
# ---------------------------------------------------------------------------

CSV_FIELDS = ["review_id", "original_split", "original_class",
              "filename", "source_path", "human_label", "notes"]


def write_master_csv(records: list[dict]):
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(records)
    print(f"  Master CSV: {MASTER_CSV} ({len(records)} filas)")


def write_template_csv(records: list[dict]):
    """Plantilla vacía para que el usuario rellene human_label y notes."""
    with open(TEMPLATE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            row = dict(r)
            row["human_label"] = ""
            row["notes"] = ""
            w.writerow(row)
    print(f"  Template CSV: {TEMPLATE_CSV}")


# ---------------------------------------------------------------------------
# 4. README
# ---------------------------------------------------------------------------

def write_readme(records: list[dict]):
    classes = sorted({r["original_class"] for r in records})
    n_packs = math.ceil(len(records) / IMGS_PER_PACK)

    text = f"""# Fruits-360 Human Review — Instrucciones de Etiquetado

Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Objetivo

Etiquetar visualmente {len(records)} imágenes de peras de Fruits-360
para construir un dataset de entrenamiento de calidad fiable.

## Cómo revisar

1. Abre los contact sheets en: `outputs/fruits360_human_review_packs/`
2. Cada imagen tiene su ID visible (formato `F360_NNNN`).
3. Rellena el archivo: `data/fruits360_human_review/human_labels_template.csv`
4. Para cada imagen, escribe en la columna `human_label` una de estas etiquetas:

| Etiqueta | Significado |
|----------|-------------|
| GOOD     | Pera sana, comercialmente aceptable, sin defectos visibles |
| BAD      | Pera con defectos claros: manchas, podredumbre, golpes, mordidas |
| INVALID  | Imagen no útil: pera cortada, fondo visible, imagen borrosa o negra |
| REVIEW   | Dudosa — no estás seguro, necesita segunda opinión |

5. Puedes añadir notas libres en la columna `notes`.
6. Guarda el CSV cuando termines.

## Estructura de archivos

```
data/fruits360_human_review/
  images/                          (imágenes renombradas con ID)
  fruits360_human_review_master.csv  (registro completo, no modificar)
  human_labels_template.csv         (rellenar este archivo)

outputs/fruits360_human_review_packs/
  review_pack_overview.jpg          (vista general de todos los packs)
  review_pack_001.jpg               ({IMGS_PER_PACK} imágenes por hoja)
  review_pack_002.jpg
  ...
  review_pack_{n_packs:03d}.jpg
```

## Clases de pera incluidas

{chr(10).join(f'- {c}' for c in classes)}

## Notas importantes

- Las imágenes de Fruits-360 tienen fondo blanco de estudio.
- Algunas clases incluyen peras con defectos reales o aspecto no comercial.
- No asumas que todas son buenas solo por ser Fruits-360.
- Si tienes dudas, usa REVIEW.
- INVALID se reserva para imágenes inutilizables (no para peras con defectos).
"""
    README_MD.write_text(text, encoding="utf-8")
    print(f"  README: {README_MD}")


# ---------------------------------------------------------------------------
# 5. Contact sheets por pack
# ---------------------------------------------------------------------------

COLS      = 5
THUMB_W   = 160
THUMB_H   = 140
LABEL_H   = 36
HEADER_H  = 40
CELL_W    = THUMB_W
CELL_H    = THUMB_H + LABEL_H

# Mapa clase → color
def _class_color_map(records):
    classes = sorted({r["original_class"] for r in records})
    return {cls: CLASS_PALETTE[i % len(CLASS_PALETTE)] for i, cls in enumerate(classes)}


def make_pack_sheet(pack_records: list[dict], pack_num: int,
                    class_colors: dict, out_path: Path):
    n    = len(pack_records)
    rows = math.ceil(n / COLS)
    sheet_w = CELL_W * COLS
    sheet_h = HEADER_H + CELL_H * rows

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw  = ImageDraw.Draw(sheet)

    f_hdr = _get_font(15)
    f_id  = _get_font(11)
    f_cls = _get_font(10)

    draw.text((6, 8),
              f"Review Pack {pack_num:03d}  —  {n} imágenes  (etiquetas: GOOD / BAD / INVALID / REVIEW)",
              fill=(210, 210, 210), font=f_hdr)

    for i, rec in enumerate(pack_records):
        col = i % COLS
        row = i // COLS
        x0  = col * CELL_W
        y0  = HEADER_H + row * CELL_H

        cls   = rec["original_class"]
        color = class_colors.get(cls, (180, 180, 180))
        draw.rectangle([(x0, y0), (x0 + CELL_W, y0 + CELL_H)], fill=(28, 28, 28))

        thumb = _load_thumb(IMAGES_DIR / rec["filename"], THUMB_W - 2, THUMB_H - 2)
        sheet.paste(thumb, (x0 + 1, y0 + 1))

        # Franja de etiqueta
        draw.rectangle([(x0, y0 + THUMB_H), (x0 + CELL_W, y0 + CELL_H)], fill=(35, 35, 35))

        # review_id en negrita (color de clase)
        draw.text((x0 + 3, y0 + THUMB_H + 2),
                  rec["review_id"], fill=color, font=f_id)

        # clase — truncada si larga
        cls_short = cls.replace("Pear ", "P").replace("common", "com")
        draw.text((x0 + 3, y0 + THUMB_H + 16),
                  cls_short[:20], fill=(140, 140, 140), font=f_cls)

        # borde lateral del color de clase
        draw.rectangle([(x0, y0), (x0 + 3, y0 + THUMB_H)], fill=color)

    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out_path), quality=90)


def make_all_packs(records: list[dict], class_colors: dict) -> int:
    n_packs = math.ceil(len(records) / IMGS_PER_PACK)
    for p in range(n_packs):
        batch = records[p * IMGS_PER_PACK: (p + 1) * IMGS_PER_PACK]
        out   = PACKS_DIR / f"review_pack_{p+1:03d}.jpg"
        make_pack_sheet(batch, p + 1, class_colors, out)
    print(f"  Packs generados: {n_packs}  ->  {PACKS_DIR}")
    return n_packs


# ---------------------------------------------------------------------------
# 6. Overview
# ---------------------------------------------------------------------------

def make_overview(records: list[dict], n_packs: int, class_colors: dict):
    """Una fila por pack, mostrando 6 thumbnails de muestra y estadísticas."""
    SAMPLE_PER_PACK = 6
    OVTH_W = 100
    OVTH_H = 90
    TEXT_W = 200
    ROW_H  = OVTH_H + 4
    HDR_H  = 44

    sheet_w = TEXT_W + OVTH_W * SAMPLE_PER_PACK
    sheet_h = HDR_H + ROW_H * n_packs

    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 18))
    draw  = ImageDraw.Draw(sheet)

    f_hdr = _get_font(14)
    f_md  = _get_font(11)
    f_sm  = _get_font(10)

    draw.text((6, 6),
              f"Fruits-360 Human Review — Overview  "
              f"({len(records)} imgs / {n_packs} packs / {IMGS_PER_PACK} imgs por pack)",
              fill=(215, 215, 215), font=f_hdr)
    draw.text((6, 24),
              "Instrucciones: outputs/fruits360_human_review_packs/  |  "
              "Etiquetas: GOOD / BAD / INVALID / REVIEW",
              fill=(140, 140, 140), font=f_sm)

    for p in range(n_packs):
        batch = records[p * IMGS_PER_PACK: (p + 1) * IMGS_PER_PACK]
        y0    = HDR_H + p * ROW_H

        # Fondo alternado
        bg = (28, 28, 28) if p % 2 == 0 else (32, 32, 32)
        draw.rectangle([(0, y0), (sheet_w, y0 + ROW_H)], fill=bg)

        # Panel de texto
        cls_counts: dict[str, int] = {}
        for r in batch:
            cls_counts[r["original_class"]] = cls_counts.get(r["original_class"], 0) + 1

        draw.text((4, y0 + 2),
                  f"Pack {p+1:03d}  ({len(batch)} imgs)",
                  fill=(200, 200, 200), font=f_md)
        cls_line = "  ".join(
            f"{c.replace('Pear ','P')}={n}" for c, n in sorted(cls_counts.items())
        )
        draw.text((4, y0 + 18), cls_line[:36], fill=(120, 140, 120), font=f_sm)

        # Thumbnails de muestra (primeras SAMPLE_PER_PACK del pack)
        for si, rec in enumerate(batch[:SAMPLE_PER_PACK]):
            x0  = TEXT_W + si * OVTH_W
            th  = _load_thumb(IMAGES_DIR / rec["filename"], OVTH_W - 2, OVTH_H - 2)
            sheet.paste(th, (x0 + 1, y0 + 2))
            color = class_colors.get(rec["original_class"], (160, 160, 160))
            # Franja de color izquierda
            draw.rectangle([(x0, y0), (x0 + 2, y0 + OVTH_H)], fill=color)
            # ID pequeño
            draw.text((x0 + 4, y0 + OVTH_H - 14),
                      rec["review_id"], fill=(200, 200, 200), font=f_sm)

    out = PACKS_DIR / "review_pack_overview.jpg"
    sheet.save(str(out), quality=90)
    print(f"  Overview: {out}")


# ---------------------------------------------------------------------------
# 7. Reporte final
# ---------------------------------------------------------------------------

def write_report(records: list[dict], n_packs: int):
    by_class: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for r in records:
        by_class[r["original_class"]] = by_class.get(r["original_class"], 0) + 1
        by_split[r["original_split"]] = by_split.get(r["original_split"], 0) + 1

    lines = [
        "# Fruits-360 Human Review Packs v4 — Informe",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Resumen",
        "",
        f"| Metrica | Valor |",
        f"|---------|-------|",
        f"| Total imagenes seleccionadas | {len(records)} |",
        f"| Packs generados | {n_packs} |",
        f"| Imagenes por pack | {IMGS_PER_PACK} |",
        f"| Seed de muestreo | {SEED} |",
        "",
        "## Distribucion por split",
        "",
        "| Split | Imagenes |",
        "|-------|----------|",
    ]
    for sp in ["Training", "Validation", "Test"]:
        lines.append(f"| {sp} | {by_split.get(sp, 0)} |")

    lines += [
        "",
        "## Distribucion por clase Pear",
        "",
        "| Clase | Imagenes |",
        "|-------|----------|",
    ]
    for cls, n in sorted(by_class.items()):
        lines.append(f"| {cls} | {n} |")

    lines += [
        "",
        "## Packs generados",
        "",
        "| Pack | Imagenes | Rango IDs |",
        "|------|----------|-----------|",
    ]
    for p in range(n_packs):
        batch  = records[p * IMGS_PER_PACK: (p + 1) * IMGS_PER_PACK]
        id_ini = batch[0]["review_id"]
        id_fin = batch[-1]["review_id"]
        lines.append(f"| {p+1:03d} | {len(batch)} | {id_ini} — {id_fin} |")

    lines += [
        "",
        "## Archivos generados",
        "",
        f"- `data/fruits360_human_review/images/`  ({len(records)} imagenes con ID F360_NNNN)",
        f"- `data/fruits360_human_review/fruits360_human_review_master.csv`",
        f"- `data/fruits360_human_review/human_labels_template.csv`  (rellenar este)",
        f"- `data/fruits360_human_review/README_HUMAN_LABELING.md`",
        f"- `outputs/fruits360_human_review_packs/review_pack_overview.jpg`",
    ]
    for p in range(n_packs):
        lines.append(f"- `outputs/fruits360_human_review_packs/review_pack_{p+1:03d}.jpg`")

    lines += [
        "",
        "## Instrucciones de uso",
        "",
        "1. Abre los contact sheets en `outputs/fruits360_human_review_packs/`.",
        "2. Cada imagen muestra su ID `F360_NNNN` y la clase Pear de origen.",
        "3. Rellena `human_labels_template.csv` con: GOOD / BAD / INVALID / REVIEW.",
        "4. Una vez etiquetado, el CSV se usará para entrenar el clasificador de calidad.",
        "",
        "## Notas",
        "",
        "- El muestreo es equilibrado entre clases pero no ha sido revisado imagen a imagen.",
        "- Algunas clases de Fruits-360 contienen peras con defectos reales o no comerciales.",
        "- `INVALID` es para imágenes inutilizables (imagen negra, cortada, etc.).",
        "- `REVIEW` es para casos dudosos que necesitan segunda opinión.",
    ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== create_fruits360_human_review_packs_v4 ===")

    # Crear directorios base
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    PACKS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Recoger imágenes
    print("  Buscando imágenes Pear en Fruits-360...")
    all_imgs = collect_pear_images()
    print(f"  Total encontradas: {len(all_imgs)}")

    # 2. Muestreo equilibrado
    sample = balanced_sample(all_imgs, MAX_TOTAL, SEED)
    print(f"  Muestra seleccionada: {len(sample)}")

    # 3. Copiar con IDs
    print("  Copiando imágenes con IDs...")
    records = copy_with_ids(sample)
    print(f"  Copiadas: {len(records)} -> {IMAGES_DIR}")

    # 4. CSVs
    write_master_csv(records)
    write_template_csv(records)

    # 5. README
    write_readme(records)

    # 6. Contact sheets
    class_colors = _class_color_map(records)
    print("  Generando contact sheets...")
    n_packs = make_all_packs(records, class_colors)

    # 7. Overview
    print("  Generando overview...")
    make_overview(records, n_packs, class_colors)

    # 8. Reporte
    write_report(records, n_packs)

    # 9. Validación final
    print()
    required = [
        MASTER_CSV, TEMPLATE_CSV, README_MD,
        PACKS_DIR / "review_pack_overview.jpg",
        PACKS_DIR / "review_pack_001.jpg",
        REPORT_MD,
    ]
    all_ok = True
    for p in required:
        exists = p.exists()
        print(f"  {'OK' if exists else 'MISSING':7} {p.name}")
        if not exists:
            all_ok = False

    status = "OK" if all_ok else "INCOMPLETE"
    print()
    print(f"STATUS: {status}")
    print(f"TOTAL_IMAGES: {len(records)}")
    print(f"PACKS: {n_packs}")
    print(f"IMAGES_DIR: {IMAGES_DIR}")
    print(f"PACKS_DIR: {PACKS_DIR}")
    print(f"MASTER_CSV: {MASTER_CSV}")
    print(f"OVERVIEW: {PACKS_DIR / 'review_pack_overview.jpg'}")
    print(f"FIRST_PACK: {PACKS_DIR / 'review_pack_001.jpg'}")
    print(f"REPORT: {REPORT_MD}")


if __name__ == "__main__":
    main()
