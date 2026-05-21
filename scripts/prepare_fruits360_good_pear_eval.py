"""
prepare_fruits360_good_pear_eval.py
Selecciona y copia imágenes de peras buenas de Fruits-360 para auditar el pipeline.

Lee:   data_external/fruits360_original_size/{Training,Validation,Test}/Pear*
Crea:
  data/samples_quality_fruits360_good_eval/            (imágenes renombradas)
  data/samples_quality_fruits360_good_eval_expectations.csv
  outputs/fruits360_good_pear_eval_preview/contact_sheet.jpg
  reports/fruits360_good_pear_eval_dataset_report.md
"""

import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_BASE  = PROJECT_ROOT / "data_external" / "fruits360_original_size"
SPLITS    = ["Training", "Validation", "Test"]
DEST_DIR  = PROJECT_ROOT / "data" / "samples_quality_fruits360_good_eval"
EXPECT_CSV = PROJECT_ROOT / "data" / "samples_quality_fruits360_good_eval_expectations.csv"
PREVIEW_DIR = PROJECT_ROOT / "outputs" / "fruits360_good_pear_eval_preview"
REPORT_MD   = PROJECT_ROOT / "reports" / "fruits360_good_pear_eval_dataset_report.md"

MAX_PER_CLASS_SPLIT = 5
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def safe_stem(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def find_pear_dirs(split: str):
    d = SRC_BASE / split
    if not d.exists():
        return []
    return sorted([x for x in d.iterdir() if x.is_dir() and x.name.lower().startswith("pear")])


def copy_dataset():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    copied = []
    class_counts = {}

    for split in SPLITS:
        for pear_dir in find_pear_dirs(split):
            imgs = sorted([f for f in pear_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
            if not imgs:
                continue
            selected = rng.sample(imgs, min(MAX_PER_CLASS_SPLIT, len(imgs)))
            for src in selected:
                new_name = f"{split}__{safe_stem(pear_dir.name)}__{src.name}"
                # Normalizar extensión a .jpg
                if src.suffix.lower() != ".jpg":
                    new_name = Path(new_name).with_suffix(".jpg").name
                dest = DEST_DIR / new_name
                # Copiar convirtiando a jpg si necesario
                if src.suffix.lower() == ".jpg":
                    shutil.copy2(src, dest)
                else:
                    buf = np.frombuffer(src.read_bytes(), dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        cv2.imwrite(str(dest), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    else:
                        shutil.copy2(src, dest)
                copied.append({"image": new_name, "split": split, "class": pear_dir.name})
                key = f"{split}/{pear_dir.name}"
                class_counts[key] = class_counts.get(key, 0) + 1

    return copied, class_counts


def write_expectations(copied):
    EXPECT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPECT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "expected_group", "allowed_decisions", "notes"])
        w.writeheader()
        for entry in copied:
            w.writerow({
                "image": entry["image"],
                "expected_group": "fruits360_good",
                "allowed_decisions": "PASA|REVISAR",
                "notes": "pera sana de Fruits-360; no debe ser RECHAZA",
            })
    print(f"  Expectativas: {EXPECT_CSV} ({len(copied)} filas)")


def _load_thumb(path: Path, w=120, h=120):
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError
        scale = min(w / bgr.shape[1], h / bgr.shape[0])
        nw, nh = max(1, int(bgr.shape[1] * scale)), max(1, int(bgr.shape[0] * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)
        yo, xo = (h - nh) // 2, (w - nw) // 2
        canvas[yo:yo+nh, xo:xo+nw] = bgr
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    except Exception:
        return Image.fromarray(np.full((h, w, 3), 60, dtype=np.uint8))


def make_preview(copied):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    # Mostrar hasta 60 imágenes (primera de cada clase×split, luego resto)
    sample = copied[:60]
    cols = 10
    rows_n = (len(sample) + cols - 1) // cols
    CELL_W, CELL_H = 130, 150
    HEADER = 30
    sheet_w = CELL_W * cols
    sheet_h = HEADER + CELL_H * rows_n

    img = Image.new("RGB", (sheet_w, sheet_h), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
        font_hdr = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except Exception:
        font = font_hdr = ImageFont.load_default()

    draw.text((6, 6), f"Fruits-360 Good Pear Preview — {len(copied)} imagenes copiadas",
              fill=(220, 220, 220), font=font_hdr)

    for i, entry in enumerate(sample):
        col, row = i % cols, i // cols
        x0 = col * CELL_W
        y0 = HEADER + row * CELL_H
        thumb = _load_thumb(DEST_DIR / entry["image"], CELL_W - 4, CELL_H - 22)
        img.paste(thumb, (x0 + 2, y0 + 2))
        label = entry["class"].replace("Pear ", "P") + " " + entry["split"][:2]
        draw.text((x0 + 3, y0 + CELL_H - 18), label, fill=(180, 220, 180), font=font)

    out = PREVIEW_DIR / "contact_sheet.jpg"
    img.save(str(out), quality=90)
    print(f"  Preview: {out}")


def write_report(copied, class_counts):
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fruits-360 Good Pear Eval — Dataset Report",
        "",
        f"Generado automáticamente.",
        "",
        "## Origen",
        f"- `data_external/fruits360_original_size/`",
        f"- Splits usados: Training, Validation, Test",
        f"- Máximo por clase×split: {MAX_PER_CLASS_SPLIT}",
        f"- Seed: {SEED}",
        "",
        "## Carpetas Pear encontradas",
        "",
    ]
    for split in SPLITS:
        pear_dirs = find_pear_dirs(split)
        lines.append(f"### {split} ({len(pear_dirs)} clases)")
        for pd in pear_dirs:
            key = f"{split}/{pd.name}"
            n = class_counts.get(key, 0)
            lines.append(f"- `{pd.name}`: {n} imagenes copiadas")
        lines.append("")

    per_split = {}
    per_class = {}
    for e in copied:
        per_split[e["split"]] = per_split.get(e["split"], 0) + 1
        per_class[e["class"]] = per_class.get(e["class"], 0) + 1

    lines += [
        "## Resumen",
        "",
        f"| Metrica | Valor |",
        f"|---------|-------|",
        f"| Total imagenes copiadas | {len(copied)} |",
    ]
    for sp in SPLITS:
        lines.append(f"| {sp} | {per_split.get(sp, 0)} |")
    lines += [
        "",
        "## Rutas finales",
        "",
        f"- Dataset: `data/samples_quality_fruits360_good_eval/`",
        f"- Expectativas: `data/samples_quality_fruits360_good_eval_expectations.csv`",
        f"- Preview: `outputs/fruits360_good_pear_eval_preview/contact_sheet.jpg`",
        "",
        "## Advertencia",
        "",
        "Este dataset sirve como **control de peras buenas**.",
        "No contiene defectos reales — se usa para medir falsos rechazos del pipeline.",
        "NO usar como dataset de entrenamiento de defectos.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD}")


def main():
    print("=== prepare_fruits360_good_pear_eval ===")

    # Verificar origen
    if not SRC_BASE.exists():
        print(f"ERROR: no existe {SRC_BASE}")
        sys.exit(1)

    copied, class_counts = copy_dataset()
    print(f"  Copiadas: {len(copied)} imagenes -> {DEST_DIR}")

    write_expectations(copied)
    make_preview(copied)
    write_report(copied, class_counts)

    print(f"\n=== Dataset listo ===")
    print(f"  {len(copied)} imagenes en {DEST_DIR}")


if __name__ == "__main__":
    main()
