"""
prepare_mendeley_good_bad_cls.py
Prepara el dataset Mendeley good/bad para clasificacion YOLO.

Split: 70% train / 20% val / 10% test  (seed=42)
Copia imagenes, no las mueve.
Genera reporte en reports/mendeley_good_bad_cls_report.md
"""

import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "data_external" / "mendeley_good_bad_pear" / "raw_clean"
DST_ROOT = PROJECT_ROOT / "data" / "pear_quality_cls_mendeley"
REPORT_PATH = PROJECT_ROOT / "reports" / "mendeley_good_bad_cls_report.md"

SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path) -> list:
    imgs = []
    for f in folder.iterdir():
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in VALID_EXT:
            imgs.append(f)
    return sorted(imgs)


def split_list(items: list, ratios: dict, seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def copy_split(files: list, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dst_dir / src.name)


def main():
    print("=== prepare_mendeley_good_bad_cls ===")

    # Recoger imagenes
    good_imgs = collect_images(SRC_ROOT / "good")
    bad_imgs = collect_images(SRC_ROOT / "bad")
    print(f"  good source : {len(good_imgs)} imagenes")
    print(f"  bad  source : {len(bad_imgs)} imagenes")

    if not good_imgs or not bad_imgs:
        print("ERROR: no se encontraron imagenes en las carpetas source.")
        sys.exit(1)

    # Split estratificado por clase
    good_split = split_list(good_imgs, SPLITS, SEED)
    bad_split = split_list(bad_imgs, SPLITS, SEED)

    counts = {}
    for split_name in ("train", "val", "test"):
        counts[split_name] = {}
        for cls_name, cls_split in [("good", good_split), ("bad", bad_split)]:
            files = cls_split[split_name]
            dst = DST_ROOT / split_name / cls_name
            copy_split(files, dst)
            counts[split_name][cls_name] = len(files)
            print(f"  {split_name}/{cls_name}: {len(files)} imagenes -> {dst}")

    # Totales
    for split_name in ("train", "val", "test"):
        total = sum(counts[split_name].values())
        print(f"  {split_name} total: {total}")

    # Reporte
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_good = len(good_imgs)
    total_bad = len(bad_imgs)
    total_all = total_good + total_bad

    lines = [
        "# Mendeley Good/Bad Pear — Dataset Report",
        "",
        "## Fuente",
        f"- `{SRC_ROOT / 'good'}`: {total_good} imagenes",
        f"- `{SRC_ROOT / 'bad'}`: {total_bad} imagenes",
        f"- Total: {total_all} imagenes",
        "",
        "## Split (seed=42)",
        "",
        "| Split | good | bad | Total |",
        "|-------|------|-----|-------|",
    ]
    for s in ("train", "val", "test"):
        g = counts[s]["good"]
        b = counts[s]["bad"]
        lines.append(f"| {s} | {g} | {b} | {g+b} |")

    lines += [
        "",
        "## Destino",
        f"`{DST_ROOT}`",
        "",
        "## Clases",
        "- `good` (0 o 1 segun orden alfabetico en YOLO)",
        "- `bad`  (0 o 1 segun orden alfabetico en YOLO)",
        "",
        "## Notas",
        "- Imagenes copiadas (no movidas).",
        "- Archivos ocultos (`.trashed-*`) excluidos.",
        "- Split estratificado por clase con `random.Random(42)`.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Reporte guardado: {REPORT_PATH}")
    print("=== Listo ===")


if __name__ == "__main__":
    main()
