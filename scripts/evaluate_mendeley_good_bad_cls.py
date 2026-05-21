"""
evaluate_mendeley_good_bad_cls.py
Evalua el clasificador good/bad sobre un directorio de imagenes.

Uso:
  python scripts/evaluate_mendeley_good_bad_cls.py \
      --model runs/pear_quality_cls/mendeley_good_bad_v1/weights/best.pt \
      --source data/samples_quality_controlled_test \
      --output outputs/mendeley_good_bad_eval

Salidas:
  <output>/predictions.csv
  <output>/prediction_grid.jpg
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GRID_CELL = 224   # tamaño de cada celda en la grid


def collect_images(source: Path) -> list:
    if source.is_file():
        return [source]
    imgs = []
    for ext in VALID_EXT:
        imgs.extend(source.glob(f"*{ext}"))
        imgs.extend(source.glob(f"*{ext.upper()}"))
    return sorted(set(imgs))


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))


def make_grid(cells: list, cols: int = 4) -> np.ndarray:
    """Une celdas BGR en una grid de N columnas."""
    rows = (len(cells) + cols - 1) // cols
    # padding con negro si faltan celdas
    while len(cells) < rows * cols:
        cells.append(np.zeros((GRID_CELL, GRID_CELL, 3), dtype=np.uint8))
    grid_rows = []
    for r in range(rows):
        row_imgs = cells[r * cols: (r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    return np.vstack(grid_rows)


def main():
    parser = argparse.ArgumentParser(description="Evalua clasificador good/bad Mendeley")
    parser.add_argument("--model", type=Path, required=True, help="Ruta a best.pt")
    parser.add_argument("--source", type=Path, required=True, help="Carpeta con imagenes a evaluar")
    parser.add_argument("--output", type=Path, default=Path("outputs/mendeley_good_bad_eval"))
    args = parser.parse_args()

    model_path = (PROJECT_ROOT / args.model).resolve()
    source_path = (PROJECT_ROOT / args.source).resolve()
    out_dir = (PROJECT_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: modelo no encontrado: {model_path}")
        sys.exit(1)
    if not source_path.exists():
        print(f"ERROR: source no encontrado: {source_path}")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # Determinar índice de clase good y bad según nombres del modelo
    class_names = model.names  # dict {0: 'bad', 1: 'good'} o similar
    good_idx = None
    bad_idx = None
    for idx, name in class_names.items():
        if name.lower() == "good":
            good_idx = idx
        elif name.lower() == "bad":
            bad_idx = idx

    if good_idx is None or bad_idx is None:
        print(f"AVISO: nombres de clase no reconocidos: {class_names}")
        print("  Asumiendo: 0=bad, 1=good")
        bad_idx = 0
        good_idx = 1

    print(f"  Clases del modelo: {class_names}")
    print(f"  good_idx={good_idx}  bad_idx={bad_idx}")

    images = collect_images(source_path)
    if not images:
        print(f"ERROR: no se encontraron imagenes en {source_path}")
        sys.exit(1)

    print(f"  Imagenes a evaluar: {len(images)}")

    rows = []
    cells = []

    for img_path in images:
        results = model.predict(
            str(img_path), imgsz=GRID_CELL, device="cpu", verbose=False
        )
        probs = results[0].probs
        if probs is None:
            continue

        all_probs = probs.data.cpu().numpy()
        good_conf = float(all_probs[good_idx]) if good_idx < len(all_probs) else 0.0
        bad_conf = float(all_probs[bad_idx]) if bad_idx < len(all_probs) else 0.0
        predicted_idx = int(probs.top1)
        predicted_class = class_names.get(predicted_idx, str(predicted_idx))
        max_conf = float(probs.top1conf.cpu().numpy())

        rows.append({
            "image": img_path.name,
            "predicted_class": predicted_class,
            "good_conf": round(good_conf, 4),
            "bad_conf": round(bad_conf, 4),
            "max_conf": round(max_conf, 4),
        })

        print(f"  {img_path.name:<30} -> {predicted_class.upper():4s} {max_conf:.2f}")

        # Construir celda para la grid
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            bgr = np.zeros((GRID_CELL, GRID_CELL, 3), dtype=np.uint8)
        cell = cv2.resize(bgr, (GRID_CELL, GRID_CELL), interpolation=cv2.INTER_AREA)

        # Color del texto: verde para GOOD, rojo para BAD
        if predicted_class.lower() == "good":
            text_color = (0, 220, 0)
        else:
            text_color = (0, 0, 220)

        label = f"{predicted_class.upper()} {max_conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cell, label, (6, GRID_CELL - 10), font, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cell, label, (6, GRID_CELL - 10), font, 0.52, text_color, 1, cv2.LINE_AA)

        cells.append(cell)

    # Guardar CSV
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "predicted_class", "good_conf", "bad_conf", "max_conf"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV guardado: {csv_path}")

    # Guardar grid
    if cells:
        cols = min(4, len(cells))
        grid = make_grid(cells, cols=cols)
        grid_path = out_dir / "prediction_grid.jpg"
        imwrite_unicode(grid_path, grid)
        print(f"  Grid guardada: {grid_path}")

    # Resumen
    good_count = sum(1 for r in rows if r["predicted_class"].lower() == "good")
    bad_count = sum(1 for r in rows if r["predicted_class"].lower() == "bad")
    print(f"\n  Resumen: GOOD={good_count}  BAD={bad_count}  Total={len(rows)}")


if __name__ == "__main__":
    main()
