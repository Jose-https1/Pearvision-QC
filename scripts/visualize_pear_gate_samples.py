"""
Genera mosaicos de muestras aleatorias del dataset pear_gate para revisión visual.
Compatible con rutas Windows que contienen acentos u otros caracteres Unicode.

Uso:
    python scripts/visualize_pear_gate_samples.py \
        --root data/pear_gate --split train --max-per-class 40 \
        --seed 42 --output outputs/pear_gate_preview
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

THUMB_SIZE = 100
COLS = 10
LABEL_HEIGHT = 14
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.28
FONT_THICKNESS = 1
FONT_COLOR = (255, 255, 255)
BG_LABEL = (30, 30, 30)


def safe_imread(path: Path) -> np.ndarray | None:
    """Lee una imagen con np.fromfile para evitar fallos en rutas Unicode de Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def safe_imwrite(path: Path, image: np.ndarray) -> bool:
    """Escribe una imagen con buffer.tofile para evitar fallos en rutas Unicode de Windows."""
    try:
        ok, buffer = cv2.imencode(path.suffix, image)
        if not ok:
            return False
        buffer.tofile(str(path))
        return True
    except Exception:
        return False


def collect_images(folder: Path) -> list[Path]:
    imgs = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(imgs)


def load_thumbnail(path: Path, size: int) -> tuple[np.ndarray, bool]:
    """Devuelve (miniatura, exito). Si falla, devuelve un placeholder rojo y exito=False."""
    img = safe_imread(path)
    if img is None:
        placeholder = np.full((size, size, 3), 80, dtype=np.uint8)
        cv2.line(placeholder, (0, 0), (size, size), (0, 0, 200), 2)
        cv2.line(placeholder, (size, 0), (0, size), (0, 0, 200), 2)
        return placeholder, False
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), True


def make_cell(path: Path, thumb_size: int, label_height: int) -> tuple[np.ndarray, bool]:
    thumb, ok = load_thumbnail(path, thumb_size)
    label_band = np.full((label_height, thumb_size, 3), BG_LABEL, dtype=np.uint8)
    name = path.stem[:18]
    cv2.putText(
        label_band, name,
        (1, label_height - 3),
        FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return np.vstack([thumb, label_band]), ok


def build_mosaic(
    images: list[Path],
    cols: int,
    thumb_size: int,
    label_height: int,
    class_name: str,
) -> tuple[np.ndarray, int, int]:
    """
    Construye el mosaico. Devuelve (mosaico, leidas_ok, leidas_fallidas).
    """
    cell_h = thumb_size + label_height
    cell_w = thumb_size
    rows = (len(images) + cols - 1) // cols

    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    ok_count = 0
    fail_count = 0
    for idx, path in enumerate(images):
        r = idx // cols
        c = idx % cols
        cell, ok = make_cell(path, thumb_size, label_height)
        mosaic[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell
        if ok:
            ok_count += 1
        else:
            fail_count += 1

    header_h = 24
    header = np.full((header_h, mosaic.shape[1], 3), (50, 50, 50), dtype=np.uint8)
    label_text = f"{class_name}  ({len(images)} muestras)"
    cv2.putText(
        header, label_text,
        (6, 17),
        FONT, 0.55, (200, 230, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([header, mosaic]), ok_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — visualizador de muestras pear_gate"
    )
    parser.add_argument("--root", required=True, help="Raíz del dataset pear_gate (ej. data/pear_gate)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                        help="Split a visualizar (default: train)")
    parser.add_argument("--max-per-class", type=int, default=40,
                        help="Número máximo de muestras por clase (default: 40)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
    parser.add_argument("--output", required=True, help="Carpeta de salida para los mosaicos")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    output_dir = (PROJECT_ROOT / args.output).resolve()

    if not root.exists():
        print(f"ERROR: la carpeta raíz no existe: {root}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print(f"Split      : {args.split}")
    print(f"Muestras   : hasta {args.max_per_class} por clase")
    print(f"Semilla    : {args.seed}")
    print(f"Salida     : {output_dir}")

    for cls in ("valid_pear", "invalid"):
        folder = root / args.split / cls
        if not folder.exists():
            print(f"\nAVISO: carpeta no encontrada, omitiendo — {folder}")
            continue

        all_imgs = collect_images(folder)
        print(f"\nClase '{cls}': {len(all_imgs)} imágenes encontradas en {args.split}/{cls}")

        if not all_imgs:
            print(f"  AVISO: {args.split}/{cls} está vacío, omitiendo.")
            continue

        sample = rng.sample(all_imgs, min(args.max_per_class, len(all_imgs)))
        sample.sort()

        mosaic, ok_count, fail_count = build_mosaic(
            sample, COLS, THUMB_SIZE, LABEL_HEIGHT, f"{args.split}/{cls}"
        )

        print(f"  Leídas correctamente : {ok_count}")
        print(f"  Fallidas (omitidas)  : {fail_count}")

        if ok_count == 0:
            print(f"  ERROR: todas las imágenes fallaron al leer. No se genera mosaico para '{cls}'.")
            continue

        out_name = f"{args.split}_{cls}_preview.jpg"
        out_path = output_dir / out_name

        if safe_imwrite(out_path, mosaic):
            print(f"  Guardado: {out_path}")
        else:
            print(f"  ERROR: no se pudo guardar {out_path}")

    print("\nListo. Abre los archivos JPG para revisar las muestras.")


if __name__ == "__main__":
    main()
