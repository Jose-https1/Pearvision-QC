"""
Visualiza etiquetas YOLO dibujando cajas sobre las imágenes y guarda el resultado.

Uso:
    python scripts/visualize_yolo_labels.py \
        --images data/train/images \
        --labels data/train/labels \
        --output outputs/annotated/yolo_labels_preview \
        --max 20
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = {
    0: "mechanical_damage",
    1: "rot",
    2: "twig_mark",
}

COLORS = {
    0: (0, 80, 255),    # naranja-rojo  mechanical_damage
    1: (0, 180, 0),     # verde         rot
    2: (200, 0, 200),   # morado        twig_mark
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_boxes(img, label_path):
    h, w = img.shape[:2]
    has_boxes = False

    if not label_path.exists():
        return img, False

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    lines = [l for l in lines if l.strip()]

    if not lines:
        return img, False

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = COLORS.get(cls, (128, 128, 128))
        label_text = CLASS_NAMES.get(cls, f"cls_{cls}")

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        bg_y1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label_text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
        has_boxes = True

    return img, has_boxes


def add_no_defect_overlay(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 40), (50, 50, 50), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(
        img, "sin defectos anotados",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA,
    )
    return img


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — visualizador de etiquetas YOLO"
    )
    parser.add_argument("--images", required=True, help="Carpeta de imágenes")
    parser.add_argument("--labels", required=True, help="Carpeta de labels (.txt YOLO)")
    parser.add_argument("--output", required=True, help="Carpeta de salida para imágenes anotadas")
    parser.add_argument("--max", type=int, default=20, help="Número máximo de imágenes a procesar")
    args = parser.parse_args()

    images_dir = (PROJECT_ROOT / args.images).resolve()
    labels_dir = (PROJECT_ROOT / args.labels).resolve()
    output_dir = (PROJECT_ROOT / args.output).resolve()

    if not images_dir.exists():
        print(f"ERROR: la carpeta de imágenes no existe: {images_dir}")
        sys.exit(1)

    images = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )[: args.max]

    if not images:
        print(f"No se encontraron imágenes en: {images_dir}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Imágenes      : {images_dir}")
    print(f"Labels        : {labels_dir}")
    print(f"Salida        : {output_dir}")
    print(f"Procesando    : {len(images)} imagen(es)")
    print()

    n_con_cajas = 0
    n_sin_cajas = 0
    n_error = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ERROR al leer: {img_path.name}")
            n_error += 1
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        img_annotated, has_boxes = draw_boxes(img.copy(), label_path)

        if has_boxes:
            n_con_cajas += 1
        else:
            img_annotated = add_no_defect_overlay(img_annotated)
            n_sin_cajas += 1

        dst = output_dir / img_path.name
        cv2.imwrite(str(dst), img_annotated)
        estado = "con cajas" if has_boxes else "sin defectos"
        print(f"  [{estado:12s}]  {img_path.name}  ->  {dst.name}")

    print()
    print(f"Guardadas     : {n_con_cajas + n_sin_cajas} imágenes en {output_dir}")
    print(f"Con cajas     : {n_con_cajas}")
    print(f"Sin defectos  : {n_sin_cajas}")
    if n_error:
        print(f"Errores       : {n_error}")
    print("\nRevisa las imágenes en la carpeta de salida para verificar el mapeo visualmente.")


if __name__ == "__main__":
    main()
