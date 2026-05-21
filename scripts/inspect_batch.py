import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.segmentation import load_image, segment_pear, get_largest_contour

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_yaml(path):
    """Loader mínimo para configs YAML de dos niveles (sin pyyaml)."""
    config = {}
    section = None
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0 and stripped.endswith(":"):
                section = stripped[:-1]
                config[section] = {}
            elif ":" in stripped and section is not None:
                key, _, raw = stripped.partition(":")
                raw = raw.strip()
                if raw.startswith("[") and raw.endswith("]"):
                    value = [int(x.strip()) for x in raw[1:-1].split(",")]
                elif raw.lower() == "true":
                    value = True
                elif raw.lower() == "false":
                    value = False
                else:
                    try:
                        value = int(raw)
                    except ValueError:
                        try:
                            value = float(raw)
                        except ValueError:
                            value = raw
                config[section][key.strip()] = value
    return config


def _save_image(path, image):
    """Guarda imagen con imencode para evitar fallos con tildes en rutas."""
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"No se pudo codificar la imagen de salida: {path}")
    with open(path, "wb") as f:
        f.write(encoded.tobytes())


def _process_one(image_path, seg_cfg, out_dir):
    image = load_image(image_path)
    h, w = image.shape[:2]

    mask = segment_pear(image, seg_cfg)

    contour = get_largest_contour(mask)
    annotated = image.copy()
    if contour is not None:
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)

    stem = image_path.stem
    annotated_path = out_dir / f"seg_{stem}.jpg"
    mask_path = out_dir / f"mask_{stem}.png"

    _save_image(annotated_path, annotated)
    _save_image(mask_path, mask)

    mask_area = int(np.count_nonzero(mask))
    image_area = h * w
    ratio = mask_area / image_area if image_area > 0 else 0.0

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "width": w,
        "height": h,
        "method": seg_cfg.get("method", "hsv"),
        "mask_area_px": mask_area,
        "mask_ratio": f"{ratio:.4f}",
        "annotated_output": str(annotated_path),
        "mask_output": str(mask_path),
        "status": "OK",
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(description="PearVision QC — procesamiento por lotes")
    parser.add_argument("--input", required=True, help="Carpeta con imágenes de entrada")
    parser.add_argument(
        "--output",
        default=None,
        help="Carpeta de salida para imágenes anotadas (por defecto: outputs/annotated)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Ruta del CSV de informe (por defecto: outputs/reports/segmentation_report.csv)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: La carpeta de entrada no existe: {input_dir}")
        sys.exit(1)

    out_dir = (
        Path(args.output) if args.output
        else PROJECT_ROOT / "outputs" / "annotated"
    )
    report_path = (
        Path(args.report) if args.report
        else PROJECT_ROOT / "outputs" / "reports" / "segmentation_report.csv"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    config = _load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    seg_cfg = config.get("segmentation", {})

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No se encontraron imágenes en: {input_dir}")
        sys.exit(0)

    print(f"Imágenes encontradas: {len(images)}")
    print("Procesando...\n")

    fieldnames = [
        "image_name", "image_path", "width", "height", "method",
        "mask_area_px", "mask_ratio", "annotated_output", "mask_output",
        "status", "error",
    ]

    ok_count = 0
    error_count = 0
    rows = []

    for img_path in images:
        print(f"  [{img_path.name}]", end=" ", flush=True)
        try:
            row = _process_one(img_path, seg_cfg, out_dir)
            ok_count += 1
            print(f"OK  (ratio={row['mask_ratio']})")
        except Exception as exc:
            error_count += 1
            row = {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "width": "",
                "height": "",
                "method": seg_cfg.get("method", "hsv"),
                "mask_area_px": "",
                "mask_ratio": "",
                "annotated_output": "",
                "mask_output": "",
                "status": "ERROR",
                "error": str(exc),
            }
            print(f"ERROR: {exc}")
        rows.append(row)

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n--- Resumen ---")
    print(f"Total:      {len(images)}")
    print(f"OK:         {ok_count}")
    print(f"Errores:    {error_count}")
    print(f"CSV:        {report_path}")


if __name__ == "__main__":
    main()
