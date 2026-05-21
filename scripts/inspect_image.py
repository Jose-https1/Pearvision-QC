import argparse
import sys
from pathlib import Path

# Añade la raíz del proyecto al path para que "src" sea importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.segmentation import load_image, segment_pear, get_largest_contour


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


def main():
    parser = argparse.ArgumentParser(description="PearVision QC — segmentación")
    parser.add_argument("--image", required=True, help="Ruta a la imagen de entrada")
    parser.add_argument("--debug", action="store_true", help="Guardar máscaras intermedias de debug")
    args = parser.parse_args()

    config = _load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    seg_cfg = config.get("segmentation", {})

    image = load_image(args.image)

    # Máscara raw sin despegado de sombra (solo útil en modo HSV, para comparación)
    if args.debug:
        raw_cfg = dict(seg_cfg)
        raw_cfg["use_shadow_detach"] = False
        mask_raw = segment_pear(image, raw_cfg)

    # Máscara final: pipeline completo según method configurado
    mask_final = segment_pear(image, seg_cfg)

    contour = get_largest_contour(mask_final)
    annotated = image.copy()
    if contour is not None:
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)

    out_dir = PROJECT_ROOT / "outputs" / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.image).stem
    suffix = Path(args.image).suffix or ".jpg"
    out_path = out_dir / f"seg_{stem}{suffix}"
    mask_path = out_dir / f"mask_{stem}.png"

    _save_image(out_path, annotated)
    _save_image(mask_path, mask_final)

    mask_area = int(np.count_nonzero(mask_final))
    image_area = image.shape[0] * image.shape[1]
    ratio = mask_area / image_area if image_area > 0 else 0.0

    print(f"Imagen:        {args.image}")
    print(f"Método:        {seg_cfg.get('method', 'hsv')}")
    print(f"Área máscara:  {mask_area} px")
    print(f"Ratio máscara: {ratio:.3f}")
    print(f"Salida:        {out_path}")
    print(f"Máscara:       {mask_path}")

    if args.debug:
        raw_path = out_dir / f"mask_raw_{stem}.png"
        final_path = out_dir / f"mask_final_{stem}.png"
        _save_image(raw_path, mask_raw)
        _save_image(final_path, mask_final)
        print(f"Debug raw:     {raw_path}")
        print(f"Debug final:   {final_path}")


if __name__ == "__main__":
    main()
