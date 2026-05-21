"""
Comprueba la coherencia del dataset YOLO antes de entrenar.
Uso:
    python scripts/check_yolo_dataset.py --data configs/yolo_pearvision.yaml
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASS_IDS = {0, 1, 2}


def _load_yaml(path):
    config = {}
    section = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0 and ":" in stripped:
                key, _, raw = stripped.partition(":")
                raw = raw.strip()
                if raw == "":
                    section = key.strip()
                    config[section] = {}
                elif raw.startswith("[") and raw.endswith("]"):
                    config[key.strip()] = [x.strip() for x in raw[1:-1].split(",")]
                else:
                    try:
                        config[key.strip()] = int(raw)
                    except ValueError:
                        config[key.strip()] = raw
            elif indent > 0 and section is not None and ":" in stripped:
                key, _, raw = stripped.partition(":")
                raw = raw.strip()
                try:
                    config[section][int(key.strip())] = raw
                except ValueError:
                    config[section][key.strip()] = raw
    return config


def check_split(split_name, images_dir, labels_dir, nc):
    print(f"\n--- {split_name.upper()} ---")

    if not images_dir.exists():
        print(f"  AVISO: carpeta no encontrada: {images_dir}")
        return

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    labels = sorted(p for p in labels_dir.iterdir() if p.suffix == ".txt") if labels_dir.exists() else []

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    print(f"  Imágenes encontradas : {len(images)}")
    print(f"  Etiquetas encontradas: {len(labels)}")

    # Imágenes sin etiqueta
    sin_label = image_stems - label_stems
    if sin_label:
        print(f"  AVISO: {len(sin_label)} imagen(es) sin .txt de etiqueta:")
        for s in sorted(sin_label)[:10]:
            print(f"    - {s}")
        if len(sin_label) > 10:
            print(f"    ... y {len(sin_label) - 10} más")
    else:
        print("  OK: todas las imágenes tienen etiqueta")

    # Etiquetas sin imagen
    sin_imagen = label_stems - image_stems
    if sin_imagen:
        print(f"  AVISO: {len(sin_imagen)} etiqueta(s) sin imagen correspondiente:")
        for s in sorted(sin_imagen)[:10]:
            print(f"    - {s}")
    else:
        print("  OK: todas las etiquetas tienen imagen")

    # Revisión de contenido de cada .txt
    errores_formato = 0
    errores_clase = 0
    errores_coords = 0
    vacios = 0

    for label_path in labels:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            vacios += 1
            continue
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errores_formato += 1
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                errores_formato += 1
                continue
            if cls not in VALID_CLASS_IDS:
                errores_clase += 1
            for val in (cx, cy, w, h):
                if not (0.0 <= val <= 1.0):
                    errores_coords += 1
                    break

    print(f"  Archivos .txt vacíos (good_pear): {vacios}")
    if errores_formato:
        print(f"  ERROR: {errores_formato} línea(s) con formato incorrecto (esperado: class cx cy w h)")
    else:
        print("  OK: formato de líneas correcto")
    if errores_clase:
        print(f"  ERROR: {errores_clase} línea(s) con class_id fuera de rango [0..{nc - 1}]")
    else:
        print("  OK: class_id dentro de rango")
    if errores_coords:
        print(f"  ERROR: {errores_coords} línea(s) con coordenadas fuera de [0.0..1.0]")
    else:
        print("  OK: coordenadas normalizadas correctas")


def main():
    parser = argparse.ArgumentParser(description="PearVision QC — verificación de dataset YOLO")
    parser.add_argument("--data", required=True, help="Ruta al archivo YAML del dataset")
    args = parser.parse_args()

    yaml_path = PROJECT_ROOT / args.data
    if not yaml_path.exists():
        print(f"ERROR: no se encontró el archivo de configuración: {yaml_path}")
        sys.exit(1)

    cfg = _load_yaml(yaml_path)
    data_root = (yaml_path.parent / cfg.get("path", ".")).resolve()
    nc = int(cfg.get("nc", 3))

    print(f"Dataset root : {data_root}")
    print(f"Clases (nc)  : {nc}")

    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if not rel:
            continue
        images_dir = data_root / rel
        labels_dir = images_dir.parent.parent / split / "labels"
        check_split(split, images_dir, labels_dir, nc)

    print("\n--- FIN DEL CHEQUEO ---")


if __name__ == "__main__":
    main()
