"""
Importa recortes individuales de cajas YOLO al raw de pear_gate.

A diferencia de import_yolo_dataset_to_pear_gate.py (que copia imágenes
completas), este script recorta cada bounding box y lo guarda como crop
independiente. Evita imágenes ambiguas con varias frutas mezcladas.

Uso dry-run:
    python scripts/import_yolo_crops_to_pear_gate.py `
        --input data/raw/external/roboflow_fruits_vegetables_01_original `
        --output data/pear_gate/raw `
        --dataset-name roboflow_fruits_vegetables_01_crops `
        --valid-classes pear `
        --invalid-classes all_except_valid `
        --padding 0.08 `
        --min-box-size 12 `
        --max-valid 1000 `
        --max-invalid 1000 `
        --seed 42 `
        --dry-run

Uso real (sin --dry-run):
    python scripts/import_yolo_crops_to_pear_gate.py `
        --input data/raw/external/roboflow_fruits_vegetables_01_original `
        --output data/pear_gate/raw `
        --dataset-name roboflow_fruits_vegetables_01_crops `
        --valid-classes pear `
        --invalid-classes all_except_valid `
        --padding 0.08 `
        --min-box-size 12 `
        --max-valid 1000 `
        --max-invalid 1000 `
        --seed 42
"""
import argparse
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_CANDIDATES = ["train", "valid", "val", "test"]
INVALID_ALL = "all_except_valid"


# ---------------------------------------------------------------------------
# Lectura de data.yaml (igual que en import_yolo_dataset_to_pear_gate.py)
# ---------------------------------------------------------------------------

def _parse_yaml_names_simple(text: str) -> dict[int, str]:
    names: dict[int, str] = {}
    m = re.search(r"names\s*:\s*\[([^\]]+)\]", text)
    if m:
        items = [x.strip().strip("'\"") for x in m.group(1).split(",")]
        return {i: v for i, v in enumerate(items) if v}
    in_names = False
    idx = 0
    for line in text.splitlines():
        if re.match(r"names\s*:", line):
            in_names = True
            continue
        if in_names:
            m2 = re.match(r"\s*-\s+(.+)", line)
            if m2:
                names[idx] = m2.group(1).strip().strip("'\"")
                idx += 1
            elif line.strip() and not line.startswith(" "):
                break
    return names


def load_class_names(input_root: Path) -> dict[int, str]:
    yaml_path = input_root / "data.yaml"
    if not yaml_path.exists():
        print("  AVISO: data.yaml no encontrado. Los nombres de clase no estarán disponibles.")
        return {}
    text = yaml_path.read_text(encoding="utf-8", errors="replace")
    try:
        import yaml
        data = yaml.safe_load(text)
        raw = data.get("names", [])
        if isinstance(raw, list):
            return {i: str(v) for i, v in enumerate(raw)}
        if isinstance(raw, dict):
            return {int(k): str(v) for k, v in raw.items()}
    except ImportError:
        print("  AVISO: PyYAML no instalado. Usando parser simple.")
    except Exception as e:
        print(f"  AVISO: error al parsear data.yaml ({e}). Usando parser simple.")
    names = _parse_yaml_names_simple(text)
    if not names:
        print("  AVISO: no se pudieron extraer nombres de clases de data.yaml.")
    return names


# ---------------------------------------------------------------------------
# Detección de splits
# ---------------------------------------------------------------------------

def find_splits(input_root: Path) -> list[tuple[str, Path, Path]]:
    found = []
    for split in SPLIT_CANDIDATES:
        images_dir = input_root / split / "images"
        labels_dir = input_root / split / "labels"
        if images_dir.exists():
            found.append((split, images_dir, labels_dir))
    if not found:
        images_dir = input_root / "images"
        labels_dir = input_root / "labels"
        if images_dir.exists():
            found.append(("root", images_dir, labels_dir))
    return found


# ---------------------------------------------------------------------------
# Lectura segura de imágenes (Unicode-safe para Windows)
# ---------------------------------------------------------------------------

def safe_imread(path: Path):
    try:
        import numpy as np
        import cv2
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def safe_imwrite_jpg(path: Path, img, quality: int = 92) -> bool:
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Estructura de candidato de crop
# ---------------------------------------------------------------------------

@dataclass
class CropCandidate:
    split: str
    img_path: Path
    box_idx: int        # índice de la caja dentro del label
    class_id: int
    class_name: str
    cx: float           # coordenadas normalizadas YOLO
    cy: float
    bw: float
    bh: float


# ---------------------------------------------------------------------------
# Escaneo de candidatos (sin cargar imágenes)
# ---------------------------------------------------------------------------

def scan_candidates(
    splits: list[tuple[str, Path, Path]],
    names: dict[int, str],
    valid_set: set[str],
    invalid_mode: str,      # "all_except_valid" o conjunto de nombres
    invalid_set: set[str],
) -> tuple[list[CropCandidate], list[CropCandidate], int, int, int]:
    """
    Devuelve (valid_cands, invalid_cands, ignored_class, empty_labels, missing_labels).
    No carga imágenes. El filtro min-box-size se aplica después al recortar.
    """
    valid_cands: list[CropCandidate] = []
    invalid_cands: list[CropCandidate] = []
    ignored_class = 0
    empty_labels = 0
    missing_labels = 0

    for split_name, images_dir, labels_dir in splits:
        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                missing_labels += 1
                continue

            lines = [l.strip() for l in label_path.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
            if not lines:
                empty_labels += 1
                continue

            for box_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue

                cname = names.get(cid, f"cls_{cid}").lower()

                is_valid = cname in valid_set
                if is_valid:
                    valid_cands.append(CropCandidate(
                        split_name, img_path, box_idx, cid, cname, cx, cy, bw, bh
                    ))
                    continue

                if invalid_mode == INVALID_ALL:
                    is_invalid = True
                else:
                    is_invalid = cname in invalid_set

                if is_invalid:
                    invalid_cands.append(CropCandidate(
                        split_name, img_path, box_idx, cid, cname, cx, cy, bw, bh
                    ))
                else:
                    ignored_class += 1

    return valid_cands, invalid_cands, ignored_class, empty_labels, missing_labels


# ---------------------------------------------------------------------------
# Recorte y guardado
# ---------------------------------------------------------------------------

def crop_and_save(
    cands: list[CropCandidate],
    dst_dir: Path,
    dataset_name: str,
    padding: float,
    min_box_size: int,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Devuelve (guardados, ignorados_por_tamaño).
    En dry-run no guarda nada.
    """
    try:
        import numpy as np
        import cv2
    except ImportError:
        print("ERROR: cv2 no disponible. Instala opencv-python para recortar imágenes.")
        sys.exit(1)

    saved = 0
    too_small = 0
    last_img_path = None
    last_img = None

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for c in cands:
        # Cargar imagen (cachear si es la misma que la anterior)
        if c.img_path != last_img_path:
            last_img = safe_imread(c.img_path)
            last_img_path = c.img_path

        if last_img is None:
            continue

        h, w = last_img.shape[:2]

        # Coordenadas de la caja con padding
        pad_w = c.bw * padding
        pad_h = c.bh * padding

        x1 = int((c.cx - c.bw / 2 - pad_w) * w)
        y1 = int((c.cy - c.bh / 2 - pad_h) * h)
        x2 = int((c.cx + c.bw / 2 + pad_w) * w)
        y2 = int((c.cy + c.bh / 2 + pad_h) * h)

        # Recortar sin salirse de la imagen
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < min_box_size or crop_h < min_box_size:
            too_small += 1
            continue

        if dry_run:
            saved += 1
            continue

        crop = last_img[y1:y2, x1:x2]
        dst_name = (
            f"{dataset_name}__{c.split}__{c.img_path.stem}"
            f"__box{c.box_idx:03d}__class_{c.class_name}.jpg"
        )
        safe_imwrite_jpg(dst_dir / dst_name, crop)
        saved += 1

    return saved, too_small


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — importa crops YOLO a pear_gate raw"
    )
    parser.add_argument("--input", required=True,
                        help="Carpeta raíz del dataset externo")
    parser.add_argument("--output", required=True,
                        help="Carpeta raíz de data/pear_gate/raw")
    parser.add_argument("--dataset-name", required=True,
                        help="Nombre corto para subcarpetas de destino")
    parser.add_argument("--valid-classes", required=True,
                        help="Clases a guardar como valid_pear (ej. pear)")
    parser.add_argument("--invalid-classes", required=True,
                        help=f"Clases a guardar como invalid. Usa '{INVALID_ALL}' para todo lo demás, "
                             "o lista separada por comas (ej. apple,orange)")
    parser.add_argument("--padding", type=float, default=0.08,
                        help="Relleno relativo alrededor de la caja (default: 0.08)")
    parser.add_argument("--min-box-size", type=int, default=12,
                        help="Tamaño mínimo en píxeles del crop (default: 12)")
    parser.add_argument("--max-valid", type=int, default=0,
                        help="Límite de crops valid_pear (0 = sin límite)")
    parser.add_argument("--max-invalid", type=int, default=0,
                        help="Límite de crops invalid (0 = sin límite)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sin guardar nada")
    parser.add_argument("--clear-output", action="store_true",
                        help="Vacía las subcarpetas de destino antes de guardar")
    args = parser.parse_args()

    input_root = (PROJECT_ROOT / args.input).resolve()
    output_root = (PROJECT_ROOT / args.output).resolve()

    if not input_root.exists():
        print(f"ERROR: la carpeta de entrada no existe: {input_root}")
        sys.exit(1)

    valid_set = {c.strip().lower() for c in args.valid_classes.split(",") if c.strip()}
    raw_invalid = args.invalid_classes.strip().lower()
    invalid_mode = INVALID_ALL if raw_invalid == INVALID_ALL else raw_invalid
    invalid_set: set[str] = set()
    if invalid_mode != INVALID_ALL:
        invalid_set = {c.strip() for c in invalid_mode.split(",") if c.strip()}

    rng = random.Random(args.seed)

    # --- Cargar nombres de clases ---
    print(f"\nDataset de entrada : {input_root}")
    names = load_class_names(input_root)
    if names:
        print(f"Clases detectadas ({len(names)}):")
        for cid, cname in sorted(names.items()):
            tag = ""
            if cname.lower() in valid_set:
                tag = "  <- VÁLIDA"
            elif invalid_mode == INVALID_ALL:
                if cname.lower() not in valid_set:
                    tag = "  <- INVÁLIDA (all_except_valid)"
            elif cname.lower() in invalid_set:
                tag = "  <- INVÁLIDA"
            print(f"  [{cid}] {cname}{tag}")

    print(f"\nClases válidas     : {sorted(valid_set)}")
    if invalid_mode == INVALID_ALL:
        print(f"Clases inválidas   : all_except_valid")
    else:
        print(f"Clases inválidas   : {sorted(invalid_set)}")
    print(f"Padding            : {args.padding}")
    print(f"Tamaño mínimo crop : {args.min_box_size} px")

    # --- Detectar splits ---
    splits = find_splits(input_root)
    if not splits:
        print(f"\nERROR: no se encontraron carpetas images/ en {input_root}")
        sys.exit(1)
    print(f"\nSplits encontrados : {[s[0] for s in splits]}")

    # --- Escanear candidatos (sin cargar imágenes) ---
    print("\nEscaneando etiquetas YOLO...")
    valid_cands, invalid_cands, ignored_class, empty_labels, missing_labels = scan_candidates(
        splits, names, valid_set, invalid_mode, invalid_set
    )

    # --- Aplicar límites con shuffle ---
    rng.shuffle(valid_cands)
    rng.shuffle(invalid_cands)
    if args.max_valid and len(valid_cands) > args.max_valid:
        valid_cands = valid_cands[:args.max_valid]
    if args.max_invalid and len(invalid_cands) > args.max_invalid:
        invalid_cands = invalid_cands[:args.max_invalid]

    # --- Resumen de candidatos ---
    total_boxes = len(valid_cands) + len(invalid_cands) + ignored_class
    print(f"\n{'─' * 55}")
    print(f"Total cajas leídas                  : {total_boxes + empty_labels}")
    print(f"Crops valid_pear candidatos         : {len(valid_cands)}")
    print(f"Crops invalid candidatos            : {len(invalid_cands)}")
    print(f"Cajas ignoradas (otra clase)        : {ignored_class}")
    print(f"Labels vacíos (imagen sin objetos)  : {empty_labels}")
    print(f"Imágenes sin label                  : {missing_labels}")
    print(f"NOTA: el filtro min-box-size ({args.min_box_size}px) se aplica al recortar")
    print(f"{'─' * 55}")

    if args.dry_run:
        print("\n[DRY-RUN activado — no se guardará nada]")

    # --- --clear-output ---
    if args.clear_output:
        for label in ("valid_pear", "invalid"):
            subdir = output_root / label / args.dataset_name
            if subdir.exists():
                if args.dry_run:
                    print(f"[DRY-RUN] Se vaciaría: {subdir}")
                else:
                    shutil.rmtree(subdir)
                    print(f"Vaciada: {subdir}")

    # --- Recortar y guardar ---
    valid_dst = output_root / "valid_pear" / args.dataset_name
    invalid_dst = output_root / "invalid" / args.dataset_name

    print(f"\nProcesando crops valid_pear ({len(valid_cands)} candidatos)...")
    saved_valid, small_valid = crop_and_save(
        valid_cands, valid_dst, args.dataset_name,
        args.padding, args.min_box_size, args.dry_run,
    )

    print(f"Procesando crops invalid ({len(invalid_cands)} candidatos)...")
    saved_invalid, small_invalid = crop_and_save(
        invalid_cands, invalid_dst, args.dataset_name,
        args.padding, args.min_box_size, args.dry_run,
    )

    mode = "guardaría" if args.dry_run else "guardó"
    print(f"\n{'─' * 55}")
    print(f"Crops que se {mode} en valid_pear  : {saved_valid}")
    print(f"  -> {valid_dst}")
    print(f"Crops que se {mode} en invalid     : {saved_invalid}")
    print(f"  -> {invalid_dst}")
    print(f"Crops descartados por tamaño        : {small_valid + small_invalid}")
    print(f"{'─' * 55}")

    if args.dry_run:
        print("\n[DRY-RUN completado — ningún archivo fue modificado]")
    else:
        print("\nImportación de crops completada.")


if __name__ == "__main__":
    main()
