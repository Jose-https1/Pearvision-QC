"""
Importa imágenes de un dataset externo YOLO/Roboflow al raw de pear_gate.

Las etiquetas YOLO se usan solo para clasificar cada imagen como
valid_pear o invalid. Las cajas NO se copian.

Uso dry-run:
    python scripts/import_yolo_dataset_to_pear_gate.py `
        --input data/raw/external/ROB_DATASET_ORIGINAL `
        --output data/pear_gate/raw `
        --dataset-name roboflow_fruits_vegetables_01 `
        --valid-classes pear `
        --invalid-classes apple,orange,banana,pineapple,tomato `
        --dry-run

Uso real:
    python scripts/import_yolo_dataset_to_pear_gate.py `
        --input data/raw/external/ROB_DATASET_ORIGINAL `
        --output data/pear_gate/raw `
        --dataset-name roboflow_fruits_vegetables_01 `
        --valid-classes pear `
        --invalid-classes apple,orange,banana,pineapple,tomato
"""
import argparse
import random
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Nombres de split que se buscan en el dataset de entrada
SPLIT_CANDIDATES = ["train", "valid", "val", "test"]


# ---------------------------------------------------------------------------
# Lectura de data.yaml
# ---------------------------------------------------------------------------

def _parse_yaml_names_simple(text: str) -> dict[int, str]:
    """
    Parseo mínimo de data.yaml sin PyYAML.
    Soporta:
      names: [cat, dog, pear]
      names:
        - cat
        - dog
    """
    names: dict[int, str] = {}

    # Forma lista en una línea: names: [a, b, c]
    m = re.search(r"names\s*:\s*\[([^\]]+)\]", text)
    if m:
        items = [x.strip().strip("'\"") for x in m.group(1).split(",")]
        return {i: v for i, v in enumerate(items) if v}

    # Forma multilínea con guiones
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
        print("  AVISO: data.yaml no encontrado. Las clases no se mapearán por nombre.")
        return {}

    text = yaml_path.read_text(encoding="utf-8", errors="replace")

    try:
        import yaml
        data = yaml.safe_load(text)
        raw_names = data.get("names", [])
        if isinstance(raw_names, list):
            return {i: str(v) for i, v in enumerate(raw_names)}
        if isinstance(raw_names, dict):
            return {int(k): str(v) for k, v in raw_names.items()}
    except ImportError:
        print("  AVISO: PyYAML no instalado. Leyendo data.yaml con parser simple.")
    except Exception as e:
        print(f"  AVISO: error al parsear data.yaml con PyYAML ({e}). Usando parser simple.")

    names = _parse_yaml_names_simple(text)
    if not names:
        print("  AVISO: no se pudieron extraer nombres de clases de data.yaml.")
    return names


# ---------------------------------------------------------------------------
# Detección de splits
# ---------------------------------------------------------------------------

def find_splits(input_root: Path) -> list[tuple[str, Path, Path]]:
    """
    Devuelve lista de (split_name, images_dir, labels_dir).
    Busca estructuras tipo <split>/images/ y <split>/labels/.
    Si no encuentra splits, prueba images/ y labels/ directamente en input_root.
    """
    found = []
    for split in SPLIT_CANDIDATES:
        images_dir = input_root / split / "images"
        labels_dir = input_root / split / "labels"
        if images_dir.exists():
            found.append((split, images_dir, labels_dir))

    if not found:
        # Fallback: dataset sin subcarpetas de split
        images_dir = input_root / "images"
        labels_dir = input_root / "labels"
        if images_dir.exists():
            found.append(("root", images_dir, labels_dir))

    return found


# ---------------------------------------------------------------------------
# Lectura de etiquetas YOLO
# ---------------------------------------------------------------------------

def read_label_class_ids(label_path: Path) -> set[int] | None:
    """
    Devuelve el conjunto de class_id presentes en el fichero de etiqueta.
    Devuelve None si el fichero no existe.
    Devuelve set vacío si el fichero existe pero está vacío (imagen sin objetos).
    """
    if not label_path.exists():
        return None
    ids: set[int] = set()
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                ids.add(int(parts[0]))
    except Exception:
        pass
    return ids


# ---------------------------------------------------------------------------
# Clasificación de imagen
# ---------------------------------------------------------------------------

def classify_image(
    class_ids: set[int],
    names: dict[int, str],
    valid_set: set[str],
    invalid_set: set[str],
    allow_mixed_valid: bool,
) -> str:
    """
    Devuelve: 'valid', 'invalid', 'ambiguous', 'sin_label', 'ignorar'.
    """
    if not class_ids and class_ids is not None:
        # fichero vacío = imagen sin objetos → ignorar
        return "ignorar"

    present_names = {names.get(cid, f"cls_{cid}").lower() for cid in class_ids}

    has_valid = bool(present_names & valid_set)
    has_invalid = bool(present_names & invalid_set)

    if has_valid and has_invalid:
        if allow_mixed_valid:
            return "valid"
        return "ambiguous"
    if has_valid:
        return "valid"
    if has_invalid:
        return "invalid"
    return "ignorar"


# ---------------------------------------------------------------------------
# Copia
# ---------------------------------------------------------------------------

def safe_dst_name(dataset_name: str, split: str, src: Path) -> str:
    """Nombre de destino con prefijo para evitar colisiones."""
    return f"{dataset_name}__{split}__{src.name}"


def do_copy(src: Path, dst_dir: Path, dst_name: str, dry_run: bool) -> bool:
    dst = dst_dir / dst_name
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — importa dataset YOLO externo a pear_gate raw"
    )
    parser.add_argument("--input", required=True,
                        help="Carpeta raíz del dataset externo (con data.yaml)")
    parser.add_argument("--output", required=True,
                        help="Carpeta raíz de data/pear_gate/raw")
    parser.add_argument("--dataset-name", required=True,
                        help="Nombre corto para subcarpeta de destino (ej. roboflow_fruits_01)")
    parser.add_argument("--valid-classes", required=True,
                        help="Clases a copiar como valid_pear, separadas por coma (ej. pear)")
    parser.add_argument("--invalid-classes", required=True,
                        help="Clases a copiar como invalid, separadas por coma (ej. apple,orange)")
    parser.add_argument("--allow-mixed-valid", action="store_true",
                        help="Si una imagen mezcla clases válidas e inválidas, copiarla como valid_pear")
    parser.add_argument("--max-valid", type=int, default=0,
                        help="Límite de imágenes valid_pear (0 = sin límite)")
    parser.add_argument("--max-invalid", type=int, default=0,
                        help="Límite de imágenes invalid (0 = sin límite)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--clear-output", action="store_true",
                        help="Vacía las subcarpetas de destino antes de copiar")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sin copiar nada")
    args = parser.parse_args()

    input_root = (PROJECT_ROOT / args.input).resolve()
    output_root = (PROJECT_ROOT / args.output).resolve()

    if not input_root.exists():
        print(f"ERROR: la carpeta de entrada no existe: {input_root}")
        sys.exit(1)

    valid_set = {c.strip().lower() for c in args.valid_classes.split(",") if c.strip()}
    invalid_set = {c.strip().lower() for c in args.invalid_classes.split(",") if c.strip()}
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
            elif cname.lower() in invalid_set:
                tag = "  <- INVÁLIDA"
            print(f"  [{cid}] {cname}{tag}")
    else:
        print("  Sin mapa de clases. Solo se usarán IDs si --valid-classes / --invalid-classes son numéricos.")

    print(f"\nClases válidas elegidas   : {sorted(valid_set)}")
    print(f"Clases inválidas elegidas : {sorted(invalid_set)}")

    # --- Detectar splits ---
    splits = find_splits(input_root)
    if not splits:
        print(f"\nERROR: no se encontraron carpetas images/ en {input_root}")
        sys.exit(1)
    print(f"\nSplits encontrados: {[s[0] for s in splits]}")

    # --- Recopilar candidatas ---
    valid_candidates: list[tuple[str, Path]] = []   # (split, img_path)
    invalid_candidates: list[tuple[str, Path]] = []
    ambiguous_count = 0
    sin_label_count = 0
    ignorar_count = 0

    for split_name, images_dir, labels_dir in splits:
        for img in sorted(images_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / (img.stem + ".txt")
            class_ids = read_label_class_ids(label_path)

            if class_ids is None:
                sin_label_count += 1
                continue

            decision = classify_image(class_ids, names, valid_set, invalid_set,
                                      args.allow_mixed_valid)
            if decision == "valid":
                valid_candidates.append((split_name, img))
            elif decision == "invalid":
                invalid_candidates.append((split_name, img))
            elif decision == "ambiguous":
                ambiguous_count += 1
            else:
                ignorar_count += 1

    # --- Aplicar límites con shuffle ---
    rng.shuffle(valid_candidates)
    rng.shuffle(invalid_candidates)

    if args.max_valid and len(valid_candidates) > args.max_valid:
        valid_candidates = valid_candidates[:args.max_valid]
    if args.max_invalid and len(invalid_candidates) > args.max_invalid:
        invalid_candidates = invalid_candidates[:args.max_invalid]

    # --- Resumen de candidatas ---
    print(f"\n{'─' * 50}")
    print(f"Imágenes candidatas valid_pear : {len(valid_candidates)}")
    print(f"Imágenes candidatas invalid    : {len(invalid_candidates)}")
    print(f"Imágenes ambiguas (omitidas)   : {ambiguous_count}")
    print(f"Imágenes sin label             : {sin_label_count}")
    print(f"Imágenes ignoradas (otra clase): {ignorar_count}")
    print(f"{'─' * 50}")

    if args.dry_run:
        print("\n[DRY-RUN activado — no se copiará nada]")

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

    # --- Copiar ---
    valid_dst = output_root / "valid_pear" / args.dataset_name
    invalid_dst = output_root / "invalid" / args.dataset_name

    total_valid_copied = 0
    for split_name, img in valid_candidates:
        dst_name = safe_dst_name(args.dataset_name, split_name, img)
        do_copy(img, valid_dst, dst_name, args.dry_run)
        total_valid_copied += 1

    total_invalid_copied = 0
    for split_name, img in invalid_candidates:
        dst_name = safe_dst_name(args.dataset_name, split_name, img)
        do_copy(img, invalid_dst, dst_name, args.dry_run)
        total_invalid_copied += 1

    mode = "copiaría" if args.dry_run else "copió"
    print(f"\nImágenes que se {mode} a valid_pear : {total_valid_copied}")
    print(f"  -> {valid_dst}")
    print(f"Imágenes que se {mode} a invalid    : {total_invalid_copied}")
    print(f"  -> {invalid_dst}")

    if args.dry_run:
        print("\n[DRY-RUN completado — ningún archivo fue modificado]")
    else:
        print("\nImportación completada.")


if __name__ == "__main__":
    main()
