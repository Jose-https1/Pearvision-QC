"""
Convierte el dataset Fruits-360 al formato pear_gate.

Uso dry-run (no copia nada):
    python scripts/convert_fruits360_to_pear_gate.py \
        --input data/raw/external/Fruits360_100x100_original \
        --output data/pear_gate/raw \
        --max-per-class 300 --max-invalid-total 6000 --seed 42 --dry-run

Uso real:
    python scripts/convert_fruits360_to_pear_gate.py \
        --input data/raw/external/Fruits360_100x100_original \
        --output data/pear_gate/raw \
        --max-per-class 300 --max-invalid-total 6000 --seed 42
"""
import argparse
import random
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FRUITS360_SPLITS = ["Training", "Test"]
SPLIT_PREFIX = {"Training": "train_original", "Test": "test_original"}


def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def collect_classes(input_root: Path):
    """
    Devuelve dos dicts {clase_original: [(split_name, Path), ...]}
    separando clases Pear* (valid) de las demás (invalid).
    """
    valid: dict[str, list[tuple[str, Path]]] = {}
    invalid: dict[str, list[tuple[str, Path]]] = {}

    for split_name in FRUITS360_SPLITS:
        split_dir = input_root / split_name
        if not split_dir.exists():
            print(f"  AVISO: no se encontró la carpeta {split_dir}")
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            images = [
                (split_name, p)
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not images:
                continue
            if class_name.startswith("Pear"):
                valid.setdefault(class_name, []).extend(images)
            else:
                invalid.setdefault(class_name, []).extend(images)

    return valid, invalid


def sample_per_class(images: list, max_per_class: int, rng: random.Random) -> list:
    if max_per_class and len(images) > max_per_class:
        return rng.sample(images, max_per_class)
    return list(images)


def sample_invalid_pool(
    invalid_classes: dict[str, list[tuple[str, Path]]],
    max_invalid_total: int,
    rng: random.Random,
) -> dict[str, list[tuple[str, Path]]]:
    """
    Reúne todas las imágenes invalid en un pool plano, mezcla con seed
    y toma como máximo max_invalid_total. Devuelve agrupado por clase.
    """
    pool: list[tuple[str, str, Path]] = []  # (class_name, split_name, path)
    for class_name, images in invalid_classes.items():
        for split_name, path in images:
            pool.append((class_name, split_name, path))

    rng.shuffle(pool)

    if max_invalid_total and len(pool) > max_invalid_total:
        pool = pool[:max_invalid_total]

    sampled: dict[str, list[tuple[str, Path]]] = {}
    for class_name, split_name, path in pool:
        sampled.setdefault(class_name, []).append((split_name, path))

    return sampled


def copy_images(
    class_name: str,
    images: list[tuple[str, Path]],
    dest_root: Path,
    dry_run: bool,
) -> int:
    norm = normalize_name(class_name)
    dest_dir = dest_root / f"fruits360_{norm}"
    copied = 0
    for split_name, src_path in images:
        prefix = SPLIT_PREFIX[split_name]
        new_name = f"{prefix}_{norm}_{src_path.name}"
        dst = dest_dir / new_name
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — convierte Fruits-360 al formato pear_gate"
    )
    parser.add_argument("--input", required=True, help="Raíz de Fruits360_100x100_original")
    parser.add_argument("--output", required=True, help="Raíz de data/pear_gate/raw")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=300,
        help="Máximo de imágenes por clase Pear* (0 = sin límite)",
    )
    parser.add_argument(
        "--max-invalid-total",
        type=int,
        default=6000,
        help="Máximo total de imágenes invalid tras mezclar el pool completo (0 = sin límite)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--dry-run", action="store_true", help="Simula sin copiar nada")
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Vacía valid_pear/ e invalid/ antes de copiar",
    )
    args = parser.parse_args()

    input_root = (PROJECT_ROOT / args.input).resolve()
    output_root = (PROJECT_ROOT / args.output).resolve()

    if not input_root.exists():
        print(f"ERROR: la carpeta de entrada no existe: {input_root}")
        sys.exit(1)

    rng = random.Random(args.seed)

    # Recopilar clases
    valid_classes, invalid_classes = collect_classes(input_root)

    # --- Resumen de clases Pear ---
    print(f"\nClases Pear encontradas ({len(valid_classes)}):")
    for cls in sorted(valid_classes):
        print(f"  {cls}  ({len(valid_classes[cls])} imágenes)")

    total_valid_candidates = sum(len(v) for v in valid_classes.values())
    total_invalid_candidates = sum(len(v) for v in invalid_classes.values())

    print(f"\nTotal candidatas valid_pear : {total_valid_candidates}")
    print(f"Clases invalid encontradas  : {len(invalid_classes)}")
    print(f"Total candidatas invalid    : {total_invalid_candidates}")

    # --- Muestreo ---
    # valid_pear: límite por clase
    valid_selected: dict[str, list] = {
        cls: sample_per_class(imgs, args.max_per_class, rng)
        for cls, imgs in valid_classes.items()
    }
    total_valid_to_copy = sum(len(v) for v in valid_selected.values())

    # invalid: pool plano mezclado con límite total
    invalid_selected = sample_invalid_pool(invalid_classes, args.max_invalid_total, rng)
    total_invalid_to_copy = sum(len(v) for v in invalid_selected.values())

    mode = "copiaría" if args.dry_run else "copiará"
    print(f"\nImágenes que se {mode} a valid_pear : {total_valid_to_copy}")
    print(f"Imágenes que se {mode} a invalid    : {total_invalid_to_copy}")
    print(f"Clases invalid representadas        : {len(invalid_selected)}")

    if args.dry_run:
        print("\n[DRY-RUN activado — no se copiará nada]")

    # --- --clear-output ---
    if args.clear_output and not args.dry_run:
        for label in ("valid_pear", "invalid"):
            target = output_root / label
            if target.exists():
                shutil.rmtree(target)
                print(f"Vaciada: {target}")
    elif args.clear_output and args.dry_run:
        print("[DRY-RUN] Se vaciarían valid_pear/ e invalid/ (--clear-output activo)")

    if args.dry_run:
        print("\n[DRY-RUN completado — ningún archivo fue modificado]")
        return

    # --- Copiar valid_pear ---
    total_valid_copied = 0
    for class_name, images in sorted(valid_selected.items()):
        total_valid_copied += copy_images(
            class_name, images, output_root / "valid_pear", dry_run=False
        )

    # --- Copiar invalid ---
    total_invalid_copied = 0
    for class_name, images in sorted(invalid_selected.items()):
        total_invalid_copied += copy_images(
            class_name, images, output_root / "invalid", dry_run=False
        )

    print(f"\nImágenes copiadas a valid_pear : {total_valid_copied}")
    print(f"Imágenes copiadas a invalid    : {total_invalid_copied}")
    print(f"Clases invalid representadas   : {len(invalid_selected)}")
    print(f"Ruta de salida: {output_root}")
    print("\nConversión completada.")


if __name__ == "__main__":
    main()
