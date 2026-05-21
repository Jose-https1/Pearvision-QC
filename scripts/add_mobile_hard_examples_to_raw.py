"""
add_mobile_hard_examples_to_raw.py

Añade imágenes móviles como hard examples al directorio raw del pear_gate.
Las peras van a valid_pear/<dataset-name>_valid
Las no-peras van a invalid/<dataset-name>_invalid

Uso:
    python scripts/add_mobile_hard_examples_to_raw.py \
        --valid-source data/samples_gate_unseen/valid_pear_mobile \
        --invalid-source data/samples_gate_unseen/invalid_mobile \
        --output data/pear_gate/raw \
        --dataset-name mobile_hard_v6 \
        --repeat-valid 10 \
        --repeat-invalid 10 \
        --dry-run
"""

import argparse
import shutil
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    images = []
    for ext in VALID_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def copy_with_repeats(
    images: list[Path],
    dest_dir: Path,
    repeat: int,
    dry_run: bool,
) -> int:
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in images:
        stem = img.stem
        suffix = img.suffix.lower()
        for rep in range(repeat):
            dest_name = f"{stem}__rep{rep:02d}{suffix}"
            dest_path = dest_dir / dest_name
            if not dry_run:
                shutil.copy2(img, dest_path)
            count += 1

    return count


def clear_dataset_folders(valid_dest: Path, invalid_dest: Path) -> None:
    for folder in (valid_dest, invalid_dest):
        if folder.exists():
            shutil.rmtree(folder)
            print(f"  Borrada: {folder}")
        else:
            print(f"  No existía (skip): {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Añade imágenes móviles como hard examples al raw del pear_gate."
    )
    parser.add_argument(
        "--valid-source",
        type=Path,
        default=Path("data/samples_gate_unseen/valid_pear_mobile"),
        help="Carpeta con peras móviles nuevas",
    )
    parser.add_argument(
        "--invalid-source",
        type=Path,
        default=Path("data/samples_gate_unseen/invalid_mobile"),
        help="Carpeta con no-peras móviles nuevas",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pear_gate/raw"),
        help="Directorio raíz de raw (por defecto: data/pear_gate/raw)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Nombre base para las subcarpetas creadas",
    )
    parser.add_argument(
        "--repeat-valid",
        type=int,
        default=1,
        help="Repeticiones por imagen de pera (por defecto: 1)",
    )
    parser.add_argument(
        "--repeat-invalid",
        type=int,
        default=1,
        help="Repeticiones por imagen de no-pera (por defecto: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar qué se haría sin copiar nada",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help=(
            "Borrar las dos carpetas de salida de este dataset-name antes de copiar. "
            "NUNCA borra todo raw, solo las subcarpetas de este dataset-name."
        ),
    )
    args = parser.parse_args()

    valid_dest = args.output / "valid_pear" / f"{args.dataset_name}_valid"
    invalid_dest = args.output / "invalid" / f"{args.dataset_name}_invalid"

    print("=" * 60)
    print("PearVision QC — add_mobile_hard_examples_to_raw")
    print("=" * 60)
    print(f"  valid-source  : {args.valid_source}")
    print(f"  invalid-source: {args.invalid_source}")
    print(f"  output        : {args.output}")
    print(f"  dataset-name  : {args.dataset_name}")
    print(f"  repeat-valid  : {args.repeat_valid}")
    print(f"  repeat-invalid: {args.repeat_invalid}")
    print(f"  dry-run       : {args.dry_run}")
    print(f"  clear-output  : {args.clear_output}")
    print()

    # Validar que las fuentes existan
    if not args.valid_source.exists():
        print(f"[ERROR] valid-source no existe: {args.valid_source}")
        raise SystemExit(1)
    if not args.invalid_source.exists():
        print(f"[ERROR] invalid-source no existe: {args.invalid_source}")
        raise SystemExit(1)

    # Recoger imágenes
    valid_images = collect_images(args.valid_source)
    invalid_images = collect_images(args.invalid_source)

    print(f"Imágenes válidas encontradas  : {len(valid_images)}")
    print(f"Imágenes inválidas encontradas: {len(invalid_images)}")
    print()

    total_valid = len(valid_images) * args.repeat_valid
    total_invalid = len(invalid_images) * args.repeat_invalid

    print(f"Copias válidas  que se {'crearían' if args.dry_run else 'crearán'}: {total_valid}")
    print(f"Copias inválidas que se {'crearían' if args.dry_run else 'crearán'}: {total_invalid}")
    print()
    print(f"Carpeta de salida válidas  : {valid_dest}")
    print(f"Carpeta de salida inválidas: {invalid_dest}")
    print()

    if args.clear_output:
        if args.dry_run:
            print("[DRY-RUN] Se borrarían las carpetas:")
            print(f"  {valid_dest}")
            print(f"  {invalid_dest}")
        else:
            print("Borrando carpetas de este dataset-name...")
            clear_dataset_folders(valid_dest, invalid_dest)
        print()

    if args.dry_run:
        print("[DRY-RUN] No se copia nada. Elimina --dry-run para ejecutar.")
        return

    # Copiar
    copied_valid = copy_with_repeats(valid_images, valid_dest, args.repeat_valid, dry_run=False)
    copied_invalid = copy_with_repeats(invalid_images, invalid_dest, args.repeat_invalid, dry_run=False)

    print(f"Copiadas {copied_valid} imágenes válidas   -> {valid_dest}")
    print(f"Copiadas {copied_invalid} imágenes inválidas -> {invalid_dest}")
    print()
    print("Listo. Puedes revisar el dataset con:")
    print(f"  python scripts/check_pear_gate_dataset.py --raw-dir {args.output}")


if __name__ == "__main__":
    main()
