"""
Comprueba el estado del dataset pear_gate sin modificar nada.

Uso:
    python scripts/check_pear_gate_dataset.py --root data/pear_gate
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_IMAGES_WARNING = 20

SPLIT_CLASSES = ["valid_pear", "invalid"]
SPLITS = ["train", "val", "test"]


def count_images(folder: Path) -> int:
    if not folder.exists():
        return -1
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def check_raw(root: Path):
    print("\n=== RAW ===")
    raw = root / "raw"

    if not raw.exists():
        print("  ERROR: la carpeta raw/ no existe.")
        return

    total_valid = 0
    total_invalid = 0

    for label in ("valid_pear", "invalid"):
        label_dir = raw / label
        print(f"\n  [raw/{label}]")
        if not label_dir.exists():
            print(f"    AVISO: carpeta no encontrada — raw/{label}")
            continue

        subcats = sorted(p for p in label_dir.iterdir() if p.is_dir())
        if not subcats:
            # imágenes sueltas directamente en la carpeta de label
            n = count_images(label_dir)
            sufijo = _sufijo(n)
            print(f"    (imágenes sueltas): {n:4d} imagen(es){sufijo}")
            if label == "valid_pear":
                total_valid += max(n, 0)
            else:
                total_invalid += max(n, 0)
        else:
            for subcat in subcats:
                n = count_images(subcat)
                sufijo = _sufijo(n)
                print(f"    {subcat.name:35s}: {n:4d} imagen(es){sufijo}")
                if label == "valid_pear":
                    total_valid += max(n, 0)
                else:
                    total_invalid += max(n, 0)

        total = total_valid if label == "valid_pear" else total_invalid
        print(f"    {'TOTAL ' + label:35s}: {total:4d} imagen(es)")

    if total_valid == 0 and total_invalid == 0:
        print("\n  AVISO: raw/ está completamente vacío. Añade imágenes antes de hacer split.")


def _sufijo(n: int) -> str:
    if n == 0:
        return "  <- VACÍA"
    if 0 < n < MIN_IMAGES_WARNING:
        return f"  <- AVISO: menos de {MIN_IMAGES_WARNING} imágenes"
    return ""


def check_splits(root: Path):
    print("\n=== SPLITS (train / val / test) ===")

    all_empty = True
    for split in SPLITS:
        print(f"\n  [{split}]")
        for cls in SPLIT_CLASSES:
            folder = root / split / cls
            n = count_images(folder)
            if n == -1:
                print(f"    AVISO: carpeta no encontrada — {split}/{cls}")
            else:
                if n > 0:
                    all_empty = False
                sufijo = "  <- VACÍA" if n == 0 else ""
                print(f"    {cls:12s}: {n:4d} imagen(es){sufijo}")

    if all_empty:
        print("\n  AVISO: train/val/test están vacíos. Ejecuta split_pear_gate_dataset.py para poblarlos.")


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — comprobación del dataset pear_gate"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Ruta raíz del dataset pear_gate (ej. data/pear_gate)",
    )
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()

    if not root.exists():
        print(f"ERROR: la carpeta raíz no existe: {root}")
        sys.exit(1)

    print(f"Dataset pear_gate : {root}")

    check_raw(root)
    check_splits(root)

    print("\n--- FIN DEL CHEQUEO ---")


if __name__ == "__main__":
    main()
