"""
Divide el dataset pear_gate en train / val / test copiando imágenes desde raw/.

Lee imágenes recursivamente desde raw/valid_pear/** y raw/invalid/**.

Uso (dry-run):
    python scripts/split_pear_gate_dataset.py \
        --root data/pear_gate --train 0.7 --val 0.2 --test 0.1 --seed 42 --dry-run

Uso (real):
    python scripts/split_pear_gate_dataset.py \
        --root data/pear_gate --train 0.7 --val 0.2 --test 0.1 --seed 42

Añadir --clear-output para vaciar train/val/test antes de copiar.
"""
import argparse
import math
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = ["train", "val", "test"]


def collect_images_recursive(raw_class_dir: Path) -> list[tuple[str, Path]]:
    """
    Busca imágenes recursivamente dentro de raw_class_dir.
    Devuelve lista de (nombre_subcarpeta_inmediata, ruta_imagen).
    Si la imagen está directamente en raw_class_dir (sin subcarpeta), usa '' como prefijo.
    """
    result = []
    if not raw_class_dir.exists():
        return result

    for img in sorted(raw_class_dir.rglob("*")):
        if not img.is_file():
            continue
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Determinar el nombre de la subcarpeta inmediata bajo raw_class_dir
        try:
            rel = img.relative_to(raw_class_dir)
        except ValueError:
            continue
        subcat = rel.parts[0] if len(rel.parts) > 1 else ""
        result.append((subcat, img))

    return result


def make_split_indices(n: int, ratios: tuple[float, float, float], seed: int):
    """Devuelve (idx_train, idx_val, idx_test) con índices aleatorios."""
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = math.floor(ratios[0] * n)
    n_val = math.floor(ratios[1] * n)
    # test recibe el resto para no perder imágenes por redondeo
    n_test = n - n_train - n_val

    return (
        indices[:n_train],
        indices[n_train: n_train + n_val],
        indices[n_train + n_val:],
    )


def clear_split_class(root: Path, cls: str, dry_run: bool):
    for split in SPLITS:
        folder = root / split / cls
        if folder.exists():
            imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            if imgs:
                if dry_run:
                    print(f"  [DRY-RUN] vaciaría {folder} ({len(imgs)} imagen(es))")
                else:
                    for p in imgs:
                        p.unlink()
                    print(f"  Vaciada: {folder} ({len(imgs)} imagen(es) eliminadas)")


def build_dst_name(subcat: str, src: Path) -> str:
    """
    Nombre de destino sin colisiones.
    Formato: {subcat}__{stem}{ext}  o  {stem}{ext} si subcat está vacío.
    """
    ext = src.suffix.lower()
    if subcat:
        safe_subcat = subcat.replace(" ", "_")
        return f"{safe_subcat}__{src.stem}{ext}"
    return f"{src.stem}{ext}"


def copy_split(
    items: list[tuple[str, Path]],
    indices: list[int],
    dst_folder: Path,
    dry_run: bool,
) -> int:
    copied = 0
    for i in indices:
        subcat, src = items[i]
        dst_name = build_dst_name(subcat, src)
        dst = dst_folder / dst_name

        if dry_run:
            rel = dst.relative_to(dst_folder.parent.parent.parent)
            print(f"    [DRY-RUN] {src.name}  ->  {rel}")
        else:
            dst_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1
    return copied


def process_class(
    root: Path,
    raw_subdir: str,
    cls_name: str,
    ratios: tuple[float, float, float],
    seed: int,
    dry_run: bool,
) -> dict:
    raw_class_dir = root / "raw" / raw_subdir
    items = collect_images_recursive(raw_class_dir)
    n = len(items)

    stats = {"raw_encontradas": n, "train": 0, "val": 0, "test": 0}

    if n == 0:
        print(f"\n  AVISO: no hay imágenes en raw/{raw_subdir}/. Saltando clase '{cls_name}'.")
        return stats

    idx_train, idx_val, idx_test = make_split_indices(n, ratios, seed)

    n_train = len(idx_train)
    n_val = len(idx_val)
    n_test = len(idx_test)

    print(f"\n  Clase '{cls_name}': {n} imágenes raw")
    print(f"    train: {n_train}  |  val: {n_val}  |  test: {n_test}")

    for split_name, idxs in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        dst = root / split_name / cls_name
        copied = copy_split(items, idxs, dst, dry_run)
        stats[split_name] = copied

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="PearVision QC — divide pear_gate raw en train/val/test"
    )
    parser.add_argument("--root", required=True, help="Raíz del dataset pear_gate")
    parser.add_argument("--train", type=float, default=0.7, help="Proporción train (default 0.7)")
    parser.add_argument("--val", type=float, default=0.2, help="Proporción val (default 0.2)")
    parser.add_argument("--test", type=float, default=0.1, help="Proporción test (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default 42)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo imprime qué haría. No copia nada.",
    )
    parser.add_argument(
        "--clear-output", action="store_true",
        help="Vacía train/val/test antes de copiar. Por defecto NO borra nada.",
    )
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        print(f"ERROR: las proporciones deben sumar 1.0 (suma actual: {total:.4f})")
        sys.exit(1)

    root = (PROJECT_ROOT / args.root).resolve()
    if not root.exists():
        print(f"ERROR: la carpeta raíz no existe: {root}")
        sys.exit(1)

    ratios = (args.train, args.val, args.test)

    print(f"Dataset pear_gate : {root}")
    print(f"Proporciones      : train={args.train}  val={args.val}  test={args.test}")
    print(f"Semilla           : {args.seed}")
    print(f"Modo              : {'DRY-RUN' if args.dry_run else 'REAL'}")
    print(f"Limpiar salida    : {'SÍ' if args.clear_output else 'NO'}")

    if args.clear_output:
        print("\n[--clear-output] Vaciando carpetas de salida...")
        for cls in ("valid_pear", "invalid"):
            clear_split_class(root, cls, args.dry_run)

    print("\n--- Procesando clases ---")

    stats_valid = process_class(
        root, "valid_pear", "valid_pear", ratios, args.seed, args.dry_run,
    )
    stats_invalid = process_class(
        root, "invalid", "invalid", ratios, args.seed, args.dry_run,
    )

    modo = "DRY-RUN (nada fue copiado)" if args.dry_run else "CONVERSIÓN REAL"
    print(f"\n{'=' * 55}")
    print(f"RESUMEN — {modo}")
    print(f"{'=' * 55}")
    print(f"  Imágenes raw valid_pear encontradas : {stats_valid['raw_encontradas']}")
    print(f"  Imágenes raw invalid encontradas    : {stats_invalid['raw_encontradas']}")
    print()
    for split in SPLITS:
        total_split = stats_valid[split] + stats_invalid[split]
        print(
            f"  {split:5s}: {total_split:4d} total "
            f"(valid_pear={stats_valid[split]}, invalid={stats_invalid[split]})"
        )
    print(f"{'=' * 55}")

    if not args.dry_run:
        print("\nPróximo paso recomendado:")
        print("  python scripts/check_pear_gate_dataset.py --root data/pear_gate")


if __name__ == "__main__":
    main()
