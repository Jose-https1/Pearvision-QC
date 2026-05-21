"""
Validación del dataset Custom Pear Defects (YOLO detection).

Comprueba:
  - Imágenes sin label correspondiente
  - Labels sin imagen correspondiente
  - Líneas mal formadas en los .txt
  - Clases fuera de rango (0-4)
  - Coordenadas fuera del rango [0.0, 1.0]

Uso:
    python scripts/prepare_custom_defect_dataset.py
    python scripts/prepare_custom_defect_dataset.py --split train
    python scripts/prepare_custom_defect_dataset.py --split val
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "data" / "pear_defects_custom"

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_CLASSES  = 5
CLASS_NAMES  = {0: "bruise", 1: "stab", 2: "twig", 3: "tcm", 4: "rot"}
SPLITS       = ["train", "val"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _sep(title=""):
    print(f"\n{'-' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'-' * 60}")


def _image_stem_set(split: str) -> set[str]:
    img_dir = DATASET_DIR / "images" / split
    return {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def _label_stem_set(split: str) -> set[str]:
    lbl_dir = DATASET_DIR / "labels" / split
    return {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}


def _validate_label_file(path: Path) -> list[str]:
    """Devuelve lista de errores encontrados en un archivo de etiquetas."""
    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [f"No se pudo leer: {exc}"]

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"línea {lineno}: esperados 5 campos, hay {len(parts)} → '{line}'")
            continue

        try:
            cls_id = int(parts[0])
        except ValueError:
            errors.append(f"línea {lineno}: class_id no es entero → '{parts[0]}'")
            continue

        if cls_id < 0 or cls_id >= NUM_CLASSES:
            errors.append(
                f"línea {lineno}: class_id={cls_id} fuera de rango "
                f"[0-{NUM_CLASSES - 1}] → clase '{CLASS_NAMES.get(cls_id, '?')}'"
            )

        try:
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"línea {lineno}: coordenadas no numéricas → '{parts[1:]}'")
            continue

        for name, val in zip(["cx", "cy", "w", "h"], coords):
            if not (0.0 <= val <= 1.0):
                errors.append(
                    f"línea {lineno}: {name}={val:.4f} fuera de [0.0, 1.0]"
                )

    return errors


def _validate_split(split: str) -> dict:
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split

    result = {
        "split":          split,
        "img_count":      0,
        "lbl_count":      0,
        "missing_labels": [],
        "orphan_labels":  [],
        "label_errors":   {},   # stem → [error strings]
    }

    if not img_dir.exists():
        print(f"  AVISO: carpeta no existe: {img_dir.relative_to(PROJECT_ROOT)}")
        return result
    if not lbl_dir.exists():
        print(f"  AVISO: carpeta no existe: {lbl_dir.relative_to(PROJECT_ROOT)}")
        return result

    imgs  = _image_stem_set(split)
    lbls  = _label_stem_set(split)

    result["img_count"] = len(imgs)
    result["lbl_count"] = len(lbls)
    result["missing_labels"] = sorted(imgs - lbls)
    result["orphan_labels"]  = sorted(lbls - imgs)

    for stem in sorted(lbls):
        lbl_path = lbl_dir / f"{stem}.txt"
        errs = _validate_label_file(lbl_path)
        if errs:
            result["label_errors"][stem] = errs

    return result


def _print_result(r: dict):
    split = r["split"]
    _sep(f"Split: {split}")
    print(f"  Imágenes encontradas : {r['img_count']}")
    print(f"  Labels encontrados   : {r['lbl_count']}")

    if r["missing_labels"]:
        print(f"\n  [FALTA LABEL] {len(r['missing_labels'])} imagen(es) sin .txt:")
        for stem in r["missing_labels"][:20]:
            print(f"    - {stem}")
        if len(r["missing_labels"]) > 20:
            print(f"    ... y {len(r['missing_labels']) - 20} más")
    else:
        print("  Imágenes sin label   : 0  OK")

    if r["orphan_labels"]:
        print(f"\n  [LABEL HUÉRFANO] {len(r['orphan_labels'])} label(s) sin imagen:")
        for stem in r["orphan_labels"][:20]:
            print(f"    - {stem}")
        if len(r["orphan_labels"]) > 20:
            print(f"    ... y {len(r['orphan_labels']) - 20} más")
    else:
        print("  Labels huérfanos     : 0  OK")

    if r["label_errors"]:
        total_err = sum(len(v) for v in r["label_errors"].values())
        print(f"\n  [ERRORES EN LABELS] {total_err} error(es) en {len(r['label_errors'])} archivo(s):")
        for stem, errs in list(r["label_errors"].items())[:10]:
            print(f"    {stem}.txt:")
            for e in errs:
                print(f"      · {e}")
        if len(r["label_errors"]) > 10:
            print(f"    ... y {len(r['label_errors']) - 10} archivos más con errores")
    else:
        print("  Errores en labels    : 0  OK")


def _summary(results: list[dict]) -> bool:
    """Imprime resumen global. Devuelve True si todo OK."""
    _sep("Resumen global")
    total_imgs = sum(r["img_count"] for r in results)
    total_lbls = sum(r["lbl_count"] for r in results)
    total_miss = sum(len(r["missing_labels"]) for r in results)
    total_orph = sum(len(r["orphan_labels"]) for r in results)
    total_errf = sum(len(r["label_errors"]) for r in results)

    print(f"  Total imágenes  : {total_imgs}")
    print(f"  Total labels    : {total_lbls}")
    print(f"  Sin label       : {total_miss}")
    print(f"  Huérfanos       : {total_orph}")
    print(f"  Archivos c/error: {total_errf}")

    ok = (total_miss == 0 and total_orph == 0 and total_errf == 0)
    if ok:
        print("\n  Dataset OK — listo para entrenar.")
    else:
        print("\n  Dataset con problemas — corrige los errores antes de entrenar.")
    return ok


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Valida el dataset Custom Pear Defects.")
    parser.add_argument("--split", choices=SPLITS,
                        help="Validar solo un split (train o val). Por defecto: ambos.")
    args = parser.parse_args()

    splits_to_check = [args.split] if args.split else SPLITS

    print("\n=== Validación Custom Pear Defects Dataset ===")
    print(f"  Directorio: {DATASET_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Clases ({NUM_CLASSES}): {', '.join(CLASS_NAMES.values())}")

    results = []
    for split in splits_to_check:
        r = _validate_split(split)
        _print_result(r)
        results.append(r)

    ok = _summary(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
