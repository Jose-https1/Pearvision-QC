"""
Prepara el dataset custom de defectos de pera a partir del dataset PSD.

- Elimina la clase 0 (pear) de los labels.
- Remapea clases 1-5 a 0-4 (bruise, stab, twig, tcm, rot).
- Descarta imagenes que quedan sin defectos tras eliminar pear.
- Copia imagenes validas y escribe labels remapeados.
- Genera reporte en reports/psd_defects_custom_report.md

Uso:
    python scripts/prepare_psd_defects_custom.py
    python scripts/prepare_psd_defects_custom.py --dry-run
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PSD_BASE = PROJECT_ROOT / "data_external" / "PSD" / "datasets" / "Date_demo"
DST_BASE = PROJECT_ROOT / "data" / "pear_defects_custom"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# PSD class 0 = pear (removed). Classes 1-5 remapped to 0-4.
REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CLASS_NAMES = {0: "bruise", 1: "stab", 2: "twig", 3: "tcm", 4: "rot"}
SPLITS = ["train", "val"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _remap_label(src_txt: Path) -> tuple[list[str], list[str]]:
    """
    Lee un label YOLO original y devuelve (lineas_remapeadas, errores).
    Omite lineas de clase 0 (pear). Remapea 1-5 a 0-4.
    """
    remapped = []
    errors = []
    try:
        lines = src_txt.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [], [f"No se pudo leer: {exc}"]

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"linea {lineno}: {len(parts)} campos (esperados 5)")
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            errors.append(f"linea {lineno}: class_id no entero '{parts[0]}'")
            continue

        if cls_id == 0:
            continue  # eliminar clase pear

        if cls_id not in REMAP:
            errors.append(f"linea {lineno}: class_id={cls_id} fuera de rango 0-5")
            continue

        new_cls = REMAP[cls_id]
        remapped.append(f"{new_cls} {' '.join(parts[1:])}")

    return remapped, errors


def _process_split(split: str, dry_run: bool) -> dict:
    src_img_dir = PSD_BASE / "images" / split
    src_lbl_dir = PSD_BASE / "labels" / split
    dst_img_dir = DST_BASE / "images" / split
    dst_lbl_dir = DST_BASE / "labels" / split

    result = {
        "split":    split,
        "copied":   0,
        "skipped":  0,
        "errors":   [],
        "class_counts": Counter(),
        "skipped_names": [],
    }

    if not src_lbl_dir.exists():
        result["errors"].append(f"Carpeta no encontrada: {src_lbl_dir}")
        return result

    txt_files = sorted(src_lbl_dir.glob("*.txt"))

    if not dry_run:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in txt_files:
        stem = txt_path.stem

        img_path = _find_image(src_img_dir, stem)
        if img_path is None:
            result["errors"].append(f"Imagen no encontrada para: {stem}")
            continue

        remapped, errs = _remap_label(txt_path)
        if errs:
            result["errors"].extend([f"[{stem}] {e}" for e in errs])

        if not remapped:
            result["skipped"] += 1
            result["skipped_names"].append(stem)
            continue

        for line in remapped:
            cls_id = int(line.split()[0])
            result["class_counts"][cls_id] += 1

        if not dry_run:
            dst_lbl = dst_lbl_dir / f"{stem}.txt"
            dst_lbl.write_text("\n".join(remapped) + "\n", encoding="utf-8")

            dst_img = dst_img_dir / img_path.name
            shutil.copy2(str(img_path), str(dst_img))

        result["copied"] += 1

    return result


def _save_report(results: list[dict]) -> Path:
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "psd_defects_custom_report.md"

    total_copied  = sum(r["copied"]  for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors  = sum(len(r["errors"]) for r in results)
    global_counts: Counter = Counter()
    for r in results:
        global_counts.update(r["class_counts"])

    lines = [
        "# Reporte: PSD -> Custom Pear Defects",
        "",
        f"**Fecha:** 2026-05-17",
        "",
        "## Resumen por split",
        "",
        "| Split | Copiadas | Descartadas (sin defecto) | Errores |",
        "|-------|---------|--------------------------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r['split']} | {r['copied']} | {r['skipped']} | {len(r['errors'])} |"
        )
    lines += [
        f"| **Total** | **{total_copied}** | **{total_skipped}** | **{total_errors}** |",
        "",
        "## Conteo por clase (total anotaciones)",
        "",
        "| ID | Nombre | Anotaciones |",
        "|----|--------|------------|",
    ]
    for cls_id in sorted(CLASS_NAMES):
        lines.append(
            f"| {cls_id} | {CLASS_NAMES[cls_id]} | {global_counts.get(cls_id, 0)} |"
        )

    for r in results:
        if r["skipped_names"]:
            lines += [
                "",
                f"## Imagenes descartadas ({r['split']})",
                "",
            ]
            for name in r["skipped_names"]:
                lines.append(f"- {name}")

    if total_errors > 0:
        lines += ["", "## Errores encontrados", ""]
        for r in results:
            for e in r["errors"]:
                lines.append(f"- [{r['split']}] {e}")

    lines += [
        "",
        "## Dataset listo",
        "",
        "El dataset esta listo para entrenamiento si no hay errores criticos.",
        "",
        "Comando de validacion:",
        "",
        "```powershell",
        "uv run python scripts/prepare_custom_defect_dataset.py",
        "```",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _print_summary(results: list[dict]):
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    global_counts: Counter = Counter()
    for r in results:
        global_counts.update(r["class_counts"])
        print(f"\n  [{r['split']}]")
        print(f"    Imagenes copiadas   : {r['copied']}")
        print(f"    Descartadas         : {r['skipped']}")
        if r["skipped_names"]:
            for n in r["skipped_names"]:
                print(f"      - {n} (solo pear, sin defectos)")
        print(f"    Errores             : {len(r['errors'])}")
        for e in r["errors"]:
            print(f"      ! {e}")

    print("\n  Conteo por clase (anotaciones totales):")
    for cls_id in sorted(CLASS_NAMES):
        print(f"    {cls_id}: {CLASS_NAMES[cls_id]:<8} {global_counts.get(cls_id, 0)}")

    total_errors = sum(len(r["errors"]) for r in results)
    if total_errors == 0:
        print("\n  Dataset OK - listo para validar y entrenar.")
    else:
        print(f"\n  {total_errors} errores encontrados - revisar antes de entrenar.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepara dataset custom de defectos de pera desde PSD."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra que haria, sin copiar archivos.")
    args = parser.parse_args()

    if args.dry_run:
        print("\n[DRY RUN] No se copiaran archivos.")

    print("\n=== prepare_psd_defects_custom ===")
    print(f"  Fuente : {PSD_BASE.relative_to(PROJECT_ROOT)}")
    print(f"  Destino: {DST_BASE.relative_to(PROJECT_ROOT)}")

    for folder in [
        PSD_BASE / "images" / "train", PSD_BASE / "images" / "val",
        PSD_BASE / "labels" / "train", PSD_BASE / "labels" / "val",
    ]:
        if not folder.exists():
            print(f"ERROR: no existe: {folder}", file=sys.stderr)
            sys.exit(1)

    results = []
    for split in SPLITS:
        print(f"\nProcesando split '{split}' ...")
        r = _process_split(split, dry_run=args.dry_run)
        results.append(r)
        print(f"  Copiadas: {r['copied']}  Descartadas: {r['skipped']}  Errores: {len(r['errors'])}")

    _print_summary(results)

    if not args.dry_run:
        report_path = _save_report(results)
        print(f"\n  Reporte: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
