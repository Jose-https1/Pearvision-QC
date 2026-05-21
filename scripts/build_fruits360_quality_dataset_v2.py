"""
build_fruits360_quality_dataset_v2.py

Reconstruye el dataset binario good/bad a partir del CSV corregido.
Crea:  data/quality_fruits360_human_v2/
       train/good/ train/bad/
       val/good/   val/bad/
       test/good/  test/bad/
       metadata/quality_fruits360_human_v2_master.csv
       metadata/split_summary.csv
       metadata/excluded_review.csv

No entrena ningun modelo.
"""

import csv
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_CSV      = PROJECT_ROOT / "data" / "fruits360_human_review" / "human_labels_template.csv"
DEST_ROOT    = PROJECT_ROOT / "data" / "quality_fruits360_human_v2"
META_DIR     = DEST_ROOT / "metadata"
REPORT_MD    = PROJECT_ROOT / "reports" / "build_fruits360_quality_v2_report.md"

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42
LABEL_MAP    = {"GOOD": "good", "BAD": "bad"}


def stratified_split(items: list, train_r: float, val_r: float, seed: int):
    rng = random.Random(seed)
    rng.shuffle(items)
    n       = len(items)
    n_train = round(n * train_r)
    n_val   = round(n * val_r)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def copy_image(src_path: str, dest_dir: Path, filename: str) -> bool:
    src = Path(src_path)
    if not src.exists():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / filename)
    return True


def main():
    print("=== build_fruits360_quality_dataset_v2 ===")
    if not SRC_CSV.exists():
        print(f"ERROR: {SRC_CSV} no existe.")
        sys.exit(1)

    with SRC_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  Filas totales en CSV: {len(rows)}")

    good_rows    = [r for r in rows if r["human_label"] == "GOOD"]
    bad_rows     = [r for r in rows if r["human_label"] == "BAD"]
    review_rows  = [r for r in rows if r["human_label"] == "REVIEW"]
    invalid_rows = [r for r in rows if r["human_label"] == "INVALID"]

    print(f"  GOOD   : {len(good_rows)}")
    print(f"  BAD    : {len(bad_rows)}")
    print(f"  REVIEW : {len(review_rows)}  (excluidos)")
    print(f"  INVALID: {len(invalid_rows)}  (excluidos)")
    print(f"  Usables: {len(good_rows) + len(bad_rows)}")

    # Splits estratificados
    good_train, good_val, good_test = stratified_split(good_rows, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    bad_train,  bad_val,  bad_test  = stratified_split(bad_rows,  TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    splits = {
        "train": good_train + bad_train,
        "val":   good_val   + bad_val,
        "test":  good_test  + bad_test,
    }

    print("\n  Splits calculados:")
    for split, srows in splits.items():
        cnt = Counter(r["human_label"] for r in srows)
        print(f"    {split:<6}: {len(srows):3d}  (GOOD={cnt.get('GOOD',0)}, BAD={cnt.get('BAD',0)})")

    # Crear estructura de directorios
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    for split in ["train", "val", "test"]:
        for cls in ["good", "bad"]:
            (DEST_ROOT / split / cls).mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Estructura creada en: {DEST_ROOT.name}")

    # Copiar imagenes
    master_rows  = []
    missing_imgs = []
    copied       = 0

    for split, srows in splits.items():
        for r in srows:
            cls      = LABEL_MAP[r["human_label"]]
            dest_dir = DEST_ROOT / split / cls
            ok       = copy_image(r["source_path"], dest_dir, r["filename"])
            copied  += int(ok)
            if not ok:
                missing_imgs.append(r["review_id"])
            master_rows.append({
                "review_id":      r["review_id"],
                "filename":       r["filename"],
                "human_label":    r["human_label"],
                "class":          cls,
                "split":          split,
                "original_split": r["original_split"],
                "original_class": r["original_class"],
                "source_path":    r["source_path"],
                "image_copied":   "yes" if ok else "MISSING",
            })

    print(f"  Imagenes copiadas: {copied}")
    if missing_imgs:
        print(f"  Faltantes: {missing_imgs}")

    # CSVs de metadata
    master_path = META_DIR / "quality_fruits360_human_v2_master.csv"
    fields      = ["review_id","filename","human_label","class","split",
                   "original_split","original_class","source_path","image_copied"]
    with master_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(master_rows, key=lambda r: r["review_id"]))
    print(f"  Master CSV: {master_path.name}")

    summary_path = META_DIR / "split_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","good","bad","total","good_pct","bad_pct"])
        w.writeheader()
        for split, srows in splits.items():
            cnt = Counter(r["human_label"] for r in srows)
            g, b, t = cnt.get("GOOD",0), cnt.get("BAD",0), len(srows)
            w.writerow({"split": split, "good": g, "bad": b, "total": t,
                        "good_pct": f"{g/t*100:.1f}%", "bad_pct": f"{b/t*100:.1f}%"})
    print(f"  Split summary: {summary_path.name}")

    excl_path   = META_DIR / "excluded_review.csv"
    excl_fields = ["review_id","filename","human_label","original_split","original_class","source_path","reason"]
    with excl_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=excl_fields)
        w.writeheader()
        for r in review_rows:
            w.writerow({k: r[k] for k in ["review_id","filename","human_label",
                                           "original_split","original_class","source_path"]}
                       | {"reason": "REVIEW — excluded from training"})
    print(f"  Excluded review: {excl_path.name}  ({len(review_rows)} filas)")

    # Reporte
    total_usable = len(good_rows) + len(bad_rows)
    lines = [
        "# Build Fruits-360 Quality Dataset V2",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Diferencias V1 vs V2",
        "",
        "| ID | Cambio |",
        "|----|--------|",
        "| F360_0198 | BAD -> REVIEW (excluida del entrenamiento) |",
        "| F360_0052 | GOOD -> REVIEW (excluida del entrenamiento) |",
        "| F360_0060 | GOOD -> GOOD (sin cambio, etiqueta confirmada) |",
        "",
        "## Distribucion V2",
        "",
        "| Etiqueta | Count | Uso |",
        "|----------|-------|-----|",
        f"| GOOD     | {len(good_rows)} | clase good -> train/val/test |",
        f"| BAD      | {len(bad_rows)} | clase bad  -> train/val/test |",
        f"| REVIEW   | {len(review_rows)} | excluido |",
        f"| INVALID  | {len(invalid_rows)} | excluido |",
        f"| **TOTAL usable** | **{total_usable}** | |",
        "",
        "## Splits (seed=42, 70/15/15, estratificado)",
        "",
        "| Split | good | bad | Total |",
        "|-------|------|-----|-------|",
    ]
    for split, srows in splits.items():
        cnt = Counter(r["human_label"] for r in srows)
        lines.append(f"| {split} | {cnt.get('GOOD',0)} | {cnt.get('BAD',0)} | {len(srows)} |")

    lines += [
        "",
        "## Confirmaciones",
        "- NO se entrenó ningun modelo.",
        "- analyze_quality.py NO fue modificado.",
        "- quality_rules.yaml NO fue modificado.",
        f"- Imagenes copiadas: {copied}/{total_usable}.",
    ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD.name}")

    print("\n=== RESUMEN V2 DATASET ===")
    print(f"  GOOD={len(good_rows)}  BAD={len(bad_rows)}  REVIEW={len(review_rows)}")
    for split, srows in splits.items():
        cnt = Counter(r["human_label"] for r in srows)
        print(f"  {split:<6}: {len(srows):3d}  (good={cnt.get('GOOD',0)}, bad={cnt.get('BAD',0)})")


if __name__ == "__main__":
    main()
