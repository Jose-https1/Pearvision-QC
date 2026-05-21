"""
consolidate_fruits360_quality_dataset_v1.py

Consolida las 300 etiquetas humanas Fruits-360 en un dataset binario listo
para entrenamiento posterior de un clasificador good/bad.

Fuente:  data/fruits360_human_review/human_labels_template.csv
Destino: data/quality_fruits360_human_v1/
         train/good/ train/bad/
         val/good/   val/bad/
         test/good/  test/bad/
         metadata/quality_fruits360_human_v1_master.csv
         metadata/split_summary.csv
         metadata/excluded_review.csv

NO se entrena ningun modelo.
"""

import csv
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_CSV      = PROJECT_ROOT / "data" / "fruits360_human_review" / "human_labels_template.csv"
DEST_ROOT    = PROJECT_ROOT / "data" / "quality_fruits360_human_v1"
META_DIR     = DEST_ROOT / "metadata"
REPORT_MD    = PROJECT_ROOT / "reports" / "consolidate_fruits360_quality_v1_report.md"

# ── Config splits ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

LABEL_MAP = {"GOOD": "good", "BAD": "bad"}   # REVIEW e INVALID excluidos


# ── Helpers ───────────────────────────────────────────────────────────────────

def stratified_split(items: list, train_r: float, val_r: float, seed: int):
    """Divide una lista en train/val/test de forma estratificada por clase."""
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== consolidate_fruits360_quality_dataset_v1 ===")

    if not SRC_CSV.exists():
        print(f"ERROR: {SRC_CSV} no existe.")
        sys.exit(1)

    # 1. Cargar CSV
    with SRC_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  Filas cargadas: {len(rows)}")

    # 2. Separar good/bad / review / invalid
    good_rows    = [r for r in rows if r["human_label"] == "GOOD"]
    bad_rows     = [r for r in rows if r["human_label"] == "BAD"]
    review_rows  = [r for r in rows if r["human_label"] == "REVIEW"]
    invalid_rows = [r for r in rows if r["human_label"] == "INVALID"]
    unlabeled    = [r for r in rows if not r["human_label"].strip()]

    print(f"  GOOD   : {len(good_rows)}")
    print(f"  BAD    : {len(bad_rows)}")
    print(f"  REVIEW : {len(review_rows)}  (excluidos)")
    print(f"  INVALID: {len(invalid_rows)}  (excluidos)")
    print(f"  Sin label: {len(unlabeled)}")

    if unlabeled:
        print("  ADVERTENCIA: hay filas sin etiqueta — se ignoraran.")

    usable = len(good_rows) + len(bad_rows)
    print(f"  Usables para dataset: {usable}")

    # 3. Split estratificado por clase
    good_train, good_val, good_test = stratified_split(good_rows, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    bad_train,  bad_val,  bad_test  = stratified_split(bad_rows,  TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    splits = {
        "train": good_train + bad_train,
        "val":   good_val   + bad_val,
        "test":  good_test  + bad_test,
    }

    print()
    print("  Splits calculados:")
    for split, srows in splits.items():
        cnt = Counter(r["human_label"] for r in srows)
        print(f"    {split:<6}: {len(srows):3d} total  "
              f"(GOOD={cnt.get('GOOD',0)}, BAD={cnt.get('BAD',0)})")

    # 4. Crear estructura de directorios
    for split in ["train", "val", "test"]:
        for cls in ["good", "bad"]:
            (DEST_ROOT / split / cls).mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    print()
    print(f"  Estructura creada en: {DEST_ROOT}")

    # 5. Copiar imágenes y construir master CSV
    master_rows  = []
    missing_imgs = []
    copied = 0

    for split, srows in splits.items():
        for r in srows:
            cls      = LABEL_MAP[r["human_label"]]
            dest_dir = DEST_ROOT / split / cls
            ok = copy_image(r["source_path"], dest_dir, r["filename"])
            if ok:
                copied += 1
            else:
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

    print(f"  Imagenes copiadas : {copied}")
    if missing_imgs:
        print(f"  Imagenes faltantes: {len(missing_imgs)} → {missing_imgs[:5]}{'...' if len(missing_imgs)>5 else ''}")

    # 6. Guardar metadata/quality_fruits360_human_v1_master.csv
    master_path = META_DIR / "quality_fruits360_human_v1_master.csv"
    master_fields = ["review_id","filename","human_label","class","split",
                     "original_split","original_class","source_path","image_copied"]
    with master_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=master_fields)
        w.writeheader()
        w.writerows(sorted(master_rows, key=lambda r: r["review_id"]))
    print(f"  Master CSV: {master_path.name}")

    # 7. Guardar metadata/split_summary.csv
    summary_path = META_DIR / "split_summary.csv"
    summary_rows = []
    for split, srows in splits.items():
        cnt = Counter(r["human_label"] for r in srows)
        summary_rows.append({
            "split": split,
            "good":  cnt.get("GOOD", 0),
            "bad":   cnt.get("BAD",  0),
            "total": len(srows),
            "good_pct": f"{cnt.get('GOOD',0)/len(srows)*100:.1f}%",
            "bad_pct":  f"{cnt.get('BAD', 0)/len(srows)*100:.1f}%",
        })
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","good","bad","total","good_pct","bad_pct"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Split summary: {summary_path.name}")

    # 8. Guardar metadata/excluded_review.csv
    excluded_path = META_DIR / "excluded_review.csv"
    excl_fields = ["review_id","filename","human_label","original_split","original_class","source_path","reason"]
    excl_rows = [
        {**{k: r[k] for k in ["review_id","filename","human_label","original_split","original_class","source_path"]},
         "reason": "REVIEW — excluded from training"}
        for r in review_rows
    ]
    with excluded_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=excl_fields)
        w.writeheader()
        w.writerows(excl_rows)
    print(f"  Excluded review: {excluded_path.name}  ({len(excl_rows)} filas)")

    # 9. Reporte .md
    write_report(splits, good_rows, bad_rows, review_rows,
                 copied, missing_imgs, master_rows)

    # 10. Resumen consola
    print()
    print("=== RESUMEN FINAL ===")
    print(f"  Dataset: {DEST_ROOT.name}")
    print(f"  Clases  : good={len(good_rows)}, bad={len(bad_rows)}")
    for split, srows in splits.items():
        cnt = Counter(r["human_label"] for r in srows)
        print(f"  {split:<6} : {len(srows):3d}  (good={cnt.get('GOOD',0)}, bad={cnt.get('BAD',0)})")
    print(f"  Imagenes copiadas: {copied} / {usable}")
    print(f"  REVIEW excluidos : {len(review_rows)}")
    print(f"  No se entrenó ningun modelo.")
    print()
    print("  ADVERTENCIA DE DESBALANCE:")
    print(f"    good={len(good_rows)} vs bad={len(bad_rows)}  "
          f"(ratio 1:{len(bad_rows)//max(len(good_rows),1)})")
    print("    El entrenamiento posterior necesitara:")
    print("      - class_weight='balanced' o pesos manuales")
    print("      - WeightedRandomSampler (PyTorch)")
    print("      - y/o augmentacion fuerte de GOOD")


def write_report(splits, good_rows, bad_rows, review_rows,
                 copied, missing_imgs, master_rows):
    total_usable = len(good_rows) + len(bad_rows)
    all_issues = []
    if missing_imgs:
        all_issues.append(f"Imagenes no encontradas: {len(missing_imgs)} → {missing_imgs}")

    status = "PASS" if not missing_imgs else "PASS (con advertencias)"

    lines = [
        "# Consolidacion Fruits-360 Quality Dataset v1",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Status: **{status}**",
        "",
        "## Fuente",
        "",
        "- CSV: `data/fruits360_human_review/human_labels_template.csv`",
        "- Imagenes: `source_path` en el CSV (data_external/fruits360_original_size/)",
        "",
        "## Destino",
        "",
        "- Dataset: `data/quality_fruits360_human_v1/`",
        "",
        "## Distribucion de etiquetas humanas",
        "",
        "| Etiqueta | Count | Uso |",
        "|----------|-------|-----|",
        f"| GOOD     | {len(good_rows)}   | clase `good` → train/val/test |",
        f"| BAD      | {len(bad_rows)}  | clase `bad`  → train/val/test |",
        f"| REVIEW   | {len(review_rows)}   | excluido |",
        "| INVALID  | 0     | excluido |",
        f"| **TOTAL usable** | **{total_usable}** | |",
        "",
        "## Splits",
        "",
        "Ratio: 70% train / 15% val / 15% test — seed=42 — estratificado por clase",
        "",
        "| Split | good | bad | Total | good% | bad% |",
        "|-------|------|-----|-------|-------|------|",
    ]
    for split, srows in splits.items():
        from collections import Counter
        cnt = Counter(r["human_label"] for r in srows)
        g, b, t = cnt.get("GOOD",0), cnt.get("BAD",0), len(srows)
        lines.append(f"| {split} | {g} | {b} | {t} | {g/t*100:.1f}% | {b/t*100:.1f}% |")

    lines += [
        "",
        "## Imagenes copiadas",
        "",
        f"- Copiadas: {copied} / {total_usable}",
    ]
    if missing_imgs:
        lines += [f"- Faltantes: {len(missing_imgs)} → {missing_imgs}"]

    lines += [
        "",
        "## Archivos generados",
        "",
        "- `data/quality_fruits360_human_v1/train/good/` — imagenes buenas de entrenamiento",
        "- `data/quality_fruits360_human_v1/train/bad/`  — imagenes malas de entrenamiento",
        "- `data/quality_fruits360_human_v1/val/good/`",
        "- `data/quality_fruits360_human_v1/val/bad/`",
        "- `data/quality_fruits360_human_v1/test/good/`",
        "- `data/quality_fruits360_human_v1/test/bad/`",
        "- `data/quality_fruits360_human_v1/metadata/quality_fruits360_human_v1_master.csv`",
        "- `data/quality_fruits360_human_v1/metadata/split_summary.csv`",
        "- `data/quality_fruits360_human_v1/metadata/excluded_review.csv`",
        "",
        "## Advertencia de desbalance",
        "",
        f"Ratio good:bad = 1:{len(bad_rows)//max(len(good_rows),1)} aprox.",
        "",
        "El entrenamiento posterior **necesitara** al menos una de:",
        "- `class_weight='balanced'` (scikit-learn) o pesos manuales en CrossEntropyLoss",
        "- `WeightedRandomSampler` (PyTorch) para equilibrar batches",
        "- Augmentacion fuerte de GOOD (flips, rotaciones, color jitter, etc.)",
        "",
        "## Confirmaciones",
        "",
        "- **NO se entrenó ningun modelo.**",
        "- Solo se copiaron imagenes y se generaron CSVs de metadata.",
        "- Las 31 imagenes REVIEW estan documentadas en `excluded_review.csv`.",
        "- Trazabilidad completa: `review_id` → `source_path` → `split/class/filename`.",
    ]
    if all_issues:
        lines += ["", "## Advertencias", ""] + [f"- {i}" for i in all_issues]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD}")


if __name__ == "__main__":
    main()
