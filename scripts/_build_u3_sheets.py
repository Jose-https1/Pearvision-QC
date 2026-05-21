"""
_build_u3_sheets.py
Reads the existing resultados_integrated_u3.csv and creates contact sheets + summary.
Run after test_integrated_u3_pipeline.py has produced the merged CSV.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUT_DIR      = PROJECT_ROOT / "outputs" / "u3_integrated_pipeline_eval"
MERGED_CSV   = OUT_DIR / "resultados_integrated_u3.csv"
SHEET_ALL    = OUT_DIR / "contact_sheet_integrated_u3_all.jpg"
SHEET_REVBAD = OUT_DIR / "contact_sheet_integrated_u3_review_bad.jpg"
SUMMARY_TXT  = OUT_DIR / "summary.txt"

SOURCE_DIRS = [
    PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3",
    PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_good_batch_v2",
    PROJECT_ROOT / "data/supermarket_good_hard_examples_v1/images",
    PROJECT_ROOT / "data/supermarket_good_hard_examples_v2/images",
]

DECISION_COLOR = {"PASA": (0, 200, 0), "REVISAR": (0, 165, 255), "RECHAZA": (0, 0, 220)}
THUMB_W, THUMB_H = 200, 200
COLS = 7


def load_thumb(img_path: Path, w=THUMB_W, h=THUMB_H):
    from PIL import Image
    try:
        pil = Image.open(str(img_path)).convert("RGB")
        pil = pil.resize((w, h))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return np.full((h, w, 3), 80, dtype=np.uint8)


def add_label(thumb, text, color):
    cv2.rectangle(thumb, (0, thumb.shape[0] - 26), (thumb.shape[1], thumb.shape[0]), (20, 20, 20), -1)
    cv2.putText(thumb, text, (3, thumb.shape[0] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return thumb


def build_contact_sheet(entries, out_path: Path, title: str, cols=COLS):
    if not entries:
        blank = np.full((THUMB_H + 40, THUMB_W * cols, 3), 50, dtype=np.uint8)
        cv2.putText(blank, "Sin imagenes", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        from PIL import Image
        Image.fromarray(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=90)
        print(f"  [empty] {out_path.name}")
        return

    thumbs = []
    for img_path, decision, label, p_good in entries:
        t = load_thumb(img_path)
        color = DECISION_COLOR.get(decision, (200, 200, 200))
        lbl = f"{decision} g={float(p_good):.2f}"
        add_label(t, lbl, color)
        cv2.rectangle(t, (0, 0), (t.shape[1]-1, t.shape[0]-1), color, 3)
        thumbs.append(t)

    rows_count = (len(thumbs) + cols - 1) // cols
    rows_imgs = []
    for r in range(rows_count):
        row_t = thumbs[r * cols: (r + 1) * cols]
        while len(row_t) < cols:
            row_t.append(np.full((THUMB_H, THUMB_W, 3), 40, dtype=np.uint8))
        rows_imgs.append(np.hstack(row_t))

    header = np.full((36, THUMB_W * cols, 3), 25, dtype=np.uint8)
    cv2.putText(header, title, (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)

    sheet = np.vstack([header] + rows_imgs)
    from PIL import Image
    Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=92)
    print(f"  Saved: {out_path.name}  ({sheet.shape[1]}x{sheet.shape[0]})")


def find_image(img_name: str) -> Path:
    for src in SOURCE_DIRS:
        cand = src / img_name
        if cand.exists():
            return cand
    return Path(img_name)


def main():
    if not MERGED_CSV.exists():
        print(f"ERROR: no existe {MERGED_CSV}")
        sys.exit(1)

    with open(MERGED_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Filas en CSV: {len(rows)}")

    counts = {"PASA": 0, "REVISAR": 0, "RECHAZA": 0}
    u3_safe_upgrades = 0
    direct_bad_on_good = 0
    entry_all = []
    entry_revbad = []

    for row in rows:
        decision = row.get("decision", "PASA")
        counts[decision] = counts.get(decision, 0) + 1
        img_name = row.get("image", "")
        p_good = row.get("quality_u3_p_good", "0") or "0"
        label = row.get("display_label", img_name)
        img_path = find_image(img_name)

        entry_all.append((img_path, decision, label, p_good))
        if decision in ("REVISAR", "RECHAZA"):
            entry_revbad.append((img_path, decision, label, p_good))

        raw  = row.get("quality_u3_decision_raw", "")
        safe = row.get("quality_u3_decision_safe", "")
        if raw == "U3_BAD" and "SAFE" in safe:
            u3_safe_upgrades += 1
        if decision == "RECHAZA" and safe == "U3_BAD":
            direct_bad_on_good += 1

    total = len(rows)
    build_contact_sheet(entry_all, SHEET_ALL,
                        f"U3 Pipeline Integrado - Todas las peras ({total})")
    build_contact_sheet(entry_revbad, SHEET_REVBAD,
                        f"U3 Pipeline Integrado - REVISAR + RECHAZA ({len(entry_revbad)})")

    lines = [
        "U3 INTEGRATED PIPELINE EVAL - RESUMEN",
        "=" * 50,
        f"Total evaluadas              : {total}",
        f"PASA                         : {counts.get('PASA', 0)}",
        f"REVISAR                      : {counts.get('REVISAR', 0)}",
        f"RECHAZA                      : {counts.get('RECHAZA', 0)}",
        "",
        f"BAD directos en peras sanas  : {direct_bad_on_good}",
        f"REVIEW por safe mode         : {u3_safe_upgrades}",
        "",
        "Archivos:",
        f"  {MERGED_CSV}",
        f"  {SHEET_ALL}",
        f"  {SHEET_REVBAD}",
        f"  {SUMMARY_TXT}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
