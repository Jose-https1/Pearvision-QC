"""
test_integrated_u3_pipeline.py
Evaluates the U3-integrated pipeline by calling _process_one() directly
(no subprocess), then generates contact sheets + summary.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SOURCE_DIRS = [
    PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3",
    PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_good_batch_v2",
    PROJECT_ROOT / "data/supermarket_good_hard_examples_v1/images",
    PROJECT_ROOT / "data/supermarket_good_hard_examples_v2/images",
]

OUT_DIR  = PROJECT_ROOT / "outputs" / "u3_integrated_pipeline_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

U3_MODEL = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt"
U3_THR   = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json"
RULES    = PROJECT_ROOT / "configs/quality_rules.yaml"

MERGED_CSV   = OUT_DIR / "resultados_integrated_u3.csv"
SHEET_ALL    = OUT_DIR / "contact_sheet_integrated_u3_all.jpg"
SHEET_REVBAD = OUT_DIR / "contact_sheet_integrated_u3_review_bad.jpg"
SUMMARY_TXT  = OUT_DIR / "summary.txt"

DECISION_COLOR = {"PASA": (0, 200, 0), "REVISAR": (0, 165, 255), "RECHAZA": (0, 0, 220)}
THUMB_W, THUMB_H = 220, 220
COLS = 6

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
def collect_images(src: Path):
    if src.is_file():
        return [src]
    return sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS])


def load_thumb(img_path: Path, w=THUMB_W, h=THUMB_H):
    from PIL import Image
    try:
        pil = Image.open(str(img_path)).convert("RGB")
        pil = pil.resize((w, h))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return np.full((h, w, 3), 80, dtype=np.uint8)


def add_label(thumb, text, color):
    cv2.rectangle(thumb, (0, thumb.shape[0] - 28), (thumb.shape[1], thumb.shape[0]), (30, 30, 30), -1)
    cv2.putText(thumb, text, (4, thumb.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return thumb


def build_contact_sheet(entries, out_path: Path, title: str, cols=COLS):
    if not entries:
        blank = np.full((THUMB_H + 40, THUMB_W * cols, 3), 50, dtype=np.uint8)
        cv2.putText(blank, "Sin imagenes", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        from PIL import Image
        Image.fromarray(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=90)
        return

    thumbs = []
    for img_path, decision, label in entries:
        t = load_thumb(img_path)
        color = DECISION_COLOR.get(decision, (200, 200, 200))
        short = label[:28] if len(label) > 28 else label
        add_label(t, f"{decision} {short}", color)
        bcolor = DECISION_COLOR.get(decision, (200, 200, 200))
        cv2.rectangle(t, (0, 0), (t.shape[1]-1, t.shape[0]-1), bcolor, 3)
        thumbs.append(t)

    rows_count = (len(thumbs) + cols - 1) // cols
    rows_imgs = []
    for r in range(rows_count):
        row_thumbs = thumbs[r * cols: (r + 1) * cols]
        while len(row_thumbs) < cols:
            row_thumbs.append(np.full((THUMB_H, THUMB_W, 3), 40, dtype=np.uint8))
        rows_imgs.append(np.hstack(row_thumbs))

    header = np.full((40, THUMB_W * cols, 3), 30, dtype=np.uint8)
    cv2.putText(header, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    sheet = np.vstack([header] + rows_imgs)
    from PIL import Image
    Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).save(str(out_path), quality=90)
    print(f"  Contact sheet: {out_path}")


# ---------------------------------------------------------------------------
def main():
    # Import pipeline internals
    import analyze_quality as aq

    # Load rules and seg_config from config files
    seg_data = aq._load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    seg_config = seg_data.get("segmentation", seg_data)
    rules = aq._load_yaml(RULES)

    # Load U3
    u3_model, u3_thresholds = aq._load_u3_classifier(U3_MODEL, U3_THR)
    if u3_model is None:
        print("ERROR: no se pudo cargar U3")
        sys.exit(1)
    print(f"U3 cargado. bad_thr={u3_thresholds.get('bad_reject_threshold')}")

    # Args namespace with same defaults as analyze_quality's argparse
    class Args:
        seg_config = "configs/thresholds.yaml"
        rules = "configs/quality_rules.yaml"
        seg_method = "grabcut"
        min_pear_area = 5.0
        save = True
        no_save = False
        show = False
        yolo_conf = 0.5
        defect_conf = 0.25
        use_yolo = False
        use_defect_model = False
        use_quality_cls = False
        quality_cls_bad_thr = 0.85
        quality_cls_affect_decision = False
        use_quality_u3 = True
        quality_u3_safe_mode = True
        output = None
        image = None
        source = None

    args = Args()

    # Find existing source dirs
    existing = [d for d in SOURCE_DIRS if d.exists()]
    if not existing:
        print("ERROR: no se encontraron carpetas de imagenes")
        sys.exit(1)

    print(f"Carpetas: {len(existing)}")
    for d in existing:
        print(f"  {d}")

    all_rows = []
    fieldnames = None

    for src_dir in existing:
        images = collect_images(src_dir)
        print(f"\n--- {src_dir.name}: {len(images)} imagenes ---")
        for i, img_path in enumerate(images):
            result = aq._process_one(
                img_path, seg_config, rules, args,
                yolo_model=None,
                defect_model=None,
                defect_conf=0.25,
                quality_cls_model=None,
                quality_cls_bad_thr=0.85,
                quality_cls_affect_decision=False,
                u3_model=u3_model,
                u3_thresholds=u3_thresholds,
                u3_safe_mode=True,
            )
            result["image"] = img_path.name
            result["_src_dir"] = str(src_dir)
            m = result["metrics"]
            dec = result["decision"]
            p_good = m.get("quality_u3_p_good", 0)
            p_bad  = m.get("quality_u3_p_bad", 0)
            print(f"  [{i+1}/{len(images)}] {img_path.name:30s} {dec:8s}  "
                  f"U3 p_good={p_good:.3f} p_bad={p_bad:.3f}")
            all_rows.append(result)

    if not all_rows:
        print("ERROR: sin resultados")
        sys.exit(1)

    # Build CSV fieldnames
    fieldnames = [
        "image", "decision", "estimated_category", "display_label",
        "capture_valid", "capture_label", "capture_reason",
        "defect_pct", "dark_rot_pct", "max_region_pct",
        "pear_visible_pct", "body_visible_pct",
        "mask_area_pct", "body_area_pct",
        "bbox_fill_ratio", "bbox_aspect_ratio",
        "mask_components", "border_touch_pct", "mask_irregularity_ratio",
        "mask_warning",
        "original_width", "original_height",
        "processed_width", "processed_height",
        "detector_used", "detector_conf",
        "crop_x1", "crop_y1", "crop_x2", "crop_y2", "crop_margin",
        "yolo_defect_count", "yolo_defect_area_pct",
        "yolo_defect_max_conf", "yolo_defect_classes",
        "brown_dark_pct", "dark_area_pct",
        "mask_source", "mask_quality_ok", "mask_fail_reason",
        "quality_cls_used", "quality_cls_source", "quality_cls_pred",
        "quality_cls_good_conf", "quality_cls_bad_conf", "quality_cls_max_conf",
        "quality_cls_action",
        "body_l_mean",
        "quality_u3_enabled", "quality_u3_status", "quality_u3_pred",
        "quality_u3_p_good", "quality_u3_p_bad",
        "quality_u3_decision_raw", "quality_u3_decision_safe",
        "final_decision_before_u3", "final_decision_after_u3", "final_decision_reason",
        "source_dir",
    ]

    with open(MERGED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            m = r["metrics"]
            ci = r.get("crop_info", {})
            writer.writerow({
                "image": r["image"],
                "decision": r["decision"],
                "estimated_category": r["estimated_category"],
                "display_label": r["display_label"],
                "capture_valid": r["capture_valid"],
                "capture_label": r["capture_label"],
                "capture_reason": r["capture_reason"],
                "defect_pct": m["defect_pct"],
                "dark_rot_pct": m["rot_pct"],
                "max_region_pct": m["largest_defect_pct"],
                "pear_visible_pct": m["pear_visible_pct"],
                "body_visible_pct": m["body_visible_pct"],
                "mask_area_pct": m.get("mask_area_pct", 0.0),
                "body_area_pct": m.get("body_area_pct", 0.0),
                "bbox_fill_ratio": m.get("bbox_fill_ratio", 0.0),
                "bbox_aspect_ratio": m.get("bbox_aspect_ratio", 0.0),
                "mask_components": m.get("mask_components", 0),
                "border_touch_pct": m.get("border_touch_pct", 0.0),
                "mask_irregularity_ratio": m.get("mask_irregularity_ratio", 0.0),
                "mask_warning": r["mask_warning"],
                "original_width": r["orig_w"],
                "original_height": r["orig_h"],
                "processed_width": r["proc_w"],
                "processed_height": r["proc_h"],
                "detector_used": ci.get("detector_used", False),
                "detector_conf": ci.get("detector_conf", 0.0),
                "crop_x1": ci.get("crop_x1", 0),
                "crop_y1": ci.get("crop_y1", 0),
                "crop_x2": ci.get("crop_x2", 0),
                "crop_y2": ci.get("crop_y2", 0),
                "crop_margin": ci.get("crop_margin", 0.0),
                "yolo_defect_count": m.get("yolo_defect_count", 0),
                "yolo_defect_area_pct": m.get("yolo_defect_area_pct", 0.0),
                "yolo_defect_max_conf": m.get("yolo_defect_max_conf", 0.0),
                "yolo_defect_classes": m.get("yolo_defect_classes", ""),
                "brown_dark_pct": m.get("brown_dark_pct", 0.0),
                "dark_area_pct": m.get("dark_area_pct", 0.0),
                "mask_source": ci.get("mask_source", "classic"),
                "mask_quality_ok": ci.get("mask_quality_ok", True),
                "mask_fail_reason": ci.get("mask_fail_reason", ""),
                "quality_cls_used": m.get("quality_cls_used", False),
                "quality_cls_source": m.get("quality_cls_source", "not_used"),
                "quality_cls_pred": m.get("quality_cls_pred", "unknown"),
                "quality_cls_good_conf": m.get("quality_cls_good_conf", 0.0),
                "quality_cls_bad_conf": m.get("quality_cls_bad_conf", 0.0),
                "quality_cls_max_conf": m.get("quality_cls_max_conf", 0.0),
                "quality_cls_action": m.get("quality_cls_action", ""),
                "body_l_mean": m.get("body_l_mean", 128.0),
                "quality_u3_enabled": m.get("quality_u3_enabled", False),
                "quality_u3_status": m.get("quality_u3_status", "not_used"),
                "quality_u3_pred": m.get("quality_u3_pred", "unknown"),
                "quality_u3_p_good": m.get("quality_u3_p_good", 0.0),
                "quality_u3_p_bad": m.get("quality_u3_p_bad", 0.0),
                "quality_u3_decision_raw": m.get("quality_u3_decision_raw", ""),
                "quality_u3_decision_safe": m.get("quality_u3_decision_safe", ""),
                "final_decision_before_u3": m.get("final_decision_before_u3", ""),
                "final_decision_after_u3": m.get("final_decision_after_u3", ""),
                "final_decision_reason": m.get("final_decision_reason", ""),
                "source_dir": r.get("_src_dir", ""),
            })
    print(f"\nCSV consolidado: {MERGED_CSV}  ({len(all_rows)} imagenes)")

    # Contact sheets
    counts = {"PASA": 0, "REVISAR": 0, "RECHAZA": 0}
    u3_safe_upgrades = 0
    direct_bad_on_good = 0
    entry_all = []
    entry_revbad = []

    for r in all_rows:
        decision = r["decision"]
        m = r["metrics"]
        counts[decision] = counts.get(decision, 0) + 1
        img_name = r["image"]
        label = r.get("display_label", img_name)
        src_dir_str = r.get("_src_dir", "")

        img_path = None
        if src_dir_str:
            cand = Path(src_dir_str) / img_name
            if cand.exists():
                img_path = cand
        if img_path is None:
            for src in existing:
                cand = src / img_name
                if cand.exists():
                    img_path = cand
                    break
        if img_path is None:
            img_path = Path(img_name)

        entry_all.append((img_path, decision, label))
        if decision in ("REVISAR", "RECHAZA"):
            entry_revbad.append((img_path, decision, label))

        raw  = m.get("quality_u3_decision_raw", "")
        safe = m.get("quality_u3_decision_safe", "")
        if raw == "U3_BAD" and "SAFE" in safe:
            u3_safe_upgrades += 1
        if decision == "RECHAZA" and safe == "U3_BAD":
            direct_bad_on_good += 1

    build_contact_sheet(entry_all, SHEET_ALL,
                        f"U3 Pipeline - Todas ({len(entry_all)} peras)")
    build_contact_sheet(entry_revbad, SHEET_REVBAD,
                        f"U3 Pipeline - REVISAR+RECHAZA ({len(entry_revbad)} peras)")

    total = len(all_rows)
    summary_lines = [
        "U3 INTEGRATED PIPELINE EVAL - RESUMEN",
        "=" * 50,
        f"Total evaluadas : {total}",
        f"PASA            : {counts.get('PASA', 0)}",
        f"REVISAR         : {counts.get('REVISAR', 0)}",
        f"RECHAZA         : {counts.get('RECHAZA', 0)}",
        "",
        f"BAD directos en peras sanas    : {direct_bad_on_good}",
        f"REVIEW por safe mode (U3->REVIEW): {u3_safe_upgrades}",
        "",
        "Archivos generados:",
        f"  {MERGED_CSV}",
        f"  {SHEET_ALL}",
        f"  {SHEET_REVBAD}",
        f"  {SUMMARY_TXT}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")
    print()
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
