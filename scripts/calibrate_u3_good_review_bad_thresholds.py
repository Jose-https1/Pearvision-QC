"""
calibrate_u3_good_review_bad_thresholds.py
-------------------------------------------
Calibrates GOOD / REVIEW / BAD thresholds for U3.

Policy:
  if p_bad >= bad_reject_threshold  -> BAD
  elif p_good >= good_accept_threshold -> GOOD
  else                               -> REVIEW

Uses val set for calibration. Selects thresholds that:
  1. Minimize false BAD on GOOD pears (priority).
  2. Maintain BAD detection on truly bad pears.
  3. Send ambiguous cases to REVIEW.

DOES NOT modify V2, analyze_quality.py, or quality_rules.yaml.
"""

import sys
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt"
DATA_ROOT  = PROJECT_ROOT / "data/quality_roi_masked_clean_u3"
OUT_DIR    = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean"

CLASSES = ["bad", "good"]  # bad=0, good=1
IMG_SIZE = 224

BAD_THRESHOLDS  = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
GOOD_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def build_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)
    state = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def collect_images(split_dir: Path):
    """Returns list of (path, true_class_idx, true_class_name)."""
    items = []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.jpg")):
            items.append((p, cls_idx, cls_name))
    return items


@torch.no_grad()
def run_inference(model, transform, items, device):
    """Returns list of (p_bad, p_good, true_class_idx, true_class_name, path)."""
    results = []
    for path, true_idx, true_name in items:
        try:
            pil    = Image.open(str(path)).convert("RGB")
            tensor = transform(pil).unsqueeze(0).to(device)
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            results.append((float(probs[0]), float(probs[1]), true_idx, true_name, str(path)))
        except Exception as e:
            print(f"  [err] {path.name}: {e}")
    return results


def apply_policy(p_bad, p_good, bad_thr, good_thr):
    if p_bad >= bad_thr:
        return "BAD"
    elif p_good >= good_thr:
        return "GOOD"
    else:
        return "REVIEW"


def evaluate_thresholds(results, bad_thr, good_thr):
    """Returns metrics dict for given threshold pair."""
    n_bad_total   = sum(1 for r in results if r[3] == "bad")
    n_good_total  = sum(1 for r in results if r[3] == "good")

    n_bad_correct  = 0   # true bad -> BAD
    n_bad_review   = 0   # true bad -> REVIEW
    n_bad_to_good  = 0   # true bad -> GOOD (worst FN)
    n_good_correct = 0   # true good -> GOOD
    n_good_review  = 0   # true good -> REVIEW
    n_good_to_bad  = 0   # true good -> BAD (worst FP)

    for p_bad, p_good, true_idx, true_name, _ in results:
        decision = apply_policy(p_bad, p_good, bad_thr, good_thr)
        if true_name == "bad":
            if decision == "BAD":    n_bad_correct += 1
            elif decision == "REVIEW": n_bad_review += 1
            else:                    n_bad_to_good += 1
        else:  # true good
            if decision == "GOOD":   n_good_correct += 1
            elif decision == "REVIEW": n_good_review += 1
            else:                    n_good_to_bad += 1

    total = len(results)
    n_good_safe = n_good_correct + n_good_review  # GOOD that are NOT BAD
    false_bad_rate = n_good_to_bad / n_good_total if n_good_total > 0 else 0
    bad_recall     = n_bad_correct / n_bad_total  if n_bad_total  > 0 else 0
    bad_catch_rate = (n_bad_correct + n_bad_review) / n_bad_total if n_bad_total > 0 else 0

    return {
        "bad_thr":          bad_thr,
        "good_thr":         good_thr,
        "n_bad_total":      n_bad_total,
        "n_good_total":     n_good_total,
        "bad_correct":      n_bad_correct,
        "bad_review":       n_bad_review,
        "bad_to_good":      n_bad_to_good,
        "good_correct":     n_good_correct,
        "good_review":      n_good_review,
        "good_to_bad":      n_good_to_bad,
        "false_bad_rate":   round(false_bad_rate, 4),
        "bad_recall":       round(bad_recall, 4),
        "bad_catch_rate":   round(bad_catch_rate, 4),  # BAD sent to BAD or REVIEW
        "review_rate_good": round(n_good_review / n_good_total, 4) if n_good_total > 0 else 0,
    }


def select_best_thresholds(metrics_rows):
    """
    Selection criteria (in priority order):
    1. false_bad_rate == 0   (no GOOD pear sent directly to BAD)
    2. bad_catch_rate >= 0.90 (at least 90% of bad caught in BAD or REVIEW)
    3. minimize review_rate_good (minimize GOOD pears in REVIEW)
    4. maximize bad_recall (most bad sent directly to BAD)
    """
    candidates = [r for r in metrics_rows if r["false_bad_rate"] == 0.0 and r["bad_catch_rate"] >= 0.90]
    if not candidates:
        # Relax: false_bad_rate <= 0.05
        candidates = [r for r in metrics_rows if r["false_bad_rate"] <= 0.05 and r["bad_catch_rate"] >= 0.85]
    if not candidates:
        # Relax further
        candidates = sorted(metrics_rows, key=lambda r: (r["false_bad_rate"], -r["bad_catch_rate"]))[:1]

    # Among candidates, minimize review_rate_good, then maximize bad_recall
    best = sorted(candidates, key=lambda r: (r["review_rate_good"], -r["bad_recall"]))[0]
    return best


def main():
    print("=" * 60)
    print("calibrate_u3_good_review_bad_thresholds.py")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(device)
    transform = build_transform()

    # Run on val set
    val_items   = collect_images(DATA_ROOT / "val")
    test_items  = collect_images(DATA_ROOT / "test")
    print(f"Val  items: {len(val_items)}  (good={sum(1 for i in val_items if i[2]=='good')}, bad={sum(1 for i in val_items if i[2]=='bad')})")
    print(f"Test items: {len(test_items)} (good={sum(1 for i in test_items if i[2]=='good')}, bad={sum(1 for i in test_items if i[2]=='bad')})")

    val_results  = run_inference(model, transform, val_items,  device)
    test_results = run_inference(model, transform, test_items, device)

    print(f"\nVal p_bad stats:  min={min(r[0] for r in val_results):.3f}  max={max(r[0] for r in val_results):.3f}  mean={sum(r[0] for r in val_results)/len(val_results):.3f}")

    # Grid search on val
    metrics_rows = []
    for bad_thr in BAD_THRESHOLDS:
        for good_thr in GOOD_THRESHOLDS:
            m = evaluate_thresholds(val_results, bad_thr, good_thr)
            metrics_rows.append(m)

    # Save calibration CSV
    fieldnames = list(metrics_rows[0].keys())
    cal_csv = OUT_DIR / "threshold_calibration.csv"
    with open(str(cal_csv), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\nSaved calibration CSV: {cal_csv.name} ({len(metrics_rows)} rows)")

    # Select best
    best = select_best_thresholds(metrics_rows)
    print(f"\nSelected thresholds:")
    print(f"  bad_reject_threshold  = {best['bad_thr']}")
    print(f"  good_accept_threshold = {best['good_thr']}")
    print(f"  false_bad_rate (val)  = {best['false_bad_rate']}")
    print(f"  bad_recall (val)      = {best['bad_recall']}")
    print(f"  bad_catch_rate (val)  = {best['bad_catch_rate']}")
    print(f"  review_rate_good(val) = {best['review_rate_good']}")

    # Evaluate selected thresholds on test set
    test_m = evaluate_thresholds(test_results, best["bad_thr"], best["good_thr"])
    print(f"\nTest results with selected thresholds:")
    print(f"  false_bad_rate (test) = {test_m['false_bad_rate']}")
    print(f"  bad_recall (test)     = {test_m['bad_recall']}")
    print(f"  bad_catch_rate (test) = {test_m['bad_catch_rate']}")

    selected = {
        "bad_reject_threshold":  best["bad_thr"],
        "good_accept_threshold": best["good_thr"],
        "classes": CLASSES,
        "class_bad_idx": 0,
        "class_good_idx": 1,
        "selection_criterion": "false_bad_rate==0 AND bad_catch_rate>=0.90, minimize review_rate_good",
        "val_false_bad_rate": best["false_bad_rate"],
        "val_bad_recall":     best["bad_recall"],
        "val_bad_catch_rate": best["bad_catch_rate"],
        "test_false_bad_rate": test_m["false_bad_rate"],
        "test_bad_recall":     test_m["bad_recall"],
        "test_bad_catch_rate": test_m["bad_catch_rate"],
    }

    thr_json = OUT_DIR / "selected_thresholds.json"
    with open(str(thr_json), "w", encoding="utf-8") as fh:
        json.dump(selected, fh, indent=2)
    print(f"\nSaved: {thr_json.name}")

    # Write calibration report
    report_path = PROJECT_ROOT / "reports/u3_threshold_calibration_report.md"
    with open(str(report_path), "w", encoding="utf-8") as fh:
        fh.write("# U3 Threshold Calibration Report\n\n")
        fh.write("**Fecha:** 2026-05-21\n\n")
        fh.write("## Politica de decision\n\n")
        fh.write("```\n")
        fh.write("if p_bad >= bad_reject_threshold  -> BAD\n")
        fh.write("elif p_good >= good_accept_threshold -> GOOD\n")
        fh.write("else                               -> REVIEW\n")
        fh.write("```\n\n")
        fh.write("## Criterio de seleccion\n\n")
        fh.write("1. false_bad_rate == 0 (BUENA pera no va a BAD directamente)\n")
        fh.write("2. bad_catch_rate >= 0.90 (al menos 90% de las malas atrapadas en BAD o REVIEW)\n")
        fh.write("3. Minimizar review_rate de peras buenas\n")
        fh.write("4. Maximizar bad_recall\n\n")
        fh.write("## Umbrales seleccionados\n\n")
        fh.write(f"| Parametro | Valor |\n|---|---|\n")
        fh.write(f"| bad_reject_threshold | {best['bad_thr']} |\n")
        fh.write(f"| good_accept_threshold | {best['good_thr']} |\n\n")
        fh.write("## Resultados en validation set\n\n")
        fh.write(f"| Metrica | Valor |\n|---|---|\n")
        for k in ["false_bad_rate","bad_recall","bad_catch_rate","review_rate_good",
                  "bad_correct","bad_review","bad_to_good","good_correct","good_review","good_to_bad"]:
            fh.write(f"| {k} | {best[k]} |\n")
        fh.write("\n## Resultados en test set\n\n")
        fh.write(f"| Metrica | Valor |\n|---|---|\n")
        for k in ["false_bad_rate","bad_recall","bad_catch_rate"]:
            fh.write(f"| {k} | {test_m[k]} |\n")
        fh.write("\n## Notas\n\n")
        fh.write("- Grid search sobre val set (no test set) para evitar data leakage.\n")
        fh.write("- No se modifico V2.\n")
    print(f"Saved: {report_path.name}")

    print()
    print("=== calibration DONE ===")
    return selected


if __name__ == "__main__":
    main()
