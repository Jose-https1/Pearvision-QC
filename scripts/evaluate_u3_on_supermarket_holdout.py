"""
evaluate_u3_on_supermarket_holdout.py
---------------------------------------
Evaluates U3 on the supermarket holdout set.

Input : data/quality_roi_masked_clean_u3/holdout_supermarket/good/
Truth : all images are GOOD pears
Output: outputs/fruits360_quality_cls_u3_roi_masked_clean/supermarket_holdout_eval/

DOES NOT modify V2, analyze_quality.py, or quality_rules.yaml.
"""

import sys
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH  = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt"
THR_JSON    = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/selected_thresholds.json"
HOLDOUT_DIR = PROJECT_ROOT / "data/quality_roi_masked_clean_u3/holdout_supermarket/good"
OUT_DIR     = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean/supermarket_holdout_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES  = ["bad", "good"]
IMG_SIZE = 224
THUMB    = 140


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


def apply_policy(p_bad, p_good, bad_thr, good_thr):
    if p_bad >= bad_thr:
        return "BAD"
    elif p_good >= good_thr:
        return "GOOD"
    else:
        return "REVIEW"


def make_contact_sheet(items, path: Path, title: str, cols=5):
    """items: list of (pil_img, label_str, p_bad, p_good)"""
    if not items:
        blank = Image.new("RGB", (200, 100), (30, 30, 30))
        blank.save(str(path))
        return
    cell_h = THUMB + 32
    cell_w = THUMB + 4
    n = len(items)
    rows = max(1, (n + cols - 1) // cols)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows + 24), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title, fill=(200, 200, 200))
    for idx, (pil_img, decision, p_bad, _) in enumerate(items):
        col = idx % cols
        row = idx // cols
        px = col * cell_w + 2
        py = row * cell_h + 24
        thumb = pil_img.resize((THUMB, THUMB), Image.LANCZOS)
        sheet.paste(thumb, (px, py))
        color = (100, 220, 100) if decision == "GOOD" else \
                (220, 200, 60)  if decision == "REVIEW" else \
                (220, 80,  80)
        draw.text((px, py + THUMB + 2), f"{decision} {p_bad:.2f}", fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(path), quality=92)


def main():
    print("=" * 60)
    print("evaluate_u3_on_supermarket_holdout.py")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"[ERROR] model not found: {MODEL_PATH}")
        sys.exit(1)
    if not THR_JSON.exists():
        print(f"[ERROR] thresholds not found: {THR_JSON}")
        sys.exit(1)
    if not HOLDOUT_DIR.exists():
        print(f"[ERROR] holdout dir not found: {HOLDOUT_DIR}")
        sys.exit(1)

    with open(str(THR_JSON), encoding="utf-8") as fh:
        thresholds = json.load(fh)
    bad_thr  = thresholds["bad_reject_threshold"]
    good_thr = thresholds["good_accept_threshold"]
    print(f"Thresholds: bad>={bad_thr}  good>={good_thr}")

    images = sorted(HOLDOUT_DIR.glob("*.jpg"))
    print(f"Holdout images: {len(images)}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(device)
    transform = build_transform()

    rows = []
    all_items         = []
    review_bad_items  = []

    with torch.no_grad():
        for img_path in images:
            try:
                pil    = Image.open(str(img_path)).convert("RGB")
                tensor = transform(pil).unsqueeze(0).to(device)
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
                p_bad  = float(probs[0])
                p_good = float(probs[1])
                decision = apply_policy(p_bad, p_good, bad_thr, good_thr)
            except Exception as e:
                print(f"  [err] {img_path.name}: {e}")
                p_bad, p_good, decision = 0.5, 0.5, "REVIEW"
                pil = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

            rows.append({
                "filename": img_path.name,
                "stem": img_path.stem,
                "true_class": "good",
                "decision": decision,
                "p_bad":  round(p_bad,  4),
                "p_good": round(p_good, 4),
                "correct": int(decision != "BAD"),  # any non-BAD is acceptable for GOOD pears
            })
            print(f"  {img_path.stem:<20} | {decision:<7} bad={p_bad:.3f} good={p_good:.3f}")
            all_items.append((pil, decision, p_bad, p_good))
            if decision in ("REVIEW", "BAD"):
                review_bad_items.append((pil, decision, p_bad, p_good))

    # Stats
    n_total  = len(rows)
    n_good   = sum(1 for r in rows if r["decision"] == "GOOD")
    n_review = sum(1 for r in rows if r["decision"] == "REVIEW")
    n_bad    = sum(1 for r in rows if r["decision"] == "BAD")
    false_bad_rate = n_bad / n_total if n_total > 0 else 0
    review_rate    = n_review / n_total if n_total > 0 else 0
    mean_conf      = sum(r["p_good"] for r in rows) / n_total if n_total > 0 else 0
    review_bad_ids = [r["stem"] for r in rows if r["decision"] in ("REVIEW","BAD")]

    print()
    print("-" * 40)
    print(f"Total images  : {n_total}")
    print(f"GOOD          : {n_good}")
    print(f"REVIEW        : {n_review}")
    print(f"BAD (false)   : {n_bad}")
    print(f"False BAD rate: {false_bad_rate:.1%}")
    print(f"Review rate   : {review_rate:.1%}")
    print(f"Mean p_good   : {mean_conf:.3f}")
    print("-" * 40)

    # Save predictions CSV
    pred_csv = OUT_DIR / "predictions_supermarket_holdout.csv"
    with open(str(pred_csv), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename","stem","true_class","decision","p_bad","p_good","correct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {pred_csv.name}")

    # Contact sheets
    sheet_all = OUT_DIR / "contact_sheet_supermarket_holdout_all.jpg"
    make_contact_sheet(all_items, sheet_all,
                       f"Holdout supermarket all ({n_total}) - GOOD={n_good} REVIEW={n_review} BAD={n_bad}")
    print(f"Saved: {sheet_all.name}")

    sheet_rb = OUT_DIR / "contact_sheet_supermarket_holdout_review_bad.jpg"
    make_contact_sheet(review_bad_items, sheet_rb,
                       f"Holdout supermarket REVIEW+BAD ({len(review_bad_items)} images)",
                       cols=4)
    print(f"Saved: {sheet_rb.name}")

    # Summary text
    summary_lines = [
        "=== U3 Supermarket Holdout Evaluation ===",
        "",
        f"Model: outputs/fruits360_quality_cls_u3_roi_masked_clean/best_model.pt",
        f"Thresholds: bad>={bad_thr}  good>={good_thr}",
        "",
        f"Total images evaluated : {n_total}",
        f"Ground truth           : all GOOD",
        "",
        f"GOOD                   : {n_good} ({n_good/n_total:.1%})",
        f"REVIEW                 : {n_review} ({n_review/n_total:.1%})",
        f"BAD (false positive)   : {n_bad} ({false_bad_rate:.1%})",
        "",
        f"Mean p_good            : {mean_conf:.4f}",
        "",
    ]
    if review_bad_ids:
        summary_lines.append("REVIEW or BAD cases:")
        for sid in review_bad_ids:
            r = next(rr for rr in rows if rr["stem"] == sid)
            summary_lines.append(f"  {sid}: {r['decision']} (bad={r['p_bad']:.3f})")
    else:
        summary_lines.append("No REVIEW or BAD cases - all pears correctly identified as GOOD!")
    summary_lines += [
        "",
        "No model was modified.",
        "No V2 was modified.",
        "No analyze_quality.py was modified.",
        "No quality_rules.yaml was modified.",
    ]
    summary_txt = OUT_DIR / "summary.txt"
    with open(str(summary_txt), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))
    print(f"Saved: {summary_txt.name}")

    print()
    print("=== holdout eval DONE ===")
    return n_good, n_review, n_bad, false_bad_rate


if __name__ == "__main__":
    main()
