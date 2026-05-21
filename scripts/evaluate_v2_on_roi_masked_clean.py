"""
evaluate_v2_on_roi_masked_clean.py
-----------------------------------
TAREA 7 of PROMPT_REFINE_ROI_MASKED_PREPROCESSOR_V2.md

Loads the V2 classifier (MobileNetV3-small) and runs inference on all
*_gray_bg_clean.jpg images produced by prepare_quality_roi_masked_previews_v2.py.

DOES NOT train or modify any model.
DOES NOT modify V2, analyze_quality.py, or quality_rules.yaml.
"""

import sys
import csv
from pathlib import Path

import numpy as np
from PIL import Image

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision.transforms as T
import torchvision.models as models

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = PROJECT_ROOT / "outputs/fruits360_quality_cls_v2/best_model.pt"
INPUT_DIR    = PROJECT_ROOT / "outputs/quality_roi_masked_previews_v2/crops"
OUTPUT_DIR   = PROJECT_ROOT / "outputs/quality_roi_masked_previews_v2"
PRED_CSV     = OUTPUT_DIR / "v2_on_gray_bg_clean_predictions.csv"
SHEET_PATH   = OUTPUT_DIR / "v2_on_gray_bg_clean_contact_sheet.jpg"

# ── model ─────────────────────────────────────────────────────────────────────
CLASSES      = ["bad", "good"]   # alphabetical — matches training order
BAD_CONF_THRESHOLD = 0.70        # same operational rule used in previous evals

# ── contact sheet layout ──────────────────────────────────────────────────────
THUMB        = 160
COLS         = 4
SHEET_BG     = (30, 30, 30)
FONT_HEIGHT  = 14   # rough pixel allowance for text label per row cell
try:
    from PIL import ImageDraw, ImageFont
    _HAS_FONT = True
except Exception:
    _HAS_FONT = False


# ─────────────────────────────────────────────────────────────────────────────
def build_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def load_model(device: torch.device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, 2
    )
    state = torch.load(str(MODEL_PATH), map_location=device)
    # handle wrapped checkpoint
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_image(model, transform, img_path: Path, device: torch.device):
    try:
        pil = Image.open(str(img_path)).convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        bad_conf  = float(probs[0])
        good_conf = float(probs[1])
        pred_class = CLASSES[int(probs.argmax())]
        if bad_conf >= BAD_CONF_THRESHOLD:
            decision = "BAD"
        elif bad_conf >= 0.0:   # anything below threshold
            decision = "REVIEW" if bad_conf >= 0.30 else "GOOD"
        else:
            decision = "GOOD"
        # simpler: match original eval logic
        if pred_class == "bad" and bad_conf >= BAD_CONF_THRESHOLD:
            decision = "BAD"
        elif pred_class == "bad":
            decision = "REVIEW"
        else:
            decision = "GOOD"
        return {
            "bad_conf":  round(bad_conf,  4),
            "good_conf": round(good_conf, 4),
            "pred_class": pred_class,
            "decision":   decision,
            "error":      "",
        }
    except Exception as exc:
        return {
            "bad_conf":   None,
            "good_conf":  None,
            "pred_class": "ERROR",
            "decision":   "ERROR",
            "error":      str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
def draw_label(draw, x, y, text, fill=(220, 220, 220)):
    if _HAS_FONT:
        try:
            draw.text((x, y), text, fill=fill)
        except Exception:
            pass


def build_contact_sheet(results: list[dict], out_path: Path):
    """Creates a contact sheet: each cell = image + label below."""
    items = [r for r in results if r["error"] == ""]
    if not items:
        print("  [sheet] no valid images — skipping contact sheet")
        return

    n      = len(items)
    rows   = (n + COLS - 1) // COLS
    cell_w = THUMB
    cell_h = THUMB + FONT_HEIGHT + 4

    sheet_w = cell_w * COLS
    sheet_h = cell_h * rows
    sheet   = Image.new("RGB", (sheet_w, sheet_h), SHEET_BG)

    if _HAS_FONT:
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
    else:
        draw = None

    for idx, r in enumerate(items):
        col = idx % COLS
        row = idx // COLS
        px  = col * cell_w
        py  = row * cell_h

        try:
            img = Image.open(str(r["img_path"])).convert("RGB").resize(
                (THUMB, THUMB), Image.LANCZOS
            )
            sheet.paste(img, (px, py))
        except Exception:
            pass

        label = f"{r['filename_base'][:14]} | {r['decision']} bad={r['bad_conf']:.2f}"
        color = (100, 220, 100) if r["decision"] == "GOOD" else \
                (220, 100, 100) if r["decision"] == "BAD"  else \
                (220, 200, 80)
        if draw:
            try:
                draw.text((px + 2, py + THUMB + 2), label, fill=color, font=font)
            except Exception:
                draw.text((px + 2, py + THUMB + 2), label, fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out_path), quality=92)
    print(f"  [sheet] saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("evaluate_v2_on_roi_masked_clean.py")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"[ERROR] model not found: {MODEL_PATH}")
        sys.exit(1)

    if not INPUT_DIR.exists():
        print(f"[ERROR] input dir not found: {INPUT_DIR}")
        print("  Run prepare_quality_roi_masked_previews_v2.py first.")
        sys.exit(1)

    gray_bg_files = sorted(INPUT_DIR.glob("*_gray_bg_clean.jpg"))
    if not gray_bg_files:
        print("[ERROR] No *_gray_bg_clean.jpg files found in", INPUT_DIR)
        sys.exit(1)

    print(f"Found {len(gray_bg_files)} gray_bg_clean images")
    print(f"Model: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = load_model(device)
    transform = build_transform()

    results = []
    for img_path in gray_bg_files:
        base = img_path.stem.replace("_gray_bg_clean", "")
        pred = predict_image(model, transform, img_path, device)
        pred["filename_base"] = base
        pred["img_path"]      = img_path
        results.append(pred)
        status = f"  {base:<20} | {pred['decision']:<7} bad={pred.get('bad_conf', 'ERR')}"
        print(status)

    # ── CSV ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename_base", "img_path", "bad_conf", "good_conf",
        "pred_class", "decision", "error"
    ]
    with open(str(PRED_CSV), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["img_path"] = str(r["img_path"])
            writer.writerow(row)
    print(f"\n  [csv]   saved: {PRED_CSV.name}")

    # ── contact sheet ─────────────────────────────────────────────────────────
    build_contact_sheet(results, SHEET_PATH)

    # ── summary ──────────────────────────────────────────────────────────────
    total   = len(results)
    errors  = sum(1 for r in results if r["error"])
    good_n  = sum(1 for r in results if r["decision"] == "GOOD")
    review  = sum(1 for r in results if r["decision"] == "REVIEW")
    bad_n   = sum(1 for r in results if r["decision"] == "BAD")

    print()
    print("-" * 40)
    print(f"Total images evaluated : {total}")
    print(f"  GOOD                 : {good_n}")
    print(f"  REVIEW               : {review}")
    print(f"  BAD                  : {bad_n}")
    print(f"  ERROR                : {errors}")
    if total - errors > 0:
        false_bad_rate = (bad_n + review) / (total - errors) * 100
        print(f"  False-BAD/REVIEW rate: {false_bad_rate:.1f}% (all images are GOOD pears)")
    print("-" * 40)
    print()
    print("Outputs:")
    print(f"  {PRED_CSV}")
    print(f"  {SHEET_PATH}")
    print()
    print("No model was trained. V2 was not modified.")


if __name__ == "__main__":
    main()
