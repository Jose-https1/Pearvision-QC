"""
build_quality_roi_masked_clean_u3_dataset.py
--------------------------------------------
Builds the U3 ROI/masked clean dataset at:
  data/quality_roi_masked_clean_u3/

Sources:
  1. data/quality_fruits360_human_v2/ (Fruits-360, white bg) -> threshold masking -> gray bg
  2. outputs/quality_roi_masked_previews_v2/crops/*_gray_bg_clean.jpg
     - hard_examples_v1 (20) and hard_examples_v2 (22) -> GOOD train/val/test
     - batch_v3 (22) -> holdout_supermarket/good/ ONLY

Split: 70/15/15 stratified on non-holdout data.

DOES NOT modify V2, analyze_quality.py, or quality_rules.yaml.
"""

import sys
import shutil
import csv
import random
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── paths ─────────────────────────────────────────────────────────────────────
V2_DATASET   = PROJECT_ROOT / "data/quality_fruits360_human_v2"
V2_MASTER    = V2_DATASET / "metadata/quality_fruits360_human_v2_master.csv"
GRAY_CROPS   = PROJECT_ROOT / "outputs/quality_roi_masked_previews_v2/crops"
HARD_V1      = PROJECT_ROOT / "data/supermarket_good_hard_examples_v1/images"
HARD_V2      = PROJECT_ROOT / "data/supermarket_good_hard_examples_v2/images"
HOLDOUT_SRC  = PROJECT_ROOT / "data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3"

OUT_ROOT     = PROJECT_ROOT / "data/quality_roi_masked_clean_u3"
META_DIR     = OUT_ROOT / "metadata"

SPLITS = ["train", "val", "test"]
CLASSES = ["good", "bad"]

RANDOM_SEED = 42
IMG_SIZE = 224

# batch_v3 IDs that must go to holdout only
HOLDOUT_STEMS = {p.stem for p in HOLDOUT_SRC.glob("*.jpg")} | \
                {p.stem for p in HOLDOUT_SRC.glob("*.png")} | \
                {p.stem for p in HOLDOUT_SRC.glob("*.JPG")}


# ─────────────────────────────────────────────────────────────────────────────
def pil_save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=92)


def apply_fruits360_masking(img_path: Path) -> Image.Image | None:
    """Load a Fruits-360 image (white bg) and return gray_bg version 224x224."""
    try:
        pil = Image.open(str(img_path)).convert("RGB")
        pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(pil)

        # Threshold: background = all channels > 235 (Fruits-360 white bg)
        bg = (arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235)
        fg = (~bg).astype(np.uint8) * 255

        if HAS_CV2:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)

        gray_bg = np.full_like(arr, 128)
        result = np.where(fg[:, :, np.newaxis] > 0, arr, gray_bg)
        return Image.fromarray(result.astype(np.uint8))
    except Exception as e:
        print(f"    [mask-err] {img_path.name}: {e}")
        return None


def load_gray_bg_clean(stem: str) -> Path | None:
    """Return path to existing gray_bg_clean for a given stem, or None."""
    p = GRAY_CROPS / f"{stem}_gray_bg_clean.jpg"
    return p if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
def collect_sources():
    """Returns (good_items, bad_items, holdout_items).

    Each item: {"stem": str, "source": Path, "masked_path": Path|None, "origin": str}
    masked_path is set only if a gray_bg_clean already exists (supermarket images).
    """
    good_items  = []
    bad_items   = []
    holdout_items = []

    # 1. V2 Fruits-360 images (good + bad)
    if V2_MASTER.exists():
        with open(str(V2_MASTER), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            cls = r.get("class", "").lower()
            fn  = r.get("filename", "")
            if not fn or cls not in ("good", "bad"):
                continue
            # Find file
            found = None
            for split in SPLITS:
                candidate = V2_DATASET / split / cls / fn
                if candidate.exists():
                    found = candidate
                    break
            if found is None:
                continue
            stem = Path(fn).stem
            if stem in HOLDOUT_STEMS:
                continue  # safety check
            item = {"stem": stem, "source": found, "masked_path": None, "origin": "fruits360_v2"}
            if cls == "good":
                good_items.append(item)
            else:
                bad_items.append(item)

    # 2. Hard examples v1 (supermarket GOOD)
    for p in sorted(HARD_V1.glob("*.jpg")):
        stem = p.stem
        if stem in HOLDOUT_STEMS:
            continue
        gbc = load_gray_bg_clean(stem)
        good_items.append({"stem": stem, "source": p, "masked_path": gbc, "origin": "hard_v1"})

    # 3. Hard examples v2 (supermarket GOOD)
    for p in sorted(HARD_V2.glob("*.jpg")):
        stem = p.stem
        if stem in HOLDOUT_STEMS:
            continue
        gbc = load_gray_bg_clean(stem)
        good_items.append({"stem": stem, "source": p, "masked_path": gbc, "origin": "hard_v2"})

    # 4. Batch v3 (holdout only)
    for p in sorted(HOLDOUT_SRC.glob("*.jpg")):
        stem = p.stem
        gbc = load_gray_bg_clean(stem)
        holdout_items.append({"stem": stem, "source": p, "masked_path": gbc, "origin": "batch_v3"})

    # Deduplicate by stem
    seen = set()
    good_unique, bad_unique = [], []
    for item in good_items:
        if item["stem"] not in seen:
            seen.add(item["stem"])
            good_unique.append(item)
    for item in bad_items:
        if item["stem"] not in seen:
            seen.add(item["stem"])
            bad_unique.append(item)

    return good_unique, bad_unique, holdout_items


def stratified_split(items, train_frac=0.70, val_frac=0.15, seed=RANDOM_SEED):
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = math.floor(n * train_frac)
    n_val   = math.floor(n * val_frac)
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


def process_and_copy(item: dict, dest: Path, masked_cache: Path | None = None) -> bool:
    """Generate gray_bg image and copy to dest. Returns True on success."""
    if item["masked_path"] is not None and item["masked_path"].exists():
        # Reuse existing gray_bg_clean
        try:
            pil = Image.open(str(item["masked_path"])).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            pil_save(pil, dest)
            return True
        except Exception as e:
            print(f"    [copy-err] {item['stem']}: {e}")
            return False
    else:
        # Apply masking (Fruits-360 threshold method)
        pil = apply_fruits360_masking(item["source"])
        if pil is None:
            return False
        pil_save(pil, dest)
        return True


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("build_quality_roi_masked_clean_u3_dataset.py")
    print("=" * 60)

    good_items, bad_items, holdout_items = collect_sources()
    print(f"GOOD (non-holdout): {len(good_items)}")
    print(f"BAD  (non-holdout): {len(bad_items)}")
    print(f"Holdout supermarket: {len(holdout_items)}")
    print()

    # Split
    good_train, good_val, good_test = stratified_split(good_items)
    bad_train,  bad_val,  bad_test  = stratified_split(bad_items)

    split_map = {
        "train": {"good": good_train, "bad": bad_train},
        "val":   {"good": good_val,   "bad": bad_val},
        "test":  {"good": good_test,  "bad": bad_test},
    }

    print("Split summary:")
    for s in SPLITS:
        for c in CLASSES:
            print(f"  {s}/{c}: {len(split_map[s][c])}")
    print(f"  holdout_supermarket/good: {len(holdout_items)}")
    print()

    # Create directories
    for s in SPLITS:
        for c in CLASSES:
            (OUT_ROOT / s / c).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "holdout_supermarket" / "good").mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Process and copy images
    master_rows = []
    errors = 0

    for split_name, cls_dict in split_map.items():
        for cls_name, items in cls_dict.items():
            for item in items:
                dest = OUT_ROOT / split_name / cls_name / f"{item['stem']}.jpg"
                ok = process_and_copy(item, dest)
                if ok:
                    master_rows.append({
                        "stem": item["stem"],
                        "class": cls_name,
                        "split": split_name,
                        "origin": item["origin"],
                        "source": str(item["source"]),
                        "dest": str(dest),
                        "status": "ok",
                    })
                else:
                    errors += 1
                    master_rows.append({
                        "stem": item["stem"],
                        "class": cls_name,
                        "split": split_name,
                        "origin": item["origin"],
                        "source": str(item["source"]),
                        "dest": str(dest),
                        "status": "error",
                    })

    # Holdout
    for item in holdout_items:
        dest = OUT_ROOT / "holdout_supermarket" / "good" / f"{item['stem']}.jpg"
        ok = process_and_copy(item, dest)
        master_rows.append({
            "stem": item["stem"],
            "class": "good",
            "split": "holdout_supermarket",
            "origin": item["origin"],
            "source": str(item["source"]),
            "dest": str(dest),
            "status": "ok" if ok else "error",
        })

    print(f"Processed {len(master_rows)} images, {errors} errors")

    # Write master CSV
    master_csv = META_DIR / "u3_master.csv"
    with open(str(master_csv), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem","class","split","origin","source","dest","status"])
        writer.writeheader()
        writer.writerows(master_rows)
    print(f"  Saved: {master_csv.name}")

    # Write split summary CSV
    split_summary = META_DIR / "u3_split_summary.csv"
    with open(str(split_summary), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "class", "count"])
        for s in SPLITS + ["holdout_supermarket"]:
            for c in CLASSES:
                if s == "holdout_supermarket" and c == "bad":
                    continue
                cnt = sum(1 for r in master_rows if r["split"]==s and r["class"]==c and r["status"]=="ok")
                writer.writerow([s, c, cnt])
    print(f"  Saved: {split_summary.name}")

    # Write holdout CSV
    holdout_csv = META_DIR / "u3_holdout_supermarket.csv"
    holdout_rows = [r for r in master_rows if r["split"] == "holdout_supermarket"]
    with open(str(holdout_csv), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem","class","split","origin","source","dest","status"])
        writer.writeheader()
        writer.writerows(holdout_rows)
    print(f"  Saved: {holdout_csv.name}")

    # Verify actual file counts
    print()
    print("Verification (actual files on disk):")
    for s in SPLITS:
        for c in CLASSES:
            cnt = len(list((OUT_ROOT / s / c).glob("*.jpg")))
            print(f"  {s}/{c}: {cnt}")
    holdout_cnt = len(list((OUT_ROOT / "holdout_supermarket" / "good").glob("*.jpg")))
    print(f"  holdout_supermarket/good: {holdout_cnt}")

    # Write build report
    report_path = PROJECT_ROOT / "reports/build_quality_roi_masked_clean_u3_dataset_report.md"
    ok_rows = [r for r in master_rows if r["status"]=="ok"]
    err_rows = [r for r in master_rows if r["status"]=="error"]

    def cnt(split, cls):
        return sum(1 for r in ok_rows if r["split"]==split and r["class"]==cls)

    with open(str(report_path), "w", encoding="utf-8") as fh:
        fh.write("# Build U3 Dataset Report\n\n")
        fh.write("**Fecha:** 2026-05-21\n\n")
        fh.write("## Conteos por split\n\n")
        fh.write("| split | good | bad | total |\n")
        fh.write("|---|---|---|---|\n")
        for s in SPLITS:
            g, b = cnt(s,"good"), cnt(s,"bad")
            fh.write(f"| {s} | {g} | {b} | {g+b} |\n")
        g = cnt("holdout_supermarket","good")
        fh.write(f"| holdout_supermarket | {g} | 0 | {g} |\n")
        fh.write("\n## Fuentes\n\n")
        from collections import Counter
        origins = Counter(r["origin"] for r in ok_rows)
        for o, n in sorted(origins.items()):
            fh.write(f"- {o}: {n} imagenes\n")
        fh.write(f"\n## Errores\n\n- Total errores: {len(err_rows)}\n")
        if err_rows:
            for e in err_rows[:10]:
                fh.write(f"  - {e['stem']} ({e['origin']})\n")
        fh.write("\n## Metodo de masking\n\n")
        fh.write("- Fruits-360 (F360_*): umbral blanco (RGB > 235) + fondo gris 128\n")
        fh.write("- Supermarket (1000060*): gray_bg_clean reutilizado de quality_roi_masked_previews_v2\n")
        fh.write("\n## Notas\n\n")
        fh.write("- No se modifico V2.\n")
        fh.write("- No se modifico analyze_quality.py ni quality_rules.yaml.\n")
        fh.write("- Batch_v3 excluido de train/val/test.\n")
    print(f"  Saved: {report_path.name}")

    print()
    print("=== build DONE ===")


if __name__ == "__main__":
    main()
