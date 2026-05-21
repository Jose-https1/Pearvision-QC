"""
predict_fruits360_quality_v2_on_folder.py

Inferencia del clasificador binario V2 (good/bad) sobre una carpeta de imagenes nuevas.
Modelo: outputs/fruits360_quality_cls_v2/best_model.pt  (MobileNetV3-small, 2 clases)

Uso:
    python scripts/predict_fruits360_quality_v2_on_folder.py
    python scripts/predict_fruits360_quality_v2_on_folder.py --input <carpeta> --output <carpeta>

No modifica el modelo ni el dataset V2.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

# Forzar UTF-8 en stdout para evitar UnicodeEncodeError en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Rutas por defecto ──────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "data"    / "unseen_quality_eval_input"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "fruits360_quality_unseen_eval_example"
MODEL_PATH     = PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v2" / "best_model.pt"

# ── Transforms (identicos a eval_tf del entrenamiento) ─────────────────────────
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Clases en el mismo orden que ImageFolder (alfabetico): bad=0, good=1
CLASS_NAMES = ["bad", "good"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Constantes de contact sheet ────────────────────────────────────────────────
THUMB  = 160
COLS   = 6
PAD    = 6
TEXT_H = 36
BG     = (30, 30, 30)
COLORS = {"good": (80, 200, 80), "bad": (220, 60, 60)}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
    model = models.mobilenet_v3_small(weights=None)
    in_feats = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feats, len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model


def predict_image(model, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    tensor = EVAL_TF(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx   = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    prob_bad   = float(probs[0])
    prob_good  = float(probs[1])
    return pred_label, confidence, prob_bad, prob_good


def make_thumb(img_path: Path, label: str, conf: float) -> Image.Image:
    img  = Image.open(img_path).convert("RGB").resize((THUMB, THUMB))
    cell = Image.new("RGB", (THUMB, THUMB + TEXT_H), BG)
    cell.paste(img, (0, 0))
    draw  = ImageDraw.Draw(cell)
    color = COLORS.get(label, (200, 200, 200))
    text  = f"{img_path.name[:18]}\n{label.upper()} {conf:.2f}"
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    draw.text((2, THUMB + 2), text, fill=color, font=font)
    return cell


def make_contact_sheet(images_info: list, out_path: Path, title: str):
    if not images_info:
        placeholder = Image.new("RGB", (THUMB, THUMB + TEXT_H), BG)
        draw = ImageDraw.Draw(placeholder)
        draw.text((4, 4), "Sin imagenes", fill=(180, 180, 180))
        cells = [placeholder]
    else:
        cells = [
            make_thumb(Path(info["image_path"]), info["pred_label"], info["confidence"])
            for info in images_info
        ]

    n      = len(cells)
    cols   = min(COLS, n)
    rows   = (n + cols - 1) // cols
    cell_w = THUMB + PAD
    cell_h = THUMB + TEXT_H + PAD
    header_h = 28
    W = cols * cell_w + PAD
    H = rows * cell_h + PAD + header_h

    sheet = Image.new("RGB", (W, H), BG)
    draw  = ImageDraw.Draw(sheet)
    try:
        font_title = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
    draw.text((PAD, 6), title, fill=(220, 220, 220), font=font_title)

    for idx, cell in enumerate(cells):
        row = idx // cols
        col = idx % cols
        x = PAD + col * cell_w
        y = header_h + PAD + row * cell_h
        sheet.paste(cell, (x, y))

    sheet.save(out_path)


def write_summary(out_dir: Path, results: list, total_files: int, elapsed_s: float):
    n_good = sum(1 for r in results if r["pred_label"] == "good")
    n_bad  = sum(1 for r in results if r["pred_label"] == "bad")
    lines  = [
        "=== Fruits-360 Quality V2 --- Prediccion sobre carpeta nueva ===",
        "",
        f"Fecha:               {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Modelo:              {MODEL_PATH}",
        f"Imagenes procesadas: {len(results)} / {total_files} encontradas",
        f"Predichas GOOD:      {n_good}",
        f"Predichas BAD:       {n_bad}",
        f"Tiempo:              {elapsed_s:.2f} s",
        "",
    ]
    if len(results) == 0:
        lines += [
            "AVISO: No se encontraron imagenes (.jpg/.jpeg/.png) en la carpeta de entrada.",
            "Copia imagenes en data/unseen_quality_eval_input/ y vuelve a ejecutar.",
        ]
    else:
        lines += [
            "Archivos generados:",
            "  predictions.csv",
            "  human_review_template.csv",
            "  contact_sheet_all.jpg",
            "  contact_sheet_pred_good.jpg",
            "  contact_sheet_pred_bad.jpg",
            "  summary.txt",
            "",
            "Siguiente paso:",
            "  Revisa human_review_template.csv, rellena la columna human_label",
            "  y anota errores para decidir si construir V3.",
        ]
    lines += [
        "",
        "CONFIRMACIONES:",
        "  NO se modifico el dataset V2.",
        "  NO se modifico el modelo V2.",
        "  NO se modifico analyze_quality.py.",
        "  NO se modifico quality_rules.yaml.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prediccion V2 sobre carpeta de imagenes nuevas")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT,  help="Carpeta con imagenes de entrada")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Carpeta de salida")
    args = parser.parse_args()

    input_dir  = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    EXTS = {".jpg", ".jpeg", ".png"}
    img_paths   = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in EXTS])
    total_files = len(img_paths)

    print("=== predict_fruits360_quality_v2_on_folder ===")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Imagenes encontradas: {total_files}")

    if total_files == 0:
        print("  AVISO: carpeta vacia. Generando summary.txt.")
        write_summary(output_dir, [], 0, 0.0)
        print(f"  -> {output_dir / 'summary.txt'}")
        return

    print(f"  Cargando modelo desde {MODEL_PATH} ...")
    model = load_model()
    print(f"  Modelo cargado (device={DEVICE})")

    t0      = time.time()
    results = []
    for img_path in img_paths:
        try:
            pred_label, confidence, prob_bad, prob_good = predict_image(model, img_path)
            results.append({
                "image":      img_path.name,
                "image_path": str(img_path),
                "pred_label": pred_label,
                "confidence": round(confidence, 4),
                "prob_good":  round(prob_good, 4),
                "prob_bad":   round(prob_bad, 4),
            })
            print(f"  {img_path.name:<30}  {pred_label.upper():<6}  conf={confidence:.4f}")
        except Exception as e:
            print(f"  ERROR procesando {img_path.name}: {e}")
    elapsed = time.time() - t0

    # predictions.csv
    pred_csv = output_dir / "predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","image_path","pred_label","confidence","prob_good","prob_bad"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  -> {pred_csv.name}")

    # human_review_template.csv
    review_csv = output_dir / "human_review_template.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","image_path","pred_label","confidence","human_label","notes"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image":       r["image"],
                "image_path":  r["image_path"],
                "pred_label":  r["pred_label"],
                "confidence":  r["confidence"],
                "human_label": "",
                "notes":       "",
            })
    print(f"  -> {review_csv.name}")

    # Contact sheets
    cs_all  = output_dir / "contact_sheet_all.jpg"
    cs_good = output_dir / "contact_sheet_pred_good.jpg"
    cs_bad  = output_dir / "contact_sheet_pred_bad.jpg"

    make_contact_sheet(results, cs_all, f"ALL ({len(results)} imagenes)")
    make_contact_sheet([r for r in results if r["pred_label"] == "good"], cs_good, "PRED=GOOD")
    make_contact_sheet([r for r in results if r["pred_label"] == "bad"],  cs_bad,  "PRED=BAD")
    print(f"  -> {cs_all.name}")
    print(f"  -> {cs_good.name}")
    print(f"  -> {cs_bad.name}")

    write_summary(output_dir, results, total_files, elapsed)
    print("  -> summary.txt")

    n_good = sum(1 for r in results if r["pred_label"] == "good")
    n_bad  = sum(1 for r in results if r["pred_label"] == "bad")
    print(f"\n  GOOD={n_good}  BAD={n_bad}  ({elapsed:.2f}s)")
    print("  NO se modifico el dataset V2 ni el modelo.")


if __name__ == "__main__":
    main()
