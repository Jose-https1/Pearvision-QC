"""
predict_valid_capture_conditions_v3.py

Evaluacion del clasificador V2 (good/bad) sobre el lote de condiciones validas de captura.
Fondo blanco/gris/beige, iluminacion controlada, peras sanas de supermercado.

- NO entrena ningun modelo.
- NO modifica el dataset V2.
- NO modifica quality_rules.yaml ni analyze_quality.py.
- NO modifica best_model.pt.

Entrada:  data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3/
Salida:   outputs/supermarket_valid_conditions_batch_v3_eval/
Reporte:  reports/supermarket_valid_conditions_batch_v3_eval_report.md
"""

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Rutas ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR    = (PROJECT_ROOT / "data" / "unseen_quality_eval_input"
                / "supermarket_valid_conditions_batch_v3")
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "supermarket_valid_conditions_batch_v3_eval"
MODEL_PATH   = PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v2" / "best_model.pt"
REPORT_PATH  = PROJECT_ROOT / "reports" / "supermarket_valid_conditions_batch_v3_eval_report.md"

# ── Transforms (identicos al entrenamiento V2) ─────────────────────────────────
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

CLASS_NAMES = ["bad", "good"]   # orden alfabetico de ImageFolder
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Contact sheet ──────────────────────────────────────────────────────────────
THUMB  = 180
COLS   = 5
PAD    = 8
TEXT_H = 40
BG     = (25, 25, 25)
COLORS = {"good": (80, 210, 80), "bad": (220, 55, 55)}


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
    img    = Image.open(img_path).convert("RGB")
    tensor = EVAL_TF(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx   = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, float(probs[1]), float(probs[0])  # label, conf, prob_good, prob_bad


def make_thumb(img_path: Path, label: str, conf: float) -> Image.Image:
    img  = Image.open(img_path).convert("RGB").resize((THUMB, THUMB))
    cell = Image.new("RGB", (THUMB, THUMB + TEXT_H), BG)
    cell.paste(img, (0, 0))
    draw  = ImageDraw.Draw(cell)
    color = COLORS.get(label, (200, 200, 200))
    name  = img_path.name if len(img_path.name) <= 20 else img_path.name[:18] + ".."
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    draw.text((3, THUMB + 3), f"{name}\n{label.upper()}  {conf:.2f}", fill=color, font=font)
    return cell


def make_contact_sheet(rows: list, out_path: Path, title: str):
    if not rows:
        ph   = Image.new("RGB", (THUMB, THUMB + TEXT_H), BG)
        draw = ImageDraw.Draw(ph)
        draw.text((4, 4), "Sin imagenes", fill=(160, 160, 160))
        cells = [ph]
    else:
        cells = [make_thumb(Path(r["source_path"]), r["pred_label"], r["pred_confidence"])
                 for r in rows]
    n      = len(cells)
    cols   = min(COLS, n)
    nrows  = (n + cols - 1) // cols
    cell_w = THUMB + PAD
    cell_h = THUMB + TEXT_H + PAD
    hdr_h  = 32
    W = cols * cell_w + PAD
    H = nrows * cell_h + PAD + hdr_h
    sheet = Image.new("RGB", (W, H), BG)
    draw  = ImageDraw.Draw(sheet)
    try:
        font_title = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
    draw.text((PAD, 8), title, fill=(220, 220, 220), font=font_title)
    for idx, cell in enumerate(cells):
        r = idx // cols
        c = idx % cols
        sheet.paste(cell, (PAD + c * cell_w, hdr_h + PAD + r * cell_h))
    sheet.save(out_path, quality=92)


def write_predictions_csv(rows: list):
    path   = OUTPUT_DIR / "predictions.csv"
    fields = ["image_id", "filename", "source_path", "pred_label",
              "pred_confidence", "prob_good", "prob_bad"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def write_human_review_csv(rows: list):
    path   = OUTPUT_DIR / "human_review_template.csv"
    fields = ["image_id", "filename", "source_path", "pred_label",
              "pred_confidence", "human_label", "human_notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "image_id":        r["image_id"],
                "filename":        r["filename"],
                "source_path":     r["source_path"],
                "pred_label":      r["pred_label"],
                "pred_confidence": r["pred_confidence"],
                "human_label":     "",
                "human_notes":     "",
            })
    return path


def write_summary(rows: list, elapsed: float):
    n_good  = sum(1 for r in rows if r["pred_label"] == "good")
    n_bad   = sum(1 for r in rows if r["pred_label"] == "bad")
    conf_g  = [r["pred_confidence"] for r in rows if r["pred_label"] == "good"]
    conf_b  = [r["pred_confidence"] for r in rows if r["pred_label"] == "bad"]
    mean_g  = f"{sum(conf_g)/len(conf_g):.4f}" if conf_g else "N/A"
    mean_b  = f"{sum(conf_b)/len(conf_b):.4f}" if conf_b else "N/A"
    lines = [
        "=== Valid Capture Conditions Batch V3 — Evaluacion ===",
        "",
        f"Fecha:                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Modelo:                   {MODEL_PATH}",
        f"Imagenes procesadas:      {len(rows)}",
        f"Predichas GOOD:           {n_good}",
        f"Predichas BAD:            {n_bad}",
        f"Confianza media GOOD:     {mean_g}",
        f"Confianza media BAD:      {mean_b}",
        f"Tiempo:                   {elapsed:.2f} s",
    ]
    if n_bad:
        lines += ["", "Imágenes predichas BAD:"]
        for r in rows:
            if r["pred_label"] == "bad":
                lines.append(f"  - {r['filename']}  (conf={r['pred_confidence']:.4f})")
    lines += [
        "",
        "CONFIRMACIONES:",
        "  NO se entrenó ningun modelo.",
        "  NO se modifico el dataset V2.",
        "  NO se modificaron quality_rules.yaml ni analyze_quality.py.",
        "  NO se modifico ningun dataset anterior.",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_report(rows: list, elapsed: float):
    n_good  = sum(1 for r in rows if r["pred_label"] == "good")
    n_bad   = sum(1 for r in rows if r["pred_label"] == "bad")
    conf_g  = [r["pred_confidence"] for r in rows if r["pred_label"] == "good"]
    conf_b  = [r["pred_confidence"] for r in rows if r["pred_label"] == "bad"]
    mean_g  = f"{sum(conf_g)/len(conf_g):.4f}" if conf_g else "N/A"
    mean_b  = f"{sum(conf_b)/len(conf_b):.4f}" if conf_b else "N/A"
    acc_pct = f"{n_good / len(rows) * 100:.1f}%" if rows else "N/A"

    pred_table = "\n".join(
        f"| {r['image_id']} | {r['filename']} | {r['pred_label'].upper()} "
        f"| {r['pred_confidence']:.4f} | {r['prob_good']:.4f} | {r['prob_bad']:.4f} |"
        for r in rows
    )

    if n_bad == 0:
        bad_section = "\n**Sin imágenes predichas como BAD en este lote.**\n"
        interpretation = (
            "**V2 clasifica correctamente el 100% de las peras sanas bajo "
            "condiciones válidas de captura.** Esto confirma que el problema detectado "
            "en el lote V2 era el fondo (azul/negro), no la calidad de la fruta. "
            "V2 es válido como baseline para la demo con fondo blanco/gris/beige controlado."
        )
    else:
        bad_list = "\n".join(
            f"| `{r['filename']}` | {r['pred_confidence']:.4f} |"
            for r in rows if r["pred_label"] == "bad"
        )
        bad_section = f"""
**Imágenes predichas BAD — requieren revisión humana:**

| filename | pred_confidence |
|---|---|
{bad_list}
"""
        interpretation = (
            f"**V2 clasifica correctamente {n_good}/{len(rows)} peras ({acc_pct}) "
            "bajo condiciones válidas de captura.** "
            f"Los {n_bad} caso(s) BAD deben revisarse visualmente. "
            "Si son peras sanas, guardarlos como hard examples GOOD para V3. "
            "Si tienen defecto real, son ejemplos BAD válidos."
        )

    content = f"""# Valid Capture Conditions Batch V3 — Evaluation Report

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Lote:** `supermarket_valid_conditions_batch_v3`
**Condiciones:** fondo blanco/gris/beige mate, iluminación controlada
**Modelo:** `outputs/fruits360_quality_cls_v2/best_model.pt` (V2, congelado)
**Entrada:** `data/unseen_quality_eval_input/supermarket_valid_conditions_batch_v3/`
({len(rows)} imágenes)

---

## Resumen

| Métrica | Valor |
|---|---|
| Imágenes procesadas | {len(rows)} |
| Predichas GOOD | {n_good} |
| Predichas BAD | {n_bad} |
| Tasa de acierto (asumiendo todas GOOD) | {acc_pct} |
| Confianza media GOOD | {mean_g} |
| Confianza media BAD | {mean_b} |
| Tiempo de inferencia | {elapsed:.2f} s |

---

## Interpretación técnica

{interpretation}

---
{bad_section}

---

## Contexto y motivación

Este lote evalúa V2 específicamente bajo las condiciones recomendadas tras el análisis
del lote V2, donde se detectó que el modelo fallaba principalmente en fondos azul y negro:

| Lote | Total | GOOD | BAD | Tasa error |
|---|---|---|---|---|
| batch_v1 (fondo variado)      | 20 | 19 | 1  | 5%  |
| batch_v2 (fondos mixtos)      | 22 | 15 | 7  | 32% |
| **batch_v3 (fondo válido)**   | **{len(rows)}** | **{n_good}** | **{n_bad}** | **{100 - int(n_good/len(rows)*100)}%** |

---

## Predicciones completas

| image_id | filename | pred_label | pred_confidence | prob_good | prob_bad |
|---|---|---|---|---|---|
{pred_table}

---

## Archivos generados

- `outputs/supermarket_valid_conditions_batch_v3_eval/predictions.csv`
- `outputs/supermarket_valid_conditions_batch_v3_eval/human_review_template.csv`
- `outputs/supermarket_valid_conditions_batch_v3_eval/contact_sheet_all.jpg`
- `outputs/supermarket_valid_conditions_batch_v3_eval/contact_sheet_pred_good.jpg`
- `outputs/supermarket_valid_conditions_batch_v3_eval/contact_sheet_pred_bad.jpg`
- `outputs/supermarket_valid_conditions_batch_v3_eval/summary.txt`

---

## Confirmaciones

- **NO** se entrenó ningún modelo.
- **NO** se modificó el dataset V2 (`data/quality_fruits360_human_v2/`).
- **NO** se modificó `best_model.pt`.
- **NO** se modificó `analyze_quality.py`.
- **NO** se modificó `quality_rules.yaml`.
- **NO** se modificó ningún dataset anterior.

---

## Siguiente paso

José debe abrir `outputs/supermarket_valid_conditions_batch_v3_eval/contact_sheet_all.jpg`
y confirmar si todas las imágenes son GOOD o si hay falsos BAD.
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_paths = sorted([p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in EXTS])

    print("=== predict_valid_capture_conditions_v3 ===")
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Imagenes encontradas: {len(img_paths)}")

    if len(img_paths) == 0:
        print("  AVISO: carpeta vacia. Copia imágenes de condiciones válidas y vuelve a ejecutar.")
        return

    print(f"  Cargando modelo (device={DEVICE}) ...")
    model = load_model()
    print("  Modelo cargado.")

    t0   = time.time()
    rows = []
    for i, p in enumerate(img_paths, start=1):
        try:
            pred_label, confidence, prob_good, prob_bad = predict_image(model, p)
            rows.append({
                "image_id":        i,
                "filename":        p.name,
                "source_path":     str(p),
                "pred_label":      pred_label,
                "pred_confidence": round(confidence, 4),
                "prob_good":       round(prob_good, 4),
                "prob_bad":        round(prob_bad, 4),
            })
            print(f"  [{i:>3}] {p.name:<32}  {pred_label.upper():<5}  conf={confidence:.4f}")
        except Exception as e:
            print(f"  ERROR {p.name}: {e}")
    elapsed = time.time() - t0

    pred_csv   = write_predictions_csv(rows)
    review_csv = write_human_review_csv(rows)

    good_rows = [r for r in rows if r["pred_label"] == "good"]
    bad_rows  = [r for r in rows if r["pred_label"] == "bad"]

    make_contact_sheet(rows,      OUTPUT_DIR / "contact_sheet_all.jpg",
                       f"Valid Conditions V3 — ALL ({len(rows)} imgs)")
    make_contact_sheet(good_rows, OUTPUT_DIR / "contact_sheet_pred_good.jpg",
                       f"PRED=GOOD ({len(good_rows)})")
    make_contact_sheet(bad_rows,  OUTPUT_DIR / "contact_sheet_pred_bad.jpg",
                       f"PRED=BAD ({len(bad_rows)})")

    write_summary(rows, elapsed)
    write_report(rows, elapsed)

    n_good  = len(good_rows)
    n_bad   = len(bad_rows)
    conf_g  = [r["pred_confidence"] for r in good_rows]
    conf_b  = [r["pred_confidence"] for r in bad_rows]
    mean_g  = f"{sum(conf_g)/len(conf_g):.4f}" if conf_g else "N/A"
    mean_b  = f"{sum(conf_b)/len(conf_b):.4f}" if conf_b else "N/A"

    print()
    print("=" * 58)
    print("EVALUACIÓN VALID CAPTURE CONDITIONS V3 COMPLETADA")
    print()
    print(f"Imágenes procesadas: {len(rows)}")
    print(f"Predichas GOOD: {n_good}  (confianza media: {mean_g})")
    print(f"Predichas BAD:  {n_bad}  (confianza media: {mean_b})")
    if bad_rows:
        print()
        print("Imágenes predichas BAD:")
        for r in bad_rows:
            print(f"  - {r['filename']}  (conf={r['pred_confidence']:.4f})")
    print()
    print("Archivos principales:")
    print(f"- {OUTPUT_DIR / 'contact_sheet_all.jpg'}")
    print(f"- {OUTPUT_DIR / 'contact_sheet_pred_good.jpg'}")
    print(f"- {OUTPUT_DIR / 'contact_sheet_pred_bad.jpg'}")
    print(f"- {pred_csv}")
    print(f"- {review_csv}")
    print(f"- {REPORT_PATH}")
    print()
    print("NO se entrenó ningún modelo.")
    print("NO se modificó V2.")
    print("NO se modificó analyze_quality.py.")
    print("NO se modificó quality_rules.yaml.")
    print()
    print("Siguiente paso:")
    print("José debe abrir contact_sheet_all.jpg y confirmar si todas las imágenes")
    print("son GOOD o si hay falsos BAD.")


if __name__ == "__main__":
    main()
