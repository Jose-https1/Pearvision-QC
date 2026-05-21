"""
create_fruits360_test_errors_grid_v1.py

Carga best_model.pt, re-infiere el test set completo y genera:
  - test_errors_grid.jpg      -- grid visual de todos los errores
  - test_predictions_all.csv  -- predicciones completas del test set
  - test_errors_detail.csv    -- solo las imagenes mal clasificadas
  - report.md                 -- resumen del analisis de errores

No entrena ningun modelo.
No modifica analyze_quality.py ni quality_rules.yaml.
"""

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.nn as nn

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"  / "quality_fruits360_human_v1"
MODEL_PATH   = PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v1" / "best_model.pt"
OUT_DIR      = PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v1"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_test_errors_grid_v1_report.md"

# ── Config (debe coincidir con el entrenamiento) ───────────────────────────────
IMG_SIZE = 224
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(path: Path, n_classes: int, device: str):
    model = models.mobilenet_v3_small(weights=None)
    in_feats = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feats, n_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(model, data_dir: Path, device: str):
    """Infiere el test set completo. Retorna lista de dicts por imagen."""
    test_ds = datasets.ImageFolder(str(data_dir / "test"), transform=eval_tf)
    # bad=0, good=1  (orden alfabetico)
    class_to_idx = test_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    results = []
    with torch.no_grad():
        for i, (img_tensor, true_label) in enumerate(test_ds):
            img_path  = Path(test_ds.samples[i][0])
            inp       = img_tensor.unsqueeze(0).to(device)
            logits    = model(inp)
            probs     = F.softmax(logits, dim=1).squeeze()
            pred_idx  = probs.argmax().item()
            conf      = probs[pred_idx].item()
            true_name = idx_to_class[true_label]
            pred_name = idx_to_class[pred_idx]
            correct   = (pred_idx == true_label)

            results.append({
                "img_path":    img_path,
                "filename":    img_path.name,
                "true_label":  true_name,
                "pred_label":  pred_name,
                "confidence":  conf,
                "prob_bad":    probs[class_to_idx["bad"]].item(),
                "prob_good":   probs[class_to_idx["good"]].item(),
                "correct":     correct,
                "error_type":  "" if correct else f"{true_name}->{pred_name}",
            })

    return results, idx_to_class


def make_errors_grid(errors: list, out_path: Path):
    """Grid visual: una celda por imagen mal clasificada."""
    n = len(errors)
    if n == 0:
        print("  No hay errores que visualizar.")
        return

    # Layout: hasta 4 columnas
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.8),
                             squeeze=False)
    fig.suptitle(
        f"Errores del clasificador en TEST  (total={n})\n"
        f"bad->good = FP (clasificada como buena siendo mala)   "
        f"good->bad = FN (clasificada como mala siendo buena)",
        fontsize=9, y=1.01
    )

    for idx, err in enumerate(errors):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        try:
            img = mpimg.imread(str(err["img_path"]))
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "no img", transform=ax.transAxes,
                    ha="center", va="center")

        error_type = err["error_type"]
        color = "#d62728" if error_type == "bad->good" else "#ff7f0e"

        title = (
            f"{err['filename']}\n"
            f"Real: {err['true_label']}  |  Pred: {err['pred_label']}\n"
            f"Conf: {err['confidence']:.2f}  [{error_type}]"
        )
        ax.set_title(title, fontsize=7, color=color, pad=3)
        ax.axis("off")

        # Borde de color segun tipo de error
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # Ocultar celdas sobrantes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Grid guardado: {out_path.name}")


def make_all_predictions_grid(results: list, out_path: Path):
    """Grid de TODAS las imagenes del test, con borde verde=OK rojo=error."""
    n     = len(results)
    ncols = min(7, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.4, nrows * 2.8),
                             squeeze=False)
    fig.suptitle("Test set completo — verde=correcto  rojo=error", fontsize=9)

    for idx, res in enumerate(results):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        try:
            img = mpimg.imread(str(res["img_path"]))
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#cccccc")

        color = "#2ca02c" if res["correct"] else "#d62728"
        ax.set_title(
            f"{res['true_label'][:3]}/{res['pred_label'][:3]}\n{res['confidence']:.2f}",
            fontsize=6, color=color, pad=2
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Grid completo guardado: {out_path.name}")


def save_csvs(results: list, errors: list, out_dir: Path):
    # Todas las predicciones
    all_path = out_dir / "test_predictions_all.csv"
    fields = ["filename", "true_label", "pred_label", "confidence",
              "prob_bad", "prob_good", "correct", "error_type"]
    with all_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in fields}
            row["prob_bad"]    = f"{r['prob_bad']:.4f}"
            row["prob_good"]   = f"{r['prob_good']:.4f}"
            row["confidence"]  = f"{r['confidence']:.4f}"
            w.writerow(row)
    print(f"  CSV predicciones: {all_path.name}")

    # Solo errores
    err_path = out_dir / "test_errors_detail.csv"
    with err_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in errors:
            row = {k: r[k] for k in fields}
            row["prob_bad"]    = f"{r['prob_bad']:.4f}"
            row["prob_good"]   = f"{r['prob_good']:.4f}"
            row["confidence"]  = f"{r['confidence']:.4f}"
            w.writerow(row)
    print(f"  CSV errores: {err_path.name}")


def write_report(results: list, errors: list):
    n_total   = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_errors  = len(errors)
    acc       = n_correct / n_total if n_total else 0

    fp_errors = [e for e in errors if e["error_type"] == "bad->good"]
    fn_errors = [e for e in errors if e["error_type"] == "good->bad"]

    lines = [
        "# Fruits-360 Test Errors Grid v1",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Resumen TEST",
        "",
        f"| Metrica | Valor |",
        f"|---------|-------|",
        f"| Total imagenes test | {n_total} |",
        f"| Correctas | {n_correct} |",
        f"| Errores | {n_errors} |",
        f"| Accuracy | {acc:.4f} |",
        f"| FP (bad pred como good) | {len(fp_errors)} |",
        f"| FN (good pred como bad) | {len(fn_errors)} |",
        "",
        "## Errores FP — bad clasificada como good",
        "",
        "| Filename | Conf | Prob_bad | Prob_good |",
        "|----------|------|----------|-----------|",
    ]
    for e in sorted(fp_errors, key=lambda x: x["confidence"], reverse=True):
        lines.append(f"| {e['filename']} | {e['confidence']:.3f} | "
                     f"{e['prob_bad']:.3f} | {e['prob_good']:.3f} |")

    lines += [
        "",
        "## Errores FN — good clasificada como bad",
        "",
        "| Filename | Conf | Prob_bad | Prob_good |",
        "|----------|------|----------|-----------|",
    ]
    for e in sorted(fn_errors, key=lambda x: x["confidence"], reverse=True):
        lines.append(f"| {e['filename']} | {e['confidence']:.3f} | "
                     f"{e['prob_bad']:.3f} | {e['prob_good']:.3f} |")

    lines += [
        "",
        "## Archivos generados",
        "",
        "- `outputs/fruits360_quality_cls_v1/test_errors_grid.jpg`",
        "- `outputs/fruits360_quality_cls_v1/test_all_grid.jpg`",
        "- `outputs/fruits360_quality_cls_v1/test_predictions_all.csv`",
        "- `outputs/fruits360_quality_cls_v1/test_errors_detail.csv`",
        "",
        "## Confirmaciones",
        "",
        "- NO se entrenó ningun modelo.",
        "- analyze_quality.py NO fue modificado.",
        "- quality_rules.yaml NO fue modificado.",
        "- Se uso el modelo best_model.pt ya existente.",
    ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD.name}")


def main():
    print("=== create_fruits360_test_errors_grid_v1 ===")
    print(f"  Device: {DEVICE}")

    if not MODEL_PATH.exists():
        print(f"ERROR: modelo no encontrado: {MODEL_PATH}")
        return

    print(f"  Cargando modelo: {MODEL_PATH.name}")
    model = load_model(MODEL_PATH, N_CLASSES, DEVICE)

    print("  Infiriendo test set...")
    results, idx_to_class = run_inference(model, DATA_DIR, DEVICE)

    errors    = [r for r in results if not r["correct"]]
    fp_errors = [e for e in errors if e["error_type"] == "bad->good"]
    fn_errors = [e for e in errors if e["error_type"] == "good->bad"]

    n_total   = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    acc       = n_correct / n_total

    print(f"  Test: {n_total} imgs  |  correctas={n_correct}  errores={len(errors)}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  FP (bad->good): {len(fp_errors)}")
    print(f"  FN (good->bad): {len(fn_errors)}")

    # Grids
    print("\n  Generando visualizaciones...")
    make_errors_grid(errors, OUT_DIR / "test_errors_grid.jpg")
    make_all_predictions_grid(results, OUT_DIR / "test_all_grid.jpg")

    # CSVs
    save_csvs(results, errors, OUT_DIR)

    # Reporte
    write_report(results, errors)

    # Detalle consola de errores
    print()
    print("  ERRORES FP — bad clasificada como good:")
    for e in sorted(fp_errors, key=lambda x: x["confidence"], reverse=True):
        print(f"    {e['filename']:<20} conf={e['confidence']:.3f}  "
              f"p_bad={e['prob_bad']:.3f}  p_good={e['prob_good']:.3f}")
    print()
    print("  ERRORES FN — good clasificada como bad:")
    for e in sorted(fn_errors, key=lambda x: x["confidence"], reverse=True):
        print(f"    {e['filename']:<20} conf={e['confidence']:.3f}  "
              f"p_bad={e['prob_bad']:.3f}  p_good={e['prob_good']:.3f}")
    print()
    print("  analyze_quality.py NO modificado.")
    print("  quality_rules.yaml  NO modificado.")


if __name__ == "__main__":
    main()
