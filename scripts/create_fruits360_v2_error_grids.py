"""
create_fruits360_v2_error_grids.py

Carga best_model.pt V2, re-infiere el test set completo y genera:
  - test_all_grid.jpg             -- todas las imagenes (verde=OK, rojo=error)
  - test_errors_grid.jpg          -- solo los errores
  - test_errors_detail.csv        -- per-error CSV
  - bad_as_good_errors_grid.jpg   -- FP: bad predicha como good
  - good_as_bad_errors_grid.jpg   -- FN: good predicha como bad

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
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.nn as nn

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"    / "quality_fruits360_human_v2"
MODEL_PATH   = PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v2" / "best_model.pt"
OUT_DIR      = PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v2"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_v2_error_grids_report.md"

# ── Config (debe coincidir con el entrenamiento V2) ────────────────────────────
IMG_SIZE  = 224
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
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
    test_ds      = datasets.ImageFolder(str(data_dir / "test"), transform=eval_tf)
    class_to_idx = test_ds.class_to_idx          # bad=0, good=1
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
                "img_path":   img_path,
                "filename":   img_path.name,
                "true_label": true_name,
                "pred_label": pred_name,
                "confidence": conf,
                "prob_bad":   probs[class_to_idx["bad"]].item(),
                "prob_good":  probs[class_to_idx["good"]].item(),
                "correct":    correct,
                "error_type": "" if correct else f"{true_name}->{pred_name}",
            })

    return results


def make_errors_grid(errors: list, out_path: Path, title: str = None):
    n = len(errors)
    if n == 0:
        print(f"  Sin errores para {out_path.name} — archivo no generado.")
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.8),
                             squeeze=False)

    default_title = (
        f"Errores clasificador V2 en TEST  (total={n})\n"
        f"bad->good = FP (predicha buena siendo mala)   "
        f"good->bad = FN (predicha mala siendo buena)"
    )
    fig.suptitle(title or default_title, fontsize=9, y=1.01)

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

        ax.set_title(
            f"{err['filename']}\n"
            f"Real: {err['true_label']}  |  Pred: {err['pred_label']}\n"
            f"Conf: {err['confidence']:.3f}  [{error_type}]",
            fontsize=7, color=color, pad=3
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
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {out_path.name}")


def make_all_grid(results: list, out_path: Path):
    n     = len(results)
    ncols = min(7, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.4, nrows * 2.8),
                             squeeze=False)
    fig.suptitle(
        f"Test set V2 completo ({n} imgs) — verde=correcto  rojo=error",
        fontsize=9
    )

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
    print(f"  Guardado: {out_path.name}")


def save_error_csv(errors: list, out_path: Path):
    fields = ["filename", "true_label", "pred_label", "confidence",
              "prob_bad", "prob_good", "error_type"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in errors:
            w.writerow({
                "filename":   r["filename"],
                "true_label": r["true_label"],
                "pred_label": r["pred_label"],
                "confidence": f"{r['confidence']:.4f}",
                "prob_bad":   f"{r['prob_bad']:.4f}",
                "prob_good":  f"{r['prob_good']:.4f}",
                "error_type": r["error_type"],
            })
    print(f"  Guardado: {out_path.name}")


def write_report(results: list, errors: list, fp_errors: list, fn_errors: list):
    n_total   = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    acc       = n_correct / n_total if n_total else 0

    lines = [
        "# Fruits-360 Quality V2 — Error Grids",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Resumen TEST V2",
        "",
        "| Metrica | Valor |",
        "|---------|-------|",
        f"| Total imagenes test | {n_total} |",
        f"| Correctas | {n_correct} |",
        f"| Errores totales | {len(errors)} |",
        f"| Accuracy | {acc:.4f} |",
        f"| FP bad->good | {len(fp_errors)} |",
        f"| FN good->bad | {len(fn_errors)} |",
        "",
        "## Errores FP (bad predicha como good)",
        "",
        "| Filename | Conf | Prob_bad | Prob_good |",
        "|----------|------|----------|-----------|",
    ]
    for e in sorted(fp_errors, key=lambda x: x["confidence"], reverse=True):
        lines.append(f"| {e['filename']} | {e['confidence']:.3f} | "
                     f"{e['prob_bad']:.3f} | {e['prob_good']:.3f} |")

    lines += [
        "",
        "## Errores FN (good predicha como bad)",
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
        "- `outputs/fruits360_quality_cls_v2/test_all_grid.jpg`",
        "- `outputs/fruits360_quality_cls_v2/test_errors_grid.jpg`",
        "- `outputs/fruits360_quality_cls_v2/test_errors_detail.csv`",
        "- `outputs/fruits360_quality_cls_v2/bad_as_good_errors_grid.jpg`",
        "- `outputs/fruits360_quality_cls_v2/good_as_bad_errors_grid.jpg`",
        "",
        "## Confirmaciones",
        "",
        "- NO se entrenó ningun modelo.",
        "- analyze_quality.py NO fue modificado.",
        "- quality_rules.yaml NO fue modificado.",
        "- Se uso el modelo best_model.pt V2 ya existente.",
    ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD.name}")


def main():
    print("=== create_fruits360_v2_error_grids ===")
    print(f"  Device: {DEVICE}")

    if not MODEL_PATH.exists():
        print(f"ERROR: modelo no encontrado: {MODEL_PATH}")
        return

    print(f"  Cargando modelo: {MODEL_PATH.name}")
    model = load_model(MODEL_PATH, N_CLASSES, DEVICE)

    print("  Infiriendo test set V2...")
    results = run_inference(model, DATA_DIR, DEVICE)

    errors    = [r for r in results if not r["correct"]]
    fp_errors = [e for e in errors if e["error_type"] == "bad->good"]
    fn_errors = [e for e in errors if e["error_type"] == "good->bad"]

    n_total   = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    acc       = n_correct / n_total

    print(f"  Test V2: {n_total} imgs  |  correctas={n_correct}  errores={len(errors)}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  FP (bad->good): {len(fp_errors)}")
    print(f"  FN (good->bad): {len(fn_errors)}")

    print("\n  Generando visualizaciones...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    make_all_grid(results, OUT_DIR / "test_all_grid.jpg")

    make_errors_grid(errors, OUT_DIR / "test_errors_grid.jpg")

    make_errors_grid(
        fp_errors,
        OUT_DIR / "bad_as_good_errors_grid.jpg",
        title=f"V2 FP: bad predicha como good  (n={len(fp_errors)})"
    )

    make_errors_grid(
        fn_errors,
        OUT_DIR / "good_as_bad_errors_grid.jpg",
        title=f"V2 FN: good predicha como bad  (n={len(fn_errors)})"
    )

    save_error_csv(errors, OUT_DIR / "test_errors_detail.csv")

    write_report(results, errors, fp_errors, fn_errors)

    print()
    print("  ERRORES FP — bad predicha como good:")
    for e in sorted(fp_errors, key=lambda x: x["confidence"], reverse=True):
        print(f"    {e['filename']:<20} conf={e['confidence']:.3f}  "
              f"p_bad={e['prob_bad']:.3f}  p_good={e['prob_good']:.3f}")

    print()
    print("  ERRORES FN — good predicha como bad:")
    for e in sorted(fn_errors, key=lambda x: x["confidence"], reverse=True):
        print(f"    {e['filename']:<20} conf={e['confidence']:.3f}  "
              f"p_bad={e['prob_bad']:.3f}  p_good={e['prob_good']:.3f}")

    print()
    print("  analyze_quality.py NO modificado.")
    print("  quality_rules.yaml  NO modificado.")
    print("  Ningun modelo entrenado.")


if __name__ == "__main__":
    main()
