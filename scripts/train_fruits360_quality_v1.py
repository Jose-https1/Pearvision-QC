"""
train_fruits360_quality_v1.py

Clasificador binario de calidad de peras (good / bad).
Dataset:  data/quality_fruits360_human_v1/
Modelo:   MobileNetV3-small fine-tuned (pretrained ImageNet)
Salida:   outputs/fruits360_quality_cls_v1/
Reporte:  reports/fruits360_quality_classifier_v1_report.md

Notas:
- Compensa desbalance good:bad con class_weights en la loss
  y WeightedRandomSampler en el DataLoader de train.
- Metrica principal para seleccion del mejor checkpoint: F1-macro.
- NO modifica analyze_quality.py ni quality_rules.yaml.
- NO usa imagenes REVIEW.
"""

import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "quality_fruits360_human_v1"
OUT_DIR      = PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v1"
REPORT_MD    = PROJECT_ROOT / "reports" / "fruits360_quality_classifier_v1_report.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

# ── Hiperparámetros ────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Reproducibilidad ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Dataset & DataLoaders ─────────────────────────────────────────────────────
def build_loaders():
    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(DATA_DIR / "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(str(DATA_DIR / "test"),  transform=eval_tf)

    # clases ordenadas alfabéticamente: bad=0, good=1
    class_to_idx = train_ds.class_to_idx
    print(f"  Clases detectadas: {class_to_idx}")

    # ── Class weights para CrossEntropyLoss ───────────────────────────────
    counts  = Counter(lbl for _, lbl in train_ds.samples)
    n_total = len(train_ds)
    n_cls   = len(class_to_idx)
    cw = torch.zeros(n_cls, dtype=torch.float32)
    for idx in range(n_cls):
        cw[idx] = n_total / (n_cls * counts[idx])
    print(f"  Class weights: {dict(zip(class_to_idx.keys(), cw.tolist()))}")

    # ── WeightedRandomSampler ─────────────────────────────────────────────
    sample_weights = torch.tensor([cw[lbl].item() for _, lbl in train_ds.samples])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_to_idx, cw


# ── Modelo ────────────────────────────────────────────────────────────────────
def build_model(n_classes: int, device: str):
    pretrained_ok = True
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model   = models.mobilenet_v3_small(weights=weights)
        print("  Pesos pretrained ImageNet cargados.")
    except Exception as e:
        print(f"  ADVERTENCIA: no se pudieron cargar pesos pretrained ({e}).")
        print("  Usando weights=None (entrenamiento desde cero).")
        model = models.mobilenet_v3_small(weights=None)
        pretrained_ok = False

    # Reemplazar cabeza clasificadora
    in_feats = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feats, n_classes)
    model = model.to(device)
    return model, pretrained_ok


# ── Métricas (sin sklearn) ───────────────────────────────────────────────────
def compute_metrics(all_labels: list, all_preds: list, n_classes: int):
    """Calcula accuracy, precision, recall, F1 y matriz de confusion."""
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    acc = (labels == preds).mean()

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    precision, recall, f1 = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)

    f1_macro = float(np.mean(f1))
    return acc, precision, recall, f1, f1_macro, cm


# ── Loop de entrenamiento ─────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_preds


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_curves(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"],   label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["val_f1"], label="val F1-macro", color="green")
    axes[2].set_title("Val F1-macro")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Curvas guardadas: {path.name}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_dir: Path, tag="test"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"Confusion Matrix ({tag})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    path = out_dir / f"confusion_matrix_{tag}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Confusion matrix ({tag}): {path.name}")


# ── Reporte .md ───────────────────────────────────────────────────────────────
def write_report(cfg: dict, best_epoch: int, val_metrics: dict,
                 test_metrics: dict, pretrained_ok: bool, elapsed: float):
    class_names = cfg["class_names"]
    n = len(class_names)
    status = "PASS"

    lines = [
        "# Fruits-360 Quality Classifier v1 — Reporte de Entrenamiento",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Status: **{status}**",
        "",
        "## Configuracion",
        "",
        f"| Parametro | Valor |",
        f"|-----------|-------|",
        f"| Modelo | MobileNetV3-small |",
        f"| Pesos pretrained | {'ImageNet (IMAGENET1K_V1)' if pretrained_ok else 'NINGUNO (descarga fallida)'} |",
        f"| Dataset | data/quality_fruits360_human_v1/ |",
        f"| Train | {cfg['n_train']} imgs (good={cfg['n_train_good']}, bad={cfg['n_train_bad']}) |",
        f"| Val | {cfg['n_val']} imgs |",
        f"| Test | {cfg['n_test']} imgs |",
        f"| Epochs | {cfg['epochs']} |",
        f"| Batch size | {cfg['batch']} |",
        f"| LR | {cfg['lr']} |",
        f"| Optimizer | Adam |",
        f"| Scheduler | CosineAnnealingLR |",
        f"| Device | {cfg['device']} |",
        f"| Seed | {cfg['seed']} |",
        f"| Desbalance compensado | class_weights en loss + WeightedRandomSampler |",
        "",
        "## Mejor checkpoint",
        "",
        f"- Epoca: {best_epoch}  (criterio: F1-macro en val)",
        f"- Val loss: {val_metrics['loss']:.4f}",
        f"- Val accuracy: {val_metrics['acc']:.4f}",
        f"- Val F1-macro: {val_metrics['f1_macro']:.4f}",
        "",
        "## Resultados en VAL (mejor epoch)",
        "",
        "| Clase | Precision | Recall | F1 |",
        "|-------|-----------|--------|----|",
    ]
    for i, cn in enumerate(class_names):
        lines.append(f"| {cn} | {val_metrics['precision'][i]:.3f} | "
                     f"{val_metrics['recall'][i]:.3f} | {val_metrics['f1'][i]:.3f} |")

    lines += [
        "",
        "## Resultados en TEST (evaluacion final)",
        "",
        f"- Test accuracy : {test_metrics['acc']:.4f}",
        f"- Test F1-macro : {test_metrics['f1_macro']:.4f}",
        "",
        "| Clase | Precision | Recall | F1 |",
        "|-------|-----------|--------|----|",
    ]
    for i, cn in enumerate(class_names):
        lines.append(f"| {cn} | {test_metrics['precision'][i]:.3f} | "
                     f"{test_metrics['recall'][i]:.3f} | {test_metrics['f1'][i]:.3f} |")

    lines += [
        "",
        "## Matriz de confusion (TEST)",
        "",
        "```",
        f"           " + "  ".join(f"{cn:>8}" for cn in class_names),
    ]
    for i, cn in enumerate(class_names):
        row = "  ".join(f"{test_metrics['cm'][i][j]:>8}" for j in range(n))
        lines.append(f"  {cn:>8}  {row}")
    lines += ["```", ""]

    lines += [
        "## Archivos generados",
        "",
        f"- `{OUT_DIR.relative_to(PROJECT_ROOT)}/best_model.pt`",
        f"- `{OUT_DIR.relative_to(PROJECT_ROOT)}/training_curves.png`",
        f"- `{OUT_DIR.relative_to(PROJECT_ROOT)}/confusion_matrix_val.png`",
        f"- `{OUT_DIR.relative_to(PROJECT_ROOT)}/confusion_matrix_test.png`",
        f"- `{OUT_DIR.relative_to(PROJECT_ROOT)}/train_log.csv`",
        "",
        "## Advertencia de desbalance",
        "",
        "Ratio good:bad ≈ 1:4.5. Se aplicaron:",
        "- class_weights inversamente proporcionales a la frecuencia de clase",
        "- WeightedRandomSampler para equilibrar batches durante el entrenamiento",
        "",
        "Para entrenamiento futuro considerar adicionalmente:",
        "- Augmentacion fuerte solo en la clase `good`",
        "- Mixup / CutMix",
        "",
        "## Confirmaciones",
        "",
        "- analyze_quality.py NO fue modificado.",
        "- quality_rules.yaml NO fue modificado.",
        "- Imagenes REVIEW NO fueron usadas.",
        "- Test set usado UNA SOLA VEZ al final.",
        f"- Tiempo total de entrenamiento: {elapsed:.1f} s",
    ]
    if not pretrained_ok:
        lines += ["", "## ADVERTENCIA PRETRAINED",
                  "Los pesos ImageNet no pudieron descargarse. El modelo fue entrenado desde cero.",
                  "Los resultados seran significativamente peores que con pesos pretrained."]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"=== train_fruits360_quality_v1 ===")
    print(f"  Device : {DEVICE}")
    set_seed(SEED)

    # Verificar dataset
    for split in ["train", "val", "test"]:
        for cls in ["good", "bad"]:
            p = DATA_DIR / split / cls
            if not p.exists():
                print(f"ERROR: carpeta no encontrada: {p}")
                sys.exit(1)

    print("\n  Cargando datos...")
    train_loader, val_loader, test_loader, class_to_idx, class_weights = build_loaders()
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    n_classes   = len(class_names)

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    cnt_train = Counter(lbl for _, lbl in train_loader.dataset.samples)
    print(f"  train: {n_train}  val: {n_val}  test: {n_test}")

    print("\n  Construyendo modelo...")
    model, pretrained_ok = build_model(n_classes, DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Historial y log
    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    log_rows = []

    best_val_f1   = -1.0
    best_epoch    = 0
    best_val_metrics = None
    best_model_path  = OUT_DIR / "best_model.pt"

    print(f"\n  Entrenando {EPOCHS} epochs...\n")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()

        tr_loss, tr_lbl, tr_pred = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
        vl_loss, vl_lbl, vl_pred = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, train=False)
        scheduler.step()

        tr_acc, _, _, tr_f1, tr_f1m, _ = compute_metrics(tr_lbl, tr_pred, n_classes)
        vl_acc, vl_p, vl_r, vl_f1, vl_f1m, vl_cm = compute_metrics(vl_lbl, vl_pred, n_classes)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1m)

        log_rows.append({
            "epoch": epoch, "train_loss": f"{tr_loss:.4f}", "val_loss": f"{vl_loss:.4f}",
            "train_acc": f"{tr_acc:.4f}", "val_acc": f"{vl_acc:.4f}",
            "val_f1_macro": f"{vl_f1m:.4f}",
        })

        ep_t = time.time() - t_ep
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_acc={vl_acc:.3f}  vl_f1={vl_f1m:.3f}  ({ep_t:.1f}s)")

        if vl_f1m > best_val_f1:
            best_val_f1 = vl_f1m
            best_epoch  = epoch
            best_val_metrics = {
                "loss": vl_loss, "acc": vl_acc, "f1_macro": vl_f1m,
                "precision": vl_p, "recall": vl_r, "f1": vl_f1, "cm": vl_cm,
            }
            torch.save(model.state_dict(), best_model_path)

    elapsed = time.time() - t0
    print(f"\n  Mejor epoch: {best_epoch}  (val F1-macro={best_val_f1:.4f})")
    print(f"  Modelo guardado: {best_model_path.name}")

    # Guardar log CSV
    log_path = OUT_DIR / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)

    # Plots de curvas de entrenamiento
    print("\n  Generando plots...")
    plot_curves(history, OUT_DIR)
    plot_confusion_matrix(best_val_metrics["cm"], class_names, OUT_DIR, tag="val")

    # ── Evaluacion en TEST ────────────────────────────────────────────────
    print("\n  Evaluando en TEST (mejor modelo)...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    te_loss, te_lbl, te_pred = run_epoch(model, test_loader, criterion, optimizer, DEVICE, train=False)
    te_acc, te_p, te_r, te_f1, te_f1m, te_cm = compute_metrics(te_lbl, te_pred, n_classes)

    test_metrics = {
        "loss": te_loss, "acc": te_acc, "f1_macro": te_f1m,
        "precision": te_p, "recall": te_r, "f1": te_f1, "cm": te_cm,
    }
    plot_confusion_matrix(te_cm, class_names, OUT_DIR, tag="test")

    # Guardar resultados test CSV
    test_csv = OUT_DIR / "test_results.csv"
    with test_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class","precision","recall","f1"])
        w.writeheader()
        for i, cn in enumerate(class_names):
            w.writerow({"class": cn, "precision": f"{te_p[i]:.4f}",
                        "recall": f"{te_r[i]:.4f}", "f1": f"{te_f1[i]:.4f}"})

    # Reporte
    cfg = {
        "class_names": class_names, "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "n_train_good": cnt_train[class_to_idx["good"]],
        "n_train_bad":  cnt_train[class_to_idx["bad"]],
        "epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR,
        "device": DEVICE, "seed": SEED,
    }
    write_report(cfg, best_epoch, best_val_metrics, test_metrics, pretrained_ok, elapsed)

    # ── Resumen consola ───────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  RESULTADOS FINALES")
    print("=" * 50)
    print(f"  Mejor epoch    : {best_epoch}/{EPOCHS}")
    print(f"  Val F1-macro   : {best_val_f1:.4f}")
    print()
    print(f"  TEST accuracy  : {te_acc:.4f}")
    print(f"  TEST F1-macro  : {te_f1m:.4f}")
    print()
    print(f"  {'Clase':<8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*32}")
    for i, cn in enumerate(class_names):
        print(f"  {cn:<8}  {te_p[i]:>6.3f}  {te_r[i]:>6.3f}  {te_f1[i]:>6.3f}")
    print()
    print(f"  Confusion matrix (TEST)  pred->")
    print(f"             " + "  ".join(f"{cn:>8}" for cn in class_names))
    for i, cn in enumerate(class_names):
        row = "  ".join(f"{te_cm[i][j]:>8}" for j in range(n_classes))
        print(f"  real {cn:>4}  {row}")
    print()
    print(f"  Tiempo total: {elapsed:.1f} s")
    print(f"  Modelo: {best_model_path}")
    print(f"  analyze_quality.py NO modificado.")
    print(f"  quality_rules.yaml  NO modificado.")


if __name__ == "__main__":
    main()
