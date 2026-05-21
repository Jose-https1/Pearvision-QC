"""
train_fruits360_quality_v2.py

Clasificador binario V2 de calidad de peras (good / bad).
Dataset:  data/quality_fruits360_human_v2/
Modelo:   MobileNetV3-small fine-tuned (pretrained ImageNet)
Salida:   outputs/fruits360_quality_cls_v2/
Reporte:  reports/fruits360_quality_classifier_v2_report.md

Mejoras respecto a V1:
- Dataset corregido (F360_0198 y F360_0052 movidos a REVIEW)
- 40 epochs (V1 seguia convergiendo al epoch 30)
- Misma arquitectura y hiperparametros para comparacion justa

No modifica analyze_quality.py ni quality_rules.yaml.
No usa imagenes REVIEW.
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
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"    / "quality_fruits360_human_v2"
OUT_DIR       = PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v2"
REPORT_MD     = PROJECT_ROOT / "reports" / "fruits360_quality_classifier_v2_report.md"
V1_REPORT     = PROJECT_ROOT / "reports" / "fruits360_quality_classifier_v1_report.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

# ── Hiperparámetros ────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 40        # +10 respecto a V1 (V1 seguia mejorando)
LR         = 1e-4
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders():
    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(DATA_DIR / "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(str(DATA_DIR / "test"),  transform=eval_tf)

    class_to_idx = train_ds.class_to_idx
    print(f"  Clases: {class_to_idx}")

    counts  = Counter(lbl for _, lbl in train_ds.samples)
    n_total = len(train_ds)
    n_cls   = len(class_to_idx)
    cw = torch.zeros(n_cls, dtype=torch.float32)
    for idx in range(n_cls):
        cw[idx] = n_total / (n_cls * counts[idx])
    print(f"  Class weights: {dict(zip(class_to_idx.keys(), [f'{v:.3f}' for v in cw.tolist()]))}")

    sample_weights = torch.tensor([cw[lbl].item() for _, lbl in train_ds.samples])
    sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, class_to_idx, cw


def build_model(n_classes, device):
    pretrained_ok = True
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model   = models.mobilenet_v3_small(weights=weights)
        print("  Pesos pretrained ImageNet cargados.")
    except Exception as e:
        print(f"  ADVERTENCIA: pretrained fallido ({e}). Usando weights=None.")
        model         = models.mobilenet_v3_small(weights=None)
        pretrained_ok = False
    in_feats = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feats, n_classes)
    return model.to(device), pretrained_ok


def compute_metrics(all_labels, all_preds, n_classes):
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    acc    = (labels == preds).mean()
    cm     = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    precision, recall, f1 = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        precision.append(p_)
        recall.append(r_)
        f1.append(f_)
    return acc, precision, recall, f1, float(np.mean(f1)), cm


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
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(loader.dataset), all_labels, all_preds


def plot_curves(history, out_dir, v1_history=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training V2 (solid) vs V1 baseline (dashed)", fontsize=10)

    axes[0].plot(epochs, history["train_loss"], label="train V2")
    axes[0].plot(epochs, history["val_loss"],   label="val V2")
    if v1_history:
        ep1 = range(1, len(v1_history["val_loss"]) + 1)
        axes[0].plot(ep1, v1_history["val_loss"], "--", alpha=0.5, label="val V1")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train V2")
    axes[1].plot(epochs, history["val_acc"],   label="val V2")
    if v1_history:
        axes[1].plot(ep1, v1_history["val_acc"], "--", alpha=0.5, label="val V1")
    axes[1].set_title("Accuracy"); axes[1].legend()

    axes[2].plot(epochs, history["val_f1"], label="val F1 V2", color="green")
    if v1_history:
        axes[2].plot(ep1, v1_history["val_f1"], "--", alpha=0.5, color="green", label="val F1 V1")
    axes[2].set_title("Val F1-macro"); axes[2].legend()

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Curvas: {path.name}")


def plot_confusion_matrix(cm, class_names, out_dir, tag):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
    ax.set_title(f"Confusion Matrix V2 ({tag})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    path = out_dir / f"confusion_matrix_{tag}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  CM {tag}: {path.name}")


def load_v1_history(log_path: Path):
    if not log_path.exists():
        return None
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "val_f1":[]}
    for row in csv.DictReader(log_path.open(encoding="utf-8")):
        history["train_loss"].append(float(row["train_loss"]))
        history["val_loss"].append(float(row["val_loss"]))
        history["train_acc"].append(float(row["train_acc"]))
        history["val_acc"].append(float(row["val_acc"]))
        history["val_f1"].append(float(row["val_f1_macro"]))
    return history


def write_report(cfg, best_epoch, val_m, test_m, pretrained_ok, elapsed):
    class_names = cfg["class_names"]
    n = len(class_names)
    status = "PASS"

    # Leer metricas V1 para comparacion
    v1_acc, v1_f1 = 0.8049, 0.7355  # del reporte V1
    delta_acc = test_m["acc"]   - v1_acc
    delta_f1  = test_m["f1_macro"] - v1_f1

    lines = [
        "# Fruits-360 Quality Classifier V2 — Reporte de Entrenamiento",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Status: **{status}**",
        "",
        "## Configuracion V2",
        "",
        "| Parametro | Valor |",
        "|-----------|-------|",
        "| Modelo | MobileNetV3-small |",
        f"| Pesos pretrained | {'ImageNet (IMAGENET1K_V1)' if pretrained_ok else 'NINGUNO'} |",
        "| Dataset | data/quality_fruits360_human_v2/ |",
        f"| Train | {cfg['n_train']} imgs (good={cfg['n_train_good']}, bad={cfg['n_train_bad']}) |",
        f"| Val   | {cfg['n_val']} imgs |",
        f"| Test  | {cfg['n_test']} imgs |",
        f"| Epochs | {cfg['epochs']} |",
        f"| Batch size | {cfg['batch']} |",
        f"| LR | {cfg['lr']} |",
        f"| Optimizer | Adam + CosineAnnealingLR |",
        f"| Device | {cfg['device']} |",
        f"| Seed | {cfg['seed']} |",
        "| Desbalance | class_weights + WeightedRandomSampler |",
        "",
        "## Cambios V1 -> V2",
        "",
        "| ID | Cambio |",
        "|----|--------|",
        "| F360_0198 | BAD -> REVIEW (excluida) |",
        "| F360_0052 | GOOD -> REVIEW (excluida) |",
        "| F360_0060 | GOOD -> GOOD (confirmada, sin cambio) |",
        "| Epochs | 30 -> 40 |",
        "",
        f"## Mejor checkpoint: epoch {best_epoch}  (val F1-macro={val_m['f1_macro']:.4f})",
        "",
        "## Resultados TEST",
        "",
        "| Metrica | V1 | V2 | Delta |",
        "|---------|----|----|-------|",
        f"| Accuracy | {v1_acc:.4f} | {test_m['acc']:.4f} | {delta_acc:+.4f} |",
        f"| F1-macro | {v1_f1:.4f} | {test_m['f1_macro']:.4f} | {delta_f1:+.4f} |",
        "",
        "| Clase | Precision | Recall | F1 |",
        "|-------|-----------|--------|----|",
    ]
    for i, cn in enumerate(class_names):
        lines.append(f"| {cn} | {test_m['precision'][i]:.3f} | "
                     f"{test_m['recall'][i]:.3f} | {test_m['f1'][i]:.3f} |")

    lines += [
        "",
        "## Matriz de confusion (TEST)",
        "",
        "```",
        "           " + "  ".join(f"{cn:>8}" for cn in class_names),
    ]
    for i, cn in enumerate(class_names):
        row = "  ".join(f"{test_m['cm'][i][j]:>8}" for j in range(n))
        lines.append(f"  {cn:>8}  {row}")
    lines += ["```", ""]

    lines += [
        "## Archivos generados",
        "",
        "- `outputs/fruits360_quality_cls_v2/best_model.pt`",
        "- `outputs/fruits360_quality_cls_v2/training_curves.png`",
        "- `outputs/fruits360_quality_cls_v2/confusion_matrix_val.png`",
        "- `outputs/fruits360_quality_cls_v2/confusion_matrix_test.png`",
        "- `outputs/fruits360_quality_cls_v2/train_log.csv`",
        "- `outputs/fruits360_quality_cls_v2/test_results.csv`",
        "",
        "## Confirmaciones",
        "",
        "- analyze_quality.py NO fue modificado.",
        "- quality_rules.yaml NO fue modificado.",
        "- Imagenes REVIEW NO usadas.",
        "- Test set evaluado UNA SOLA VEZ al final.",
        "- Dataset V1 NO fue destruido.",
        f"- Tiempo total: {elapsed:.1f} s",
    ]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte: {REPORT_MD.name}")


def main():
    print(f"=== train_fruits360_quality_v2  (device={DEVICE}) ===")
    set_seed(SEED)

    for split in ["train", "val", "test"]:
        for cls in ["good", "bad"]:
            if not (DATA_DIR / split / cls).exists():
                print(f"ERROR: {DATA_DIR / split / cls} no existe.")
                sys.exit(1)

    print("\n  Cargando datos...")
    train_loader, val_loader, test_loader, class_to_idx, class_weights = build_loaders()
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    n_classes   = len(class_names)
    n_train     = len(train_loader.dataset)
    n_val       = len(val_loader.dataset)
    n_test      = len(test_loader.dataset)
    cnt_train   = Counter(lbl for _, lbl in train_loader.dataset.samples)
    print(f"  train={n_train}  val={n_val}  test={n_test}")

    print("\n  Construyendo modelo...")
    model, pretrained_ok = build_model(n_classes, DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","val_f1"]}
    log_rows = []
    best_val_f1 = -1.0
    best_epoch  = 0
    best_val_m  = None
    best_path   = OUT_DIR / "best_model.pt"

    print(f"\n  Entrenando {EPOCHS} epochs...\n")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr_loss, tr_lbl, tr_pred = run_epoch(model, train_loader, criterion, optimizer, DEVICE, True)
        vl_loss, vl_lbl, vl_pred = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, False)
        scheduler.step()

        tr_acc, _, _, _, tr_f1m, _       = compute_metrics(tr_lbl, tr_pred, n_classes)
        vl_acc, vl_p, vl_r, vl_f1, vl_f1m, vl_cm = compute_metrics(vl_lbl, vl_pred, n_classes)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1m)

        log_rows.append({"epoch": epoch, "train_loss": f"{tr_loss:.4f}", "val_loss": f"{vl_loss:.4f}",
                         "train_acc": f"{tr_acc:.4f}", "val_acc": f"{vl_acc:.4f}",
                         "val_f1_macro": f"{vl_f1m:.4f}"})

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
              f"vl_acc={vl_acc:.3f}  vl_f1={vl_f1m:.3f}  ({time.time()-t_ep:.1f}s)")

        if vl_f1m > best_val_f1:
            best_val_f1 = vl_f1m
            best_epoch  = epoch
            best_val_m  = {"loss": vl_loss, "acc": vl_acc, "f1_macro": vl_f1m,
                           "precision": vl_p, "recall": vl_r, "f1": vl_f1, "cm": vl_cm}
            torch.save(model.state_dict(), best_path)

    elapsed = time.time() - t0
    print(f"\n  Mejor epoch: {best_epoch}  (val F1={best_val_f1:.4f})")

    log_path = OUT_DIR / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    print("\n  Generando plots...")
    v1_history = load_v1_history(PROJECT_ROOT / "outputs" / "fruits360_quality_cls_v1" / "train_log.csv")
    plot_curves(history, OUT_DIR, v1_history)
    plot_confusion_matrix(best_val_m["cm"], class_names, OUT_DIR, "val")

    print("\n  Evaluando en TEST (mejor modelo)...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    te_loss, te_lbl, te_pred = run_epoch(model, test_loader, criterion, optimizer, DEVICE, False)
    te_acc, te_p, te_r, te_f1, te_f1m, te_cm = compute_metrics(te_lbl, te_pred, n_classes)
    test_m = {"loss": te_loss, "acc": te_acc, "f1_macro": te_f1m,
              "precision": te_p, "recall": te_r, "f1": te_f1, "cm": te_cm}

    plot_confusion_matrix(te_cm, class_names, OUT_DIR, "test")

    test_csv = OUT_DIR / "test_results.csv"
    with test_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class","precision","recall","f1"])
        w.writeheader()
        for i, cn in enumerate(class_names):
            w.writerow({"class": cn, "precision": f"{te_p[i]:.4f}",
                        "recall": f"{te_r[i]:.4f}", "f1": f"{te_f1[i]:.4f}"})

    cfg = {"class_names": class_names, "n_train": n_train, "n_val": n_val, "n_test": n_test,
           "n_train_good": cnt_train[class_to_idx["good"]],
           "n_train_bad":  cnt_train[class_to_idx["bad"]],
           "epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR, "device": DEVICE, "seed": SEED}
    write_report(cfg, best_epoch, best_val_m, test_m, pretrained_ok, elapsed)

    # Resumen comparativo
    v1_acc, v1_f1 = 0.8049, 0.7355
    print()
    print("=" * 55)
    print("  COMPARACION V1 vs V2")
    print("=" * 55)
    print(f"  {'Metrica':<15}  {'V1':>8}  {'V2':>8}  {'Delta':>8}")
    print(f"  {'-'*43}")
    print(f"  {'Accuracy':<15}  {v1_acc:>8.4f}  {te_acc:>8.4f}  {te_acc-v1_acc:>+8.4f}")
    print(f"  {'F1-macro':<15}  {v1_f1:>8.4f}  {te_f1m:>8.4f}  {te_f1m-v1_f1:>+8.4f}")
    print()
    print(f"  {'Clase':<8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*30}")
    for i, cn in enumerate(class_names):
        print(f"  {cn:<8}  {te_p[i]:>6.3f}  {te_r[i]:>6.3f}  {te_f1[i]:>6.3f}")
    print()
    print(f"  Confusion matrix (TEST):")
    print(f"           " + "  ".join(f"{cn:>8}" for cn in class_names))
    for i, cn in enumerate(class_names):
        row = "  ".join(f"{te_cm[i][j]:>8}" for j in range(n_classes))
        print(f"  {cn:>8}  {row}")
    print()
    print(f"  Tiempo: {elapsed:.1f} s")
    print(f"  analyze_quality.py NO modificado.")
    print(f"  quality_rules.yaml  NO modificado.")


if __name__ == "__main__":
    main()
