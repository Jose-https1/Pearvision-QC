"""
train_quality_roi_masked_clean_u3.py
--------------------------------------
Trains U3 ROI/masked clean classifier (MobileNetV3-small, binary good/bad).

Input : data/quality_roi_masked_clean_u3/train/ , val/ , test/
Output: outputs/fruits360_quality_cls_u3_roi_masked_clean/

DOES NOT modify V2, analyze_quality.py, or quality_rules.yaml.
"""

import sys
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torchvision.transforms as T
import torchvision.models as models

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix
    )
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data/quality_roi_masked_clean_u3"
OUT_DIR   = PROJECT_ROOT / "outputs/fruits360_quality_cls_u3_roi_masked_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── hyperparams ───────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 16
EPOCHS       = 40
PATIENCE     = 8
LR           = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED  = 42

CLASSES = ["bad", "good"]  # alphabetical — bad=0, good=1

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
class PearDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples = []
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((p, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(str(path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


def build_transforms(augment: bool):
    base = [T.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        base += [
            T.RandomHorizontalFlip(),
            T.RandomRotation(12),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    base += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(base)


def build_model(num_classes=2, device=None):
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, num_classes
    )
    return model.to(device)


def make_sampler(dataset: PearDataset):
    labels = [s[1] for s in dataset.samples]
    from collections import Counter
    counts = Counter(labels)
    class_w = {c: 1.0 / counts[c] for c in counts}
    sample_w = [class_w[l] for l in labels]
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_paths, all_probs = [], [], [], []
    for imgs, labels, paths in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out   = model(imgs)
        probs = torch.softmax(out, dim=1)
        loss  = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(paths)
    return total_loss / total, correct / total, all_preds, all_labels, all_probs, all_paths


def save_confusion_matrix(y_true, y_pred, path: Path, title: str):
    if not HAS_MPL or not HAS_SKL:
        return
    from sklearn.metrics import confusion_matrix as _cm
    cm = _cm(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


def save_training_curves(train_losses, val_losses, train_accs, val_accs, path: Path):
    if not HAS_MPL:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="train")
    ax1.plot(epochs, val_losses, label="val")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(epochs, train_accs, label="train")
    ax2.plot(epochs, val_accs, label="val")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


INV_NORM_MEAN = torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225])
INV_NORM_STD  = torch.tensor([1/0.229, 1/0.224, 1/0.225])

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # t: [3,H,W] normalized
    t = t.clone()
    for c in range(3):
        t[c] = t[c] * [0.229, 0.224, 0.225][c] + [0.485, 0.456, 0.406][c]
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_grid(items, path: Path, thumb=112, cols=8, title=""):
    """items: list of (img_tensor, label_str, pred_str, conf)"""
    if not items:
        blank = Image.new("RGB", (200, 100), (30, 30, 30))
        blank.save(str(path))
        return
    from PIL import ImageDraw
    cell_h = thumb + 28
    cell_w = thumb
    n = len(items)
    rows = max(1, (n + cols - 1) // cols)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows + 20), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for idx, (tensor, true_str, pred_str, conf) in enumerate(items):
        col = idx % cols
        row = idx // cols
        px, py = col * cell_w, row * cell_h + 20
        try:
            thumb_img = tensor_to_pil(tensor).resize((thumb, thumb))
            sheet.paste(thumb_img, (px, py))
        except Exception:
            pass
        correct = true_str == pred_str
        color = (100, 220, 100) if correct else (220, 80, 80)
        label = f"T:{true_str[0].upper()} P:{pred_str[0].upper()} {conf:.2f}"
        draw.text((px + 2, py + thumb + 2), label, fill=color)
    draw.text((4, 2), title, fill=(200, 200, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(path))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("train_quality_roi_masked_clean_u3.py")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    train_ds = PearDataset(DATA_ROOT / "train", build_transforms(augment=True))
    val_ds   = PearDataset(DATA_ROOT / "val",   build_transforms(augment=False))
    test_ds  = PearDataset(DATA_ROOT / "test",  build_transforms(augment=False))

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    sampler = make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # Class weights for loss
    from collections import Counter
    train_labels = [s[1] for s in train_ds.samples]
    counts = Counter(train_labels)
    w = torch.tensor([1.0 / counts[i] for i in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    model     = build_model(2, device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    patience_left = PATIENCE
    best_epoch    = 0

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    log_rows = []

    print()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        flag = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch    = epoch
            patience_left = PATIENCE
            torch.save(model.state_dict(), str(OUT_DIR / "best_model.pt"))
            flag = " *BEST*"
        else:
            patience_left -= 1

        print(f"  Ep {epoch:02d}/{EPOCHS}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f}"
              f"{flag}")

        log_rows.append({"epoch": epoch, "tr_loss": round(tr_loss, 5),
                         "tr_acc": round(tr_acc, 4), "vl_loss": round(vl_loss, 5),
                         "vl_acc": round(vl_acc, 4)})

        if patience_left == 0:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    torch.save(model.state_dict(), str(OUT_DIR / "last_model.pt"))

    # Save log
    with open(str(OUT_DIR / "train_log.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch","tr_loss","tr_acc","vl_loss","vl_acc"])
        writer.writeheader(); writer.writerows(log_rows)

    # Save training curves
    save_training_curves(train_losses, val_losses, train_accs, val_accs,
                         OUT_DIR / "training_curves.png")

    # Load best model for evaluation
    model.load_state_dict(torch.load(str(OUT_DIR / "best_model.pt"), map_location=device))

    # Val confusion matrix
    _, _, val_preds, val_labels, _, _ = evaluate(model, val_loader, criterion, device)
    save_confusion_matrix(val_labels, val_preds,
                          OUT_DIR / "confusion_matrix_val.png", "Val Confusion Matrix")

    # Test evaluation
    _, _, test_preds, test_labels, test_probs, test_paths = evaluate(
        model, test_loader, criterion, device)

    # Test results CSV
    results_rows = []
    for path, pred, label, probs in zip(test_paths, test_preds, test_labels, test_probs):
        results_rows.append({
            "path": path,
            "true_class": CLASSES[label],
            "pred_class": CLASSES[pred],
            "p_bad":  round(probs[0], 4),
            "p_good": round(probs[1], 4),
            "correct": int(pred == label),
        })
    with open(str(OUT_DIR / "test_results.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path","true_class","pred_class","p_bad","p_good","correct"])
        writer.writeheader(); writer.writerows(results_rows)

    # Confusion matrix test
    save_confusion_matrix(test_labels, test_preds,
                          OUT_DIR / "confusion_matrix_test.png", "Test Confusion Matrix")

    # Metrics
    acc = accuracy_score(test_labels, test_preds) if HAS_SKL else sum(r["correct"] for r in results_rows) / len(results_rows)
    if HAS_SKL:
        prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds,
                                                             labels=[0, 1], zero_division=0)
        f1_macro = f1.mean()
        # bad->good (false negative on bad = model says good but true=bad)
        bad_to_good = sum(1 for r in results_rows if r["true_class"]=="bad" and r["pred_class"]=="good")
        good_to_bad = sum(1 for r in results_rows if r["true_class"]=="good" and r["pred_class"]=="bad")
    else:
        prec = rec = f1 = np.array([0, 0])
        f1_macro = 0
        bad_to_good = good_to_bad = 0

    n_bad_test  = sum(1 for r in results_rows if r["true_class"] == "bad")
    n_good_test = sum(1 for r in results_rows if r["true_class"] == "good")

    print()
    print("-" * 40)
    print(f"Test accuracy   : {acc:.4f}")
    print(f"F1 macro        : {f1_macro:.4f}")
    print(f"F1 bad          : {f1[0]:.4f}  F1 good: {f1[1]:.4f}")
    print(f"BAD->GOOD       : {bad_to_good}/{n_bad_test}")
    print(f"GOOD->BAD       : {good_to_bad}/{n_good_test}")
    print("-" * 40)

    # Grids — need raw tensors
    # Reload test dataset without collate to get tensors
    test_ds_raw = PearDataset(DATA_ROOT / "test", build_transforms(augment=False))
    test_loader_raw = DataLoader(test_ds_raw, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    all_items, error_items = [], []
    with torch.no_grad():
        for tensor, label, path in test_loader_raw:
            tensor_single = tensor[0]
            lab = label.item()
            out = model(tensor.to(device))
            probs_t = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred = int(probs_t.argmax())
            conf = float(probs_t[pred])
            true_str = CLASSES[lab]
            pred_str = CLASSES[pred]
            item = (tensor_single, true_str, pred_str, conf)
            all_items.append(item)
            if pred != lab:
                error_items.append(item)

    save_grid(all_items,   OUT_DIR / "test_all_grid.jpg",    thumb=112, cols=8,
              title=f"Test all ({len(all_items)} images)")
    save_grid(error_items, OUT_DIR / "test_errors_grid.jpg", thumb=112, cols=8,
              title=f"Test errors ({len(error_items)} images)")

    print(f"test_all_grid   : {len(all_items)} images")
    print(f"test_errors_grid: {len(error_items)} errors")

    # Training report
    report_path = PROJECT_ROOT / "reports/train_quality_roi_masked_clean_u3_report.md"
    with open(str(report_path), "w", encoding="utf-8") as fh:
        fh.write("# Train U3 ROI Masked Clean Report\n\n")
        fh.write(f"**Fecha:** 2026-05-21\n\n")
        fh.write(f"## Configuracion\n\n")
        fh.write(f"- Modelo: MobileNetV3-small pretrained ImageNet\n")
        fh.write(f"- Epochs: hasta {EPOCHS} con early stopping patience={PATIENCE}\n")
        fh.write(f"- Mejor epoch: {best_epoch}\n")
        fh.write(f"- LR: {LR}, WeightDecay: {WEIGHT_DECAY}\n")
        fh.write(f"- Device: {device}\n")
        fh.write(f"- Augmentations: HorizontalFlip, Rotation(12), ColorJitter\n\n")
        fh.write(f"## Dataset\n\n")
        fh.write(f"- Train: {len(train_ds)} (good={counts[1]}, bad={counts[0]})\n")
        fh.write(f"- Val: {len(val_ds)}\n")
        fh.write(f"- Test: {len(test_ds)}\n\n")
        fh.write(f"## Resultados test\n\n")
        fh.write(f"| Metrica | Valor |\n|---|---|\n")
        fh.write(f"| Accuracy | {acc:.4f} |\n")
        fh.write(f"| F1 macro | {f1_macro:.4f} |\n")
        fh.write(f"| F1 bad | {f1[0]:.4f} |\n")
        fh.write(f"| F1 good | {f1[1]:.4f} |\n")
        fh.write(f"| BAD->GOOD | {bad_to_good}/{n_bad_test} |\n")
        fh.write(f"| GOOD->BAD | {good_to_bad}/{n_good_test} |\n\n")
        fh.write(f"## Archivos\n\n")
        for fn in ["best_model.pt","last_model.pt","train_log.csv","test_results.csv",
                   "confusion_matrix_test.png","confusion_matrix_val.png",
                   "training_curves.png","test_errors_grid.jpg","test_all_grid.jpg"]:
            fh.write(f"- outputs/fruits360_quality_cls_u3_roi_masked_clean/{fn}\n")
        fh.write("\n## Notas\n\n")
        fh.write("- No se modifico V2.\n")
        fh.write("- No se modifico analyze_quality.py ni quality_rules.yaml.\n")
        fh.write("- No se integro U3 en el pipeline final.\n")
    print(f"  Saved: {report_path.name}")

    print()
    print("=== training DONE ===")
    return acc, f1_macro, bad_to_good, good_to_bad, n_bad_test, n_good_test


if __name__ == "__main__":
    main()
