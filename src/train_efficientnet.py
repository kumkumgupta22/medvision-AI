"""Fine-tune an ImageNet-pretrained EfficientNet-B0 for MRI slice classification."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(data_root: Path, batch_size: int, seed: int, workers: int):
    train_root, test_root = data_root / "Training", data_root / "Testing"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"Expected Training and Testing folders under {data_root}")

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_aug = datasets.ImageFolder(train_root, transform=train_transform)
    train_eval = datasets.ImageFolder(train_root, transform=eval_transform)
    test_set = datasets.ImageFolder(test_root, transform=eval_transform)
    expected = list(CLASS_NAMES)
    if train_aug.classes != expected or test_set.classes != expected:
        raise ValueError(f"Expected classes {expected}; got train={train_aug.classes}, test={test_set.classes}")

    targets = np.asarray(train_aug.targets)
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for class_id in range(len(CLASS_NAMES)):
        indices = np.flatnonzero(targets == class_id)
        if len(indices) < 2:
            raise ValueError(f"Need at least two training images for class {CLASS_NAMES[class_id]}")
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * 0.15))
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    generator = torch.Generator().manual_seed(seed)
    loader_args = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(Subset(train_aug, train_indices), shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(Subset(train_eval, val_indices), shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, train_aug.class_to_idx


def run_epoch(model, loader, loss_fn, device, optimizer=None, keep_features_frozen=False):
    training = optimizer is not None
    model.train(training)
    if training and keep_features_frozen:
        # Keep frozen backbone batch-normalization statistics fixed during head warmup.
        model.features.eval()
    loss_sum, count, truth, predictions = 0.0, 0, [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            count += labels.size(0)
            truth.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return loss_sum / max(count, 1), accuracy_score(truth, predictions), truth, predictions


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "brain_tumor")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "efficientnet_b0")
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Backbone fine-tuning learning rate")
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--freeze-epochs", type=int, default=1, help="Warm up the classifier head before unfreezing features")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers; keep at 0 for Windows compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-scratch", action="store_true", help="Do not download/use pretrained ImageNet weights")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1 or args.workers < 0 or args.freeze_epochs < 0:
        parser.error("epochs, batch size, patience, and threads must be positive; workers/freeze epochs cannot be negative")

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_to_idx = make_loaders(
        args.data, args.batch_size, args.seed, args.workers
    )

    weights = None if args.from_scratch else models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    if args.freeze_epochs:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model = model.to(device)

    if args.freeze_epochs:
        # Warm up only the new classification head; add the backbone after unfreezing.
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.head_learning_rate, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.classifier.parameters(), "lr": args.head_learning_rate},
            {"params": model.features.parameters(), "lr": args.learning_rate},
        ], weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    checkpoint_path = args.model_dir / "efficientnet_b0_best.pt"
    best_val_loss, best_val_accuracy, best_epoch = float("inf"), 0.0, 0
    stale_epochs, history = 0, []
    started = time.time()
    print(f"Device: {device}; train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}", flush=True)
    if weights is not None:
        print("Using ImageNet-pretrained EfficientNet-B0 weights.", flush=True)

    for epoch in range(1, args.epochs + 1):
        frozen = bool(args.freeze_epochs and epoch <= args.freeze_epochs)
        if args.freeze_epochs and epoch == args.freeze_epochs + 1:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            optimizer.add_param_group({"params": model.features.parameters(), "lr": args.learning_rate})
            print("Unfroze EfficientNet features for fine-tuning.", flush=True)
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, frozen)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc,
                        "backbone_lr": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
                        "head_lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch}/{args.epochs} - train loss {train_loss:.4f}, acc {train_acc:.4f}; "
              f"val loss {val_loss:.4f}, acc {val_acc:.4f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss, best_val_accuracy, best_epoch, stale_epochs = val_loss, val_acc, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": list(CLASS_NAMES),
                        "class_to_idx": class_to_idx, "image_size": IMAGE_SIZE,
                        "architecture": "EfficientNet_B0", "pretrained": weights is not None,
                        "seed": args.seed, "epoch": epoch, "val_loss": val_loss,
                        "val_accuracy": val_acc}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, _, y_true, y_pred = run_epoch(model, test_loader, loss_fn, device)
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(args.output / "classification_report.csv", index=True)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(args.output / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("EfficientNet-B0 — held-out test set")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    metrics = {
        "task": "brain MRI image/slice classification (academic prototype; not clinical diagnosis)",
        "model": "EfficientNet-B0", "pretrained": weights is not None, "device": str(device),
        "seed": args.seed, "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss, "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_loss, "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "class_names": list(CLASS_NAMES), "class_to_idx": class_to_idx,
        "train_count": len(train_loader.dataset), "validation_count": len(val_loader.dataset),
        "test_count": len(test_loader.dataset), "elapsed_seconds": time.time() - started,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nHeld-out test results:")
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_loss")}, indent=2))
    print(f"Saved best checkpoint: {checkpoint_path}", flush=True)
    print(f"Saved evaluation outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()

