"""Train and evaluate the MediVision AI CNN baseline on CPU or CUDA."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score)
from torch import nn

try:
    from .medivision_cnn import BrainMRICNN, CLASS_NAMES, make_loaders, seed_everything
except ImportError:
    from medivision_cnn import BrainMRICNN, CLASS_NAMES, make_loaders, seed_everything


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, count = 0.0, 0
    y_true, y_pred = [], []
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
            total_loss += loss.item() * labels.size(0)
            count += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    return total_loss / max(count, 1), accuracy_score(y_true, y_pred), y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "brain_tumor")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads; lower if the computer is busy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads, 4))
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_to_idx = make_loaders(args.data, args.batch_size, args.seed)
    model = BrainMRICNN(len(CLASS_NAMES)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    checkpoint = args.model_dir / "cnn_baseline_best.pt"
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    stale_epochs = 0
    history = []
    started = time.time()
    print(f"Device: {device}; train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch}/{args.epochs} - train loss {train_loss:.4f}, acc {train_acc:.4f}; "
              f"val loss {val_loss:.4f}, acc {val_acc:.4f}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss, best_val_accuracy, stale_epochs = val_loss, val_acc, 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": list(CLASS_NAMES),
                        "class_to_idx": class_to_idx, "image_size": 128, "input_channels": 1,
                        "seed": args.seed, "epoch": epoch, "val_loss": val_loss,
                        "val_accuracy": val_acc, "architecture": "BrainMRICNN"}, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

    pd.DataFrame(history).to_csv(args.output / "cnn_training_history.csv", index=False)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    test_loss, _, y_true, y_pred = run_epoch(model, test_loader, loss_fn, device)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    report = classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(args.output / "cnn_classification_report.csv", index=True)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(args.output / "cnn_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("CNN baseline — held-out test set")
    fig.tight_layout()
    fig.savefig(args.output / "cnn_confusion_matrix.png", dpi=160)
    plt.close(fig)
    metrics = {
        "task": "brain MRI image/slice classification (academic prototype; not clinical diagnosis)",
        "model": "BrainMRICNN", "device": str(device), "seed": args.seed,
        "epochs_completed": len(history), "best_epoch": saved["epoch"],
        "best_validation_loss": best_val_loss, "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_loss, "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "class_names": list(CLASS_NAMES), "class_to_idx": class_to_idx,
        "train_count": len(train_loader.dataset), "validation_count": len(val_loader.dataset),
        "test_count": len(test_loader.dataset), "elapsed_seconds": time.time() - started,
    }
    (args.output / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nHeld-out test results:")
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_loss")}, indent=2))
    print(f"Saved best checkpoint: {checkpoint}", flush=True)
    print(f"Saved metrics and evaluation outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
