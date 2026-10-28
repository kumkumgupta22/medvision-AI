"""Evaluate the saved EfficientNet-B0 checkpoint on the held-out Testing split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from torchvision import models

from train_efficientnet import CLASS_NAMES, make_loaders


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "brain_tumor")
    parser.add_argument("--checkpoint", type=Path, default=root / "models" / "efficientnet_b0_best.pt")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and threads must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_to_idx = make_loaders(args.data, args.batch_size, args.seed, workers=0)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "EfficientNet_B0":
        parser.error("The checkpoint is not an EfficientNet-B0 model")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    y_true, y_pred, total_loss, count = [], [], 0.0, 0
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            count += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    args.output.mkdir(parents=True, exist_ok=True)
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
        "model": "EfficientNet-B0", "pretrained": checkpoint.get("pretrained", False),
        "device": str(device), "seed": args.seed, "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_accuracy": checkpoint.get("val_accuracy"), "test_count": count,
        "test_loss": total_loss / max(count, 1), "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "class_names": list(CLASS_NAMES), "class_to_idx": class_to_idx,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved test reports: {args.output}")


if __name__ == "__main__":
    main()
