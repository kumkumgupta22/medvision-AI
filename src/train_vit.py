"""Train a compact Vision Transformer for academic brain MRI slice classification."""
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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_SIZE = 96
PATCH_SIZE = 16
EMBED_DIM = 64
DEPTH = 3
NUM_HEADS = 4
IMAGENET_MEAN = (0.5,)
IMAGENET_STD = (0.5,)


class VisionTransformer(nn.Module):
    """Small ViT with 8x8 image patches, a class token, and three encoder blocks."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        patch_count = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.patch_embed = nn.Conv2d(1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.class_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, EMBED_DIM))
        self.embedding_dropout = nn.Dropout(0.1)
        block = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=DEPTH, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMBED_DIM, num_classes))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, patches), dim=1)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        encoded = self.norm(self.encoder(tokens))
        return self.classifier(encoded[:, 0])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedImageDataset(Dataset):
    """Keep resized tensors in memory to avoid decoding the images on every epoch."""

    def __init__(self, images: torch.Tensor, targets: torch.Tensor, indices, augment: bool = False):
        self.images = images
        self.targets = targets
        self.indices = list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        image = self.images[source_index]
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
        return image, self.targets[source_index]


def cache_folder(folder: Path):
    source = datasets.ImageFolder(folder, transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]))
    cache = torch.empty((len(source), 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    print(f"Caching {len(source)} images from {folder} in memory.", flush=True)
    for index, (image, _) in enumerate(source):
        cache[index].copy_(image)
    return cache, torch.as_tensor(source.targets, dtype=torch.long), source.classes, source.class_to_idx


def make_loaders(data_root: Path, batch_size: int, seed: int, workers: int):
    train_root, test_root = data_root / "Training", data_root / "Testing"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"Expected Training and Testing folders under {data_root}")
    train_images, targets, train_classes, class_to_idx = cache_folder(train_root)
    test_images, test_targets, test_classes, _ = cache_folder(test_root)
    expected = list(CLASS_NAMES)
    if train_classes != expected or test_classes != expected:
        raise ValueError(f"Expected classes {expected}; got train={train_classes}, test={test_classes}")

    target_array = targets.numpy()
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for class_id, class_name in enumerate(CLASS_NAMES):
        indices = np.flatnonzero(target_array == class_id)
        if len(indices) < 2:
            raise ValueError(f"Need at least two training images for class {class_name}")
        rng.shuffle(indices)
        val_count = max(1, round(len(indices) * 0.15))
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    train_set = CachedImageDataset(train_images, targets, train_indices, augment=True)
    val_set = CachedImageDataset(train_images, targets, val_indices)
    test_set = CachedImageDataset(test_images, test_targets, range(len(test_targets)))
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, class_to_idx

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count, truth, predictions = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            count += labels.size(0)
            truth.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return loss_sum / max(count, 1), accuracy_score(truth, predictions), truth, predictions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "brain_tumor")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "vit_small")
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0, help="Keep at 0 for Windows compatibility")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1 or args.workers < 0:
        parser.error("epochs, batch size, patience, and threads must be positive; workers cannot be negative")

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_to_idx = make_loaders(
        args.data, args.batch_size, args.seed, args.workers
    )
    model = VisionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    checkpoint_path = args.model_dir / "vit_tiny_best.pt"
    best_val_loss, best_val_accuracy, best_epoch = float("inf"), 0.0, 0
    stale_epochs, history = 0, []
    started = time.time()
    print(
        f"Device: {device}; train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}; ViT image={IMAGE_SIZE}, patch={PATCH_SIZE}, "
        f"dim={EMBED_DIM}, depth={DEPTH}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch {epoch}/{args.epochs} - train loss {train_loss:.4f}, acc {train_acc:.4f}; "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss, best_val_accuracy, best_epoch, stale_epochs = val_loss, val_acc, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": list(CLASS_NAMES),
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "patch_size": PATCH_SIZE,
                "embed_dim": EMBED_DIM,
                "depth": DEPTH,
                "num_heads": NUM_HEADS,
                "architecture": "ViT_Tiny",
                "pretrained": False,
                "seed": args.seed,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(args.output / "training_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    history_frame.plot(x="epoch", y=["train_loss", "val_loss"], ax=axes[0], title="Loss")
    history_frame.plot(x="epoch", y=["train_accuracy", "val_accuracy"], ax=axes[1], title="Accuracy")
    fig.tight_layout()
    fig.savefig(args.output / "training_curves.png", dpi=160)
    plt.close(fig)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, _, y_true, y_pred = run_epoch(model, test_loader, loss_fn, device)
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(args.output / "classification_report.csv")
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(args.output / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("ViT-Tiny — held-out test set")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    metrics = {
        "task": "brain MRI image/slice classification (academic prototype; not clinical diagnosis)",
        "model": "ViT-Tiny (trained from scratch)",
        "architecture": {
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
        },
        "pretrained": False,
        "device": str(device),
        "seed": args.seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "class_names": list(CLASS_NAMES),
        "class_to_idx": class_to_idx,
        "train_count": len(train_loader.dataset),
        "validation_count": len(val_loader.dataset),
        "test_count": len(test_loader.dataset),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nHeld-out test results:")
    print(json.dumps({key: metrics[key] for key in (
        "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_loss"
    )}, indent=2))
    print(f"Saved best checkpoint: {checkpoint_path}", flush=True)
    print(f"Saved evaluation outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()




