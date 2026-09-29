"""Compact CNN baseline and reproducible brain MRI slice data pipeline."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_SIZE = 128


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(data_root: Path, batch_size: int = 32, seed: int = 42):
    """Create stratified train/validation loaders and a held-out test loader."""
    train_root = data_root / "Training"
    test_root = data_root / "Testing"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"Expected Training and Testing under {data_root}")

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_aug = datasets.ImageFolder(train_root, transform=train_transform)
    train_eval = datasets.ImageFolder(train_root, transform=eval_transform)
    test_set = datasets.ImageFolder(test_root, transform=eval_transform)
    expected = list(CLASS_NAMES)
    if train_aug.classes != expected or test_set.classes != expected:
        raise ValueError(f"Expected class order {expected}; got train={train_aug.classes}, test={test_set.classes}")

    targets = np.asarray(train_aug.targets)
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for class_id in range(len(CLASS_NAMES)):
        indices = np.flatnonzero(targets == class_id)
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * 0.15))
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())
    generator = torch.Generator().manual_seed(seed)
    kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": False}
    train_loader = DataLoader(Subset(train_aug, train_indices), shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(Subset(train_eval, val_indices), shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, train_aug.class_to_idx


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class BrainMRICNN(nn.Module):
    """Small four-class CNN; the final convolutional block supports Grad-CAM."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.35), nn.Linear(192, 96),
            nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
