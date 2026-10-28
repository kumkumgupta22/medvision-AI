"""Evaluate a checkpoint on an external test split after filtering dataset overlap."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fft import dctn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train_efficientnet import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no-tumor": "notumor",
    "pituitary": "pituitary",
}


def image_files(paths: list[Path]) -> list[Path]:
    return sorted(
        p for root in paths if root.is_dir()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def dct_basis(size: int = 32) -> np.ndarray:
    x = np.arange(size, dtype=np.float32)
    k = np.arange(size, dtype=np.float32)[:, None]
    basis = np.cos(np.pi * (2 * x + 1) * k / (2 * size)).astype(np.float32)
    basis[0] *= math.sqrt(1 / size)
    basis[1:] *= math.sqrt(2 / size)
    return basis


_DCT_BASIS = dct_basis()


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    low = dctn(pixels, type=2, norm="ortho")[:8, :8]
    median = np.median(low.reshape(-1)[1:])
    value = 0
    for bit in (low > median).reshape(-1):
        value = (value << 1) | int(bit)
    return value


class ExternalImages(Dataset):
    def __init__(self, rows: list[tuple[Path, int]]):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, label = self.rows[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="External dataset root containing a test/ folder")
    parser.add_argument("--reference-data", type=Path, default=project_root / "data" / "brain_tumor")
    parser.add_argument("--checkpoint", type=Path, default=project_root / "models" / "efficientnet_b0_best.pt")
    parser.add_argument("--output", type=Path, default=project_root / "outputs" / "external_validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--phash-distance", type=int, default=5,
                        help="Exclude non-exact images within this pHash Hamming distance; use -1 to disable")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.phash_distance < -1:
        parser.error("batch size and threads must be positive; pHash distance must be -1 or greater")
    test_root = args.data / "test"
    reference_roots = [args.reference_data / "Training", args.reference_data / "Testing"]
    if not test_root.is_dir() or not all(p.is_dir() for p in reference_roots):
        parser.error("Expected --data/test and --reference-data/Training plus Testing folders")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    reference_files = image_files(reference_roots)
    test_files = image_files([test_root])
    if not reference_files or not test_files:
        parser.error("No supported image files were found")

    reference_digests: set[bytes] = set()
    digest_to_reference: dict[bytes, Path] = {}
    for path in reference_files:
        digest = sha256(path)
        reference_digests.add(digest)
        digest_to_reference.setdefault(digest, path)
    external_digests = {path: sha256(path) for path in test_files}
    exact_overlaps = {path for path, digest in external_digests.items() if digest in reference_digests}

    near_overlaps: set[Path] = set()
    if args.phash_distance >= 0:
        reference_hashes = [perceptual_hash(path) for path in digest_to_reference.values()]
        reference_hashes = np.asarray(reference_hashes, dtype=np.uint64)
        candidates = [path for path in test_files if path not in exact_overlaps]
        for start in range(0, len(candidates), 128):
            batch = candidates[start:start + 128]
            external_hashes = np.asarray([perceptual_hash(path) for path in batch], dtype=np.uint64)
            xor = np.bitwise_xor(external_hashes[:, None], reference_hashes[None, :])
            byte_view = xor.view(np.uint8).reshape(xor.shape + (8,))
            distances = np.unpackbits(byte_view, axis=2).sum(axis=2)
            for path, distance in zip(batch, distances.min(axis=1)):
                if int(distance) <= args.phash_distance:
                    near_overlaps.add(path)

    rows: list[tuple[Path, int]] = []
    invalid_images = []
    for path in test_files:
        if path in exact_overlaps or path in near_overlaps:
            continue
        class_name = CLASS_ALIASES.get(path.parent.name.lower())
        if class_name not in CLASS_NAMES:
            invalid_images.append({"path": str(path), "reason": "unrecognized class folder"})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            invalid_images.append({"path": str(path), "reason": str(error)})
            continue
        rows.append((path, CLASS_NAMES.index(class_name)))
    if not rows:
        parser.error("No external test images remain after overlap filtering")

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "EfficientNet_B0":
        parser.error("The checkpoint is not an EfficientNet-B0 model")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    loader = DataLoader(ExternalImages(rows), batch_size=args.batch_size, shuffle=False, num_workers=0)
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions)

    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(args.output / "classification_report.csv")
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(args.output / "confusion_matrix.csv")
    metrics = {
        "task": "academic brain MRI image/slice classification; not clinical diagnosis",
        "model": "EfficientNet-B0",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_accuracy": checkpoint.get("val_accuracy"),
        "device": str(device),
        "published_external_test_images": len(test_files),
        "reference_images_compared": len(reference_files),
        "reference_unique_sha256": len(reference_digests),
        "exact_hash_overlaps_excluded": len(exact_overlaps),
        "additional_near_duplicates_excluded": len(near_overlaps),
        "phash_hamming_threshold": args.phash_distance,
        "invalid_images_excluded": len(invalid_images),
        "evaluated_images": len(y_true),
        "evaluated_class_counts": dict(Counter(CLASS_NAMES[label] for _, label in rows)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "class_names": list(CLASS_NAMES),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "overlap_method": "SHA-256 exact matches and 64-bit DCT perceptual hash against all Training and Testing images.",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved external evaluation reports: {args.output}")


if __name__ == "__main__":
    main()


