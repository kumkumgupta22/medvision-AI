"""Inspect the four-class dataset, decoding each image and writing exact counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
SPLITS = ("Training", "Testing")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "brain_tumor")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "dataset_analysis.json")
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data}")
    result = {"root": str(args.data.resolve()), "classes": list(CLASSES), "splits": {}, "invalid_images": []}
    for split in SPLITS:
        result["splits"][split] = {}
        for class_name in CLASSES:
            folder = args.data / split / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing expected class directory: {folder}")
            images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            valid = 0
            for path in images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    valid += 1
                except Exception as exc:
                    result["invalid_images"].append({"path": str(path), "error": str(exc)})
            result["splits"][split][class_name] = {"image_count": len(images), "valid_count": valid}
    result["total_images"] = sum(v["image_count"] for s in result["splits"].values() for v in s.values())
    result["valid_images"] = sum(v["valid_count"] for s in result["splits"].values() for v in s.values())
    result["invalid_count"] = len(result["invalid_images"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for split, values in result["splits"].items():
        print(f"{split}: " + ", ".join(f"{name}={item['image_count']}" for name, item in values.items()))
    print(f"Total: {result['total_images']} images; valid={result['valid_images']}; invalid={result['invalid_count']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
