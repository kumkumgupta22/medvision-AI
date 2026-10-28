"""FastAPI interface for academic brain MRI image/slice classification."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models

from src.medivision_cnn import BrainMRICNN, CLASS_NAMES, IMAGE_SIZE as CNN_IMAGE_SIZE
from src.medivision_vit import VisionTransformer, IMAGE_SIZE as VIT_IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "cnn": ROOT / "models" / "cnn_baseline_best.pt",
    "efficientnet": ROOT / "models" / "efficientnet_b0_best.pt",
    "vit": ROOT / "models" / "vit_tiny_best.pt",
}
MODEL_LABELS = {
    "cnn": "CNN baseline",
    "efficientnet": "EfficientNet-B0",
    "vit": "ViT-Tiny",
}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DISCLAIMER = (
    "Academic prototype for brain MRI image/slice classification only. "
    "Not a clinical diagnostic system; do not use this output for patient-care decisions."
)
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)

app = FastAPI(title="MediVision AI", version="0.2.0", description=DISCLAIMER)
_models: dict[str, nn.Module] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str) -> nn.Module:
    if model_name not in CHECKPOINTS:
        raise ValueError("Choose CNN, EfficientNet-B0, or ViT-Tiny.")
    if model_name in _models:
        return _models[model_name]
    checkpoint_path = CHECKPOINTS[model_name]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{MODEL_LABELS[model_name]} checkpoint not found. Train that model first; "
            "model weights are stored locally and are not included in Git."
        )
    checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    if model_name == "cnn":
        if checkpoint.get("architecture") != "BrainMRICNN":
            raise RuntimeError("CNN checkpoint architecture is not supported by this API.")
        model = BrainMRICNN(num_classes=len(CLASS_NAMES))
    elif model_name == "efficientnet":
        if checkpoint.get("architecture") != "EfficientNet_B0":
            raise RuntimeError("EfficientNet checkpoint architecture is not supported by this API.")
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    else:
        if checkpoint.get("architecture") != "ViT_Tiny":
            raise RuntimeError("ViT checkpoint architecture is not supported by this API.")
        model = VisionTransformer(num_classes=len(CLASS_NAMES))
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise RuntimeError(f"{MODEL_LABELS[model_name]} checkpoint class order does not match the API.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(_device).eval()
    _models[model_name] = model
    return model


def preprocess(image: Image.Image, model_name: str) -> torch.Tensor:
    if model_name == "cnn":
        size = CNN_IMAGE_SIZE
        channels = 1
    elif model_name == "efficientnet":
        size = 224
        channels = 3
    else:
        size = VIT_IMAGE_SIZE
        channels = 1
    gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    if channels == 3:
        tensor = tensor.repeat(3, 1, 1)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    else:
        tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


def explanation_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "cnn":
        return model.features[-1]
    if model_name == "efficientnet":
        return model.features[-1]
    return model.patch_embed


def gradcam_overlay(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original: Image.Image,
    class_index: int,
) -> str:
    """Return a Grad-CAM overlay; ViT uses its spatial patch-embedding activations."""
    activations: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        output.retain_grad()
        activations.append(output)

    handle = explanation_layer(model, model_name).register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor.to(_device))
        logits[0, class_index].backward()
        feature_map = activations[0]
        gradients = feature_map.grad
        if gradients is None or feature_map.ndim != 4:
            raise RuntimeError("Could not compute a spatial Grad-CAM explanation.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        if float(cam.max()) > 0:
            cam = cam / float(cam.max())
        colors = (
            np.stack((cam, np.maximum(0, 1 - np.abs(cam - 0.5) * 2), 1 - cam), axis=-1) * 255
        ).astype(np.uint8)
        max_side = 512
        scale = min(1.0, max_side / max(original.size))
        display_size = (max(1, round(original.width * scale)), max(1, round(original.height * scale)))
        heatmap = Image.fromarray(colors, mode="RGB").resize(display_size, Image.Resampling.BILINEAR)
        base = original.convert("RGB").resize(display_size, Image.Resampling.BILINEAR)
        overlay = Image.blend(base, heatmap, alpha=0.38)
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_models": [MODEL_LABELS[name] for name in _models],
        "device": str(_device),
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(default="cnn")):
    model_name = model_name.strip().lower()
    if model_name not in CHECKPOINTS:
        raise HTTPException(status_code=400, detail="Choose CNN, EfficientNet-B0, or ViT-Tiny.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file (JPEG, PNG, or WebP).")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than the 15 MB upload limit.")
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")
    try:
        model = load_model(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    tensor = preprocess(image, model_name).to(_device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    class_index = int(np.argmax(probabilities))
    try:
        explanation = gradcam_overlay(model, model_name, tensor, image, class_index)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create the model explanation: {exc}")
    explanation_name = "Grad-CAM over ViT patch embeddings" if model_name == "vit" else "Grad-CAM"
    return {
        "model": MODEL_LABELS[model_name],
        "predicted_class": CLASS_NAMES[class_index],
        "confidence": float(probabilities[class_index]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        "explanation_type": explanation_name,
        "explanation_image": explanation,
        "disclaimer": DISCLAIMER,
    }
