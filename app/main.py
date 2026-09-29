"""FastAPI interface for academic brain MRI slice classification."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from src.medivision_cnn import BrainMRICNN, CLASS_NAMES, IMAGE_SIZE


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "cnn_baseline_best.pt"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DISCLAIMER = (
    "Academic prototype for brain MRI image/slice classification only. "
    "Not a clinical diagnostic system; do not use this output for patient-care decisions."
)

app = FastAPI(title="MediVision AI", version="0.1.0", description=DISCLAIMER)
_model: BrainMRICNN | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> BrainMRICNN:
    global _model
    if _model is None:
        if not CHECKPOINT.is_file():
            raise RuntimeError("CNN checkpoint not found. Train the baseline first.")
        checkpoint = torch.load(CHECKPOINT, map_location=_device, weights_only=False)
        if checkpoint.get("architecture") != "BrainMRICNN":
            raise RuntimeError("Checkpoint architecture is not supported by this API.")
        model = BrainMRICNN(num_classes=len(CLASS_NAMES)).to(_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model = model
    return _model


def preprocess(image: Image.Image) -> torch.Tensor:
    gray = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return (tensor - 0.5) / 0.5


def gradcam_overlay(model: BrainMRICNN, input_tensor: torch.Tensor, original: Image.Image, class_index: int) -> str:
    """Compute a Grad-CAM heatmap for the selected class and return a PNG data URI."""
    activations: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        output.retain_grad()
        activations.append(output)

    handle = model.features[-1].register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor.to(_device))
        logits[0, class_index].backward()
        feature_map = activations[0]
        gradients = feature_map.grad
        if gradients is None:
            raise RuntimeError("Could not compute Grad-CAM gradients.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        if float(cam.max()) > 0:
            cam = cam / float(cam.max())
        colors = (np.stack((cam, np.maximum(0, 1 - np.abs(cam - 0.5) * 2), 1 - cam), axis=-1) * 255).astype(np.uint8)
        heatmap = Image.fromarray(colors, mode="RGB").resize((256, 256), Image.Resampling.BILINEAR)
        base = original.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
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
    return {"status": "ok", "model_loaded": _model is not None, "device": str(_device), "disclaimer": DISCLAIMER}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file (JPEG or PNG recommended).")
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
        model = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    tensor = preprocess(image).to(_device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    class_index = int(np.argmax(probabilities))
    try:
        explanation = gradcam_overlay(model, preprocess(image), image, class_index)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create the Grad-CAM explanation: {exc}")
    return {
        "predicted_class": CLASS_NAMES[class_index],
        "confidence": float(probabilities[class_index]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        "explanation_type": "Grad-CAM (visualization of regions influencing this CNN output)",
        "explanation_image": explanation,
        "disclaimer": DISCLAIMER,
    }
