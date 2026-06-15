"""
FastAPI inference service.

Run:
    uvicorn app.api:app --reload --port 8000

POST an image to /predict and get JSON with grade + probabilities.
    curl -F "file=@fundus.png" http://localhost:8000/predict
"""
import io
import os
import sys

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, UploadFile
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import build_transforms          # noqa: E402
from src.model import build_model                 # noqa: E402
from src.preprocess import ben_graham              # noqa: E402

CFG_PATH = "config.yaml"
CKPT_PATH = "checkpoints/best_b3.pt"

app = FastAPI(title="RetinaScan AI", version="1.0.0")
_model = _cfg = _device = None


def _ensure_loaded():
    global _model, _cfg, _device
    if _model is not None:
        return
    with open(CFG_PATH) as f:
        _cfg = yaml.safe_load(f)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = build_model(_cfg["model"]["name"], _cfg["data"]["num_classes"],
                         pretrained=False).to(_device)
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=_device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        _model.load_state_dict(state)
    _model.eval()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _ensure_loaded()
    raw = await file.read()
    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    size = _cfg["data"]["image_size"]
    pre = ben_graham(img, size=size)
    tensor = build_transforms(size, False)(image=pre)["image"].unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), 1)[0].cpu().numpy()
    pred = int(probs.argmax())
    names = _cfg["class_names"]
    return {
        "grade": pred,
        "label": names[pred],
        "confidence": float(probs[pred]),
        "probabilities": {names[i]: float(probs[i]) for i in range(len(names))},
    }
