"""
Streamlit demo: upload a fundus image -> get the DR grade + Grad-CAM heatmap.

Run:
    streamlit run app/app.py
"""
import os
import sys
import tempfile

import numpy as np
import streamlit as st
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import build_transforms          # noqa: E402
from src.model import build_model                 # noqa: E402
from src.preprocess import load_and_preprocess     # noqa: E402

CFG_PATH = "config.yaml"
CKPT_PATH = "checkpoints/best_b3.pt"

st.set_page_config(page_title="RetinaScan AI", page_icon="👁️", layout="centered")


@st.cache_resource
def load_model():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["name"], cfg["data"]["num_classes"],
                        pretrained=False).to(device)
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
    model.eval()
    return model, cfg, device


st.title("👁️ RetinaScan AI")
st.caption("Diabetic Retinopathy grading from retinal fundus images — "
           "deep learning with explainable Grad-CAM.")

model, cfg, device = load_model()
file = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    size = cfg["data"]["image_size"]
    pre = load_and_preprocess(tmp_path, size=size)
    tensor = build_transforms(size, False)(image=pre)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1)[0].cpu().numpy()
    pred = int(probs.argmax())
    names = cfg["class_names"]

    c1, c2 = st.columns(2)
    c1.image(pre, caption="Preprocessed input", use_column_width=True)

    # Grad-CAM (optional, requires grad-cam installed)
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from src.gradcam import get_target_layer
        cam = GradCAM(model=model, target_layers=get_target_layer(model))
        gray = cam(input_tensor=tensor)[0]
        viz = show_cam_on_image(pre.astype(np.float32) / 255.0, gray, use_rgb=True)
        c2.image(viz, caption="Grad-CAM (model attention)", use_column_width=True)
    except Exception as e:  # noqa: BLE001
        c2.info(f"Grad-CAM unavailable: {e}")

    st.subheader(f"Prediction: **{names[pred]}**  ({probs[pred] * 100:.1f}%)")
    st.bar_chart({names[i]: float(probs[i]) for i in range(len(names))})
    st.warning("Research/educational demo only — not a medical diagnosis.")
