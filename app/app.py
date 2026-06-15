import numpy as np
import streamlit as st
import torch
import timm
import matplotlib.cm as cm
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

IMG_SIZE = 300
CKPT_PATH = "checkpoints/best_b3.pt"
MODEL_NAME = "efficientnet_b3"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

st.set_page_config(page_title="RetinaScan AI", page_icon="👁️", layout="centered")


def ben_graham(pil_img, size=IMG_SIZE, sigma=10):
    img = np.array(pil_img.convert("RGB"))
    gray = img.mean(2)
    mask = gray > 7
    if mask.sum() > 0:
        ys, xs = np.where(mask)
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = np.array(Image.fromarray(img).resize((size, size))).astype(np.float32)
    blur = np.stack([gaussian_filter(img[..., c], sigma=sigma) for c in range(3)], axis=-1)
    out = np.clip(4.0 * img - 4.0 * blur + 128, 0, 255)
    yy, xx = np.ogrid[:size, :size]
    circle = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) <= (size * 0.49) ** 2
    out = out * circle[..., None] + 128 * (~circle[..., None])
    return out.astype(np.uint8)


def to_tensor(rgb_uint8):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()


@st.cache_resource
def load_model():
    import os
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5)
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        state = {k: v.float() for k, v in state.items()}
        model.load_state_dict(state)
    model.eval()
    return model


def grad_cam(model, tensor):
    feats, grads = {}, {}
    target = model.conv_head
    h1 = target.register_forward_hook(lambda m, i, o: feats.__setitem__("v", o))
    h2 = target.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    out = model(tensor)
    cls = int(out.argmax(1))
    model.zero_grad()
    out[0, cls].backward()
    h1.remove(); h2.remove()
    f, g = feats["v"][0], grads["v"][0]
    weights = g.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * f).sum(0))
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()
    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))) / 255.0
    return cam, cls


def overlay(rgb_uint8, cam):
    heat = cm.jet(cam)[..., :3]
    blended = 0.5 * (rgb_uint8.astype(np.float32) / 255.0) + 0.5 * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


st.title("👁️ RetinaScan AI")
st.caption("Diabetic Retinopathy grading from retinal fundus images — "
           "deep learning with explainable Grad-CAM.")

model = load_model()
file = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])

if file:
    pre = ben_graham(Image.open(file))
    tensor = to_tensor(pre)
    tensor.requires_grad_(True)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1)[0].numpy()
    pred = int(probs.argmax())
    cam, _ = grad_cam(model, tensor)
    viz = overlay(pre, cam)
    c1, c2 = st.columns(2)
    c1.image(pre, caption="Preprocessed input", use_container_width=True)
    c2.image(viz, caption="Grad-CAM (model attention)", use_container_width=True)
    st.subheader(f"Prediction: **{CLASS_NAMES[pred]}**  ({probs[pred] * 100:.1f}%)")
    st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(5)})
    st.warning("Research/educational demo only — not a medical diagnosis.")
