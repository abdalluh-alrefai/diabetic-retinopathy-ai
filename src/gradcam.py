"""
Explainability: Grad-CAM heatmaps showing where the model looks.

Usage:
    python -m src.gradcam --ckpt checkpoints/best.pt --image path/to/fundus.png

This is the key "wow" feature for a medical project: it makes the model's
decision interpretable for clinicians instead of a black box.
"""
import argparse

import cv2
import numpy as np
import torch
import yaml
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .dataset import build_transforms
from .model import build_model
from .preprocess import load_and_preprocess


def get_target_layer(model):
    """Return a sensible final conv layer for common timm backbones."""
    if hasattr(model, "conv_head"):          # EfficientNet
        return [model.conv_head]
    if hasattr(model, "stages"):             # ConvNeXt
        return [model.stages[-1]]
    # generic fallback: last module with weights
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return [convs[-1]]


def explain(cfg, ckpt_path, image_path, out_path="gradcam.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["name"], cfg["data"]["num_classes"],
                        pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    size = cfg["data"]["image_size"]
    pre = load_and_preprocess(image_path, size=size)            # RGB uint8
    tf = build_transforms(size, train=False)
    tensor = tf(image=pre)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = int(model(tensor).argmax(1).item())

    cam = GradCAM(model=model, target_layers=get_target_layer(model))
    grayscale = cam(input_tensor=tensor)[0]
    rgb_float = pre.astype(np.float32) / 255.0
    viz = show_cam_on_image(rgb_float, grayscale, use_rgb=True)
    cv2.imwrite(out_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    print(f"Predicted class: {pred} ({cfg['class_names'][pred]})")
    print(f"Saved {out_path}")
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        explain(yaml.safe_load(f), a.ckpt, a.image)
