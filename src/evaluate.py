"""
Evaluate a trained checkpoint: QWK, accuracy, confusion matrix, report.

Usage:
    python -m src.evaluate --config config.yaml --ckpt checkpoints/best.pt
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import RetinopathyDataset
from .model import build_model


def main(cfg_path, ckpt_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg["data"]["csv_path"])
    _, val_df = train_test_split(df, test_size=cfg["data"]["val_split"],
                                 stratify=df["diagnosis"],
                                 random_state=cfg["train"]["seed"])
    val_ds = RetinopathyDataset(val_df, images_dir=cfg["data"]["images_dir"],
                                image_ext=cfg["data"]["image_ext"],
                                image_size=cfg["data"]["image_size"], train=False)
    loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    model = build_model(cfg["model"]["name"], cfg["data"]["num_classes"],
                        pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            preds += logits.argmax(1).cpu().tolist()
            gts += labels.tolist()

    qwk = cohen_kappa_score(gts, preds, weights="quadratic")
    names = cfg["class_names"]
    print(f"Quadratic Weighted Kappa: {qwk:.4f}\n")
    print(classification_report(gts, preds, target_names=names, digits=3))

    os.makedirs(cfg["paths"]["outputs"], exist_ok=True)
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix (QWK={qwk:.3f})")
    plt.tight_layout()
    out = os.path.join(cfg["paths"]["outputs"], "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    a = ap.parse_args()
    main(a.config, a.ckpt)
