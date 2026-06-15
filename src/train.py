"""
Training entry point.

Usage:
    python -m src.train --config config.yaml

Optimises cross-entropy and reports the official APTOS metric:
Quadratic Weighted Kappa (QWK). Saves the best checkpoint by QWK.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RetinopathyDataset
from .model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train() if train else model.eval()
    losses, preds, gts = [], [], []
    scaler_enabled = scaler is not None
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=scaler_enabled):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if train:
                if scaler_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            losses.append(loss.item())
            preds += logits.argmax(1).cpu().tolist()
            gts += labels.cpu().tolist()
    qwk = cohen_kappa_score(gts, preds, weights="quadratic")
    acc = accuracy_score(gts, preds)
    return float(np.mean(losses)), qwk, acc


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(cfg["data"]["csv_path"])
    train_df, val_df = train_test_split(
        df, test_size=cfg["data"]["val_split"],
        stratify=df["diagnosis"], random_state=cfg["train"]["seed"])

    ds_args = dict(images_dir=cfg["data"]["images_dir"],
                   image_ext=cfg["data"]["image_ext"],
                   image_size=cfg["data"]["image_size"])
    train_ds = RetinopathyDataset(train_df, train=True, **ds_args)
    val_ds = RetinopathyDataset(val_df, train=False, **ds_args)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=cfg["train"]["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=cfg["train"]["num_workers"],
                            pin_memory=True)

    model = build_model(cfg["model"]["name"], cfg["data"]["num_classes"],
                        cfg["model"]["pretrained"], cfg["model"]["drop_rate"]).to(device)

    # class weights to fight the heavy imbalance (No-DR dominates)
    counts = train_df["diagnosis"].value_counts().sort_index().values
    weights = torch.tensor(counts.sum() / (len(counts) * counts),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler() if (cfg["train"]["mixed_precision"]
                                             and device.type == "cuda") else None

    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)
    best_qwk, patience = -1.0, 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, tr_qwk, tr_acc = run_epoch(model, train_loader, criterion,
                                            optimizer, device, scaler, True)
        vl_loss, vl_qwk, vl_acc = run_epoch(model, val_loader, criterion,
                                            None, device, None, False)
        scheduler.step()
        print(f"[{epoch:02d}] train loss {tr_loss:.3f} qwk {tr_qwk:.3f} | "
              f"val loss {vl_loss:.3f} qwk {vl_qwk:.3f} acc {vl_acc:.3f}")

        if vl_qwk > best_qwk:
            best_qwk, patience = vl_qwk, 0
            ckpt = os.path.join(cfg["paths"]["checkpoints"], "best.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "qwk": best_qwk}, ckpt)
            print(f"  ✓ saved {ckpt} (qwk={best_qwk:.3f})")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print("Early stopping.")
                break
    print(f"Best validation QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
