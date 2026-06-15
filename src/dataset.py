"""PyTorch Dataset + augmentation pipeline for fundus images."""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .preprocess import ben_graham

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=25, p=0.6),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])


class RetinopathyDataset(Dataset):
    """Reads fundus images and their diagnosis grade (0-4)."""

    def __init__(self, df: pd.DataFrame, images_dir: str, image_ext: str,
                 image_size: int, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_ext = image_ext
        self.image_size = image_size
        self.tf = build_transforms(image_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.images_dir, f"{row['id_code']}{self.image_ext}")
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = ben_graham(rgb, size=self.image_size)
        img = self.tf(image=img)["image"]
        label = int(row["diagnosis"]) if "diagnosis" in row else -1
        return img, torch.tensor(label, dtype=torch.long)
