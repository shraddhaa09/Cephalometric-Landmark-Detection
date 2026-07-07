from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .config import IMAGE_SIZE, NUM_LANDMARKS, ORIGINAL_SIZE


class CephalometricDataset(Dataset):
    """Dataset for loading cephalometric images and landmark coordinates."""

    def __init__(self, csv_path: str | Path, image_folder: str | Path, transform=None, augment: bool = False):
        self.csv_path = Path(csv_path)
        self.image_folder = Path(image_folder)
        self.transform = transform
        self.augment = augment

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")

        self.df = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_filename = str(row["image_path"]).strip()
        img_path = self.image_folder / img_filename

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        landmarks = row.iloc[1:].values.astype(np.float32)
        x_coords = landmarks[0::2] * (IMAGE_SIZE / ORIGINAL_SIZE[1])
        y_coords = landmarks[1::2] * (IMAGE_SIZE / ORIGINAL_SIZE[0])
        landmarks_scaled = np.stack([x_coords, y_coords], axis=1)

        if self.augment:
            image, landmarks_scaled = self._augment(image, landmarks_scaled)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image_array = np.array(image, dtype=np.float32) / 255.0
            image = torch.tensor(image_array).permute(2, 0, 1)

        landmarks_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32)
        return image, landmarks_tensor

    @staticmethod
    def _augment(image: Image.Image, landmarks: np.ndarray):
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            landmarks[:, 0] = IMAGE_SIZE - landmarks[:, 0]

        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))

        if random.random() < 0.2:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)

        return image, landmarks
