from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms.functional as TF

from .config import IMAGE_SIZE


def apply_augmentation(image: Image.Image, landmarks: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
    image_copy = image.copy()
    landmarks_copy = landmarks.copy()

    if random.random() < 0.5:
        image_copy = ImageOps.mirror(image_copy)
        landmarks_copy[:, 0] = IMAGE_SIZE - landmarks_copy[:, 0]

    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(image_copy)
        image_copy = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.2:
        angle = random.uniform(-15, 15)
        image_copy = TF.rotate(image_copy, angle)
        radians = np.deg2rad(angle)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        center = IMAGE_SIZE / 2
        x = landmarks_copy[:, 0] - center
        y = landmarks_copy[:, 1] - center
        landmarks_copy[:, 0] = x * cos_a - y * sin_a + center
        landmarks_copy[:, 1] = x * sin_a + y * cos_a + center

    return image_copy, landmarks_copy


def save_augmentation_examples(image: Image.Image, landmarks: np.ndarray, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, title, sample in [(axes[0], "Before augmentation", (image, landmarks)), (axes[1], "After augmentation", apply_augmentation(image, landmarks))]:
        sample_image, sample_landmarks = sample
        ax.imshow(sample_image)
        ax.scatter(sample_landmarks[:, 0], sample_landmarks[:, 1], c="red", s=8)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "augmentation_example.png")
    plt.close(fig)
