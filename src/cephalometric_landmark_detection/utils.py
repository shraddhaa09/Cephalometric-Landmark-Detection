from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import IMAGE_SIZE, NUM_LANDMARKS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reshape_landmarks(predictions: torch.Tensor | np.ndarray) -> np.ndarray:
    array = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else np.asarray(predictions)
    return array.reshape(-1, 2)


def mean_euclidean_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    return float(np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=1))))


def pixel_to_mm(px: float | np.ndarray) -> float | np.ndarray:
    return px * 0.265


def save_plot(path: str | Path, figure=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if figure is None:
        plt.savefig(path)
    else:
        figure.savefig(path)


def ensure_project_root() -> Path:
    return Path(__file__).resolve().parents[2]
