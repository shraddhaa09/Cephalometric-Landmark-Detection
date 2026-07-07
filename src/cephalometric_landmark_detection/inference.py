from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import IMAGE_SIZE, MODEL_PATH
from .model import CephalometricCNN
from .utils import reshape_landmarks


def predict_landmarks(image_path: str | Path, model_path: str | Path | None = None) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephalometricCNN().to(device)
    checkpoint_path = Path(model_path or MODEL_PATH)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model.eval()

    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)
    return reshape_landmarks(predictions)
