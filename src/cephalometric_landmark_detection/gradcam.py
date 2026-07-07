from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .config import IMAGE_SIZE
from .model import CephalometricCNN


def generate_gradcam(model: CephalometricCNN, image_tensor: torch.Tensor, target_index: int = 0) -> np.ndarray:
    model.eval()
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    features = model.backbone(image_tensor)
    activations = features.detach()
    gradients = torch.autograd.grad(torch.sum(activations[:, :, :, :]), image_tensor)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = torch.nn.functional.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam
