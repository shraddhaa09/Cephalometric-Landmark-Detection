from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .config import NUM_OUTPUTS


class CephalometricCNN(nn.Module):
    """ResNet-18 based regressor for cephalometric landmark detection."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.global_pool(x)
        return self.fc(x)

    def dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        return torch.randn(batch_size, 3, 512, 512)
