from __future__ import annotations

import torch
import torch.nn as nn


class WingLoss(nn.Module):
    """Wing loss for robust landmark regression."""

    def __init__(self, w: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.w * torch.log(1 + self.w / self.epsilon),
        )
        return loss.mean()


def build_loss(loss_type: str) -> nn.Module:
    loss_type = (loss_type or "mse").lower()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_type == "wing":
        return WingLoss()
    raise ValueError(f"Unsupported loss type: {loss_type}")
