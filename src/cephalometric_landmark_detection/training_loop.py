from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, CHECKPOINT_DIR, EPOCHS, IMAGE_DIR, LEARNING_RATE, SEED, TRAIN_CSV, VAL_CSV, ensure_directories
from .dataset import CephalometricDataset
from .losses import build_loss
from .model import CephalometricCNN
from .utils import set_seed


def train_with_config(config: Dict[str, object]) -> Dict[str, object]:
    ensure_directories()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephalometricCNN().to(device)
    loss_type = str(config.get("loss_type", "smooth_l1"))
    criterion = build_loss(loss_type)
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("learning_rate", LEARNING_RATE)))

    train_dataset = CephalometricDataset(TRAIN_CSV, IMAGE_DIR, augment=True)
    val_dataset = CephalometricDataset(VAL_CSV, IMAGE_DIR, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=int(config.get("batch_size", BATCH_SIZE)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config.get("batch_size", BATCH_SIZE)), shuffle=False)

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []
    patience = int(config.get("early_stopping_patience", 8))
    epochs_without_improvement = 0

    for epoch in range(int(config.get("epochs", EPOCHS))):
        model.train()
        train_loss = 0.0
        for images, landmarks in train_loader:
            images, landmarks = images.to(device), landmarks.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, landmarks.view(landmarks.size(0), -1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, landmarks in val_loader:
                images, landmarks = images.to(device), landmarks.to(device)
                predictions = model(images)
                loss = criterion(predictions, landmarks.view(landmarks.size(0), -1))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    torch.save(model.state_dict(), CHECKPOINT_DIR / "last_checkpoint.pth")
    with open(CHECKPOINT_DIR / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return {"history": history, "best_val_loss": best_val_loss}
