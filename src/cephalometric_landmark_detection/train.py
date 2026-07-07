from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, CHECKPOINT_DIR, EPOCHS, IMAGE_DIR, LEARNING_RATE, SEED, TRAIN_CSV, VAL_CSV, ensure_directories
from .dataset import CephalometricDataset
from .model import CephalometricCNN
from .utils import set_seed


def train_model(output_path: str | Path | None = None, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE) -> None:
    ensure_directories()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephalometricCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = CephalometricDataset(TRAIN_CSV, IMAGE_DIR, augment=True)
    val_dataset = CephalometricDataset(VAL_CSV, IMAGE_DIR, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_model_path = Path(output_path or CHECKPOINT_DIR / "resnet_model.pth")

    for epoch in range(epochs):
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
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved checkpoint to {best_model_path}")

    print("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cephalometric landmark regressor")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "resnet_model.pth"))
    args = parser.parse_args()
    train_model(args.output, args.epochs, args.batch_size, args.lr)
