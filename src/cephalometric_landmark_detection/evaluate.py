from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .config import OUTPUT_DIR, TEST_CSV, IMAGE_DIR, resolve_model_path
from .dataset import CephalometricDataset
from .model import CephalometricCNN
from .utils import mean_euclidean_error


def evaluate_model(model_path: str | Path | None = None) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephalometricCNN().to(device)
    model_path = Path(model_path) if model_path is not None else resolve_model_path()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = CephalometricDataset(TEST_CSV, IMAGE_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_error = 0.0
    num_samples = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, (image, true_landmarks) in enumerate(test_loader):
        image, true_landmarks = image.to(device), true_landmarks.to(device)
        with torch.no_grad():
            predicted_landmarks = model(image).view(-1, 2)
        error = mean_euclidean_error(predicted_landmarks.cpu().numpy(), true_landmarks.squeeze().cpu().numpy())
        total_error += error
        num_samples += 1

        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        true_lm_np = true_landmarks.cpu().numpy().squeeze()
        pred_lm_np = predicted_landmarks.cpu().numpy()

        plt.imshow(image_np, cmap="gray")
        plt.scatter(true_lm_np[:, 0], true_lm_np[:, 1], c="green", label="Ground Truth", s=20)
        plt.scatter(pred_lm_np[:, 0], pred_lm_np[:, 1], c="red", label="Predicted", s=20)
        plt.title(f"Test Sample {idx + 1}")
        plt.legend()
        plt.axis("off")
        plt.savefig(OUTPUT_DIR / f"output_image_{idx + 1}.png")
        plt.close()

    avg_error = total_error / max(num_samples, 1)
    with open(Path("mee_score.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"{avg_error:.4f}")
    print(f"Mean Euclidean Error on test set: {avg_error:.4f}")
    return avg_error


if __name__ == "__main__":
    evaluate_model()
