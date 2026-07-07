from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import IMAGE_DIR, OUTPUT_DIR, TEST_CSV, resolve_model_path
from .dataset import CephalometricDataset
from .model import CephalometricCNN
from .utils import mean_euclidean_error, pixel_to_mm, reshape_landmarks


def evaluate_model(model_path: str | Path | None = None, output_dir: str | Path | None = None) -> Dict[str, object]:
    output_dir = Path(output_dir or OUTPUT_DIR / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephalometricCNN().to(device)
    resolved_path = Path(model_path) if model_path is not None else resolve_model_path()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved_path}")
    model.load_state_dict(torch.load(resolved_path, map_location=device))
    model.eval()

    test_dataset = CephalometricDataset(TEST_CSV, IMAGE_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_pred = []
    all_gt = []
    landmark_errors = []
    sample_summaries: List[Dict[str, object]] = []

    with torch.no_grad():
        for idx, (image, true_landmarks) in enumerate(test_loader):
            image = image.to(device)
            true_landmarks = true_landmarks.to(device)
            predicted_landmarks = model(image).view(-1, 2).cpu().numpy()
            true_landmarks_np = true_landmarks.squeeze().cpu().numpy()

            all_pred.append(predicted_landmarks)
            all_gt.append(true_landmarks_np)

            diffs = np.linalg.norm(predicted_landmarks - true_landmarks_np, axis=1)
            landmark_errors.append(diffs)

            sample_summaries.append({
                "index": idx,
                "mean_error_px": float(np.mean(diffs)),
                "max_error_px": float(np.max(diffs)),
                "min_error_px": float(np.min(diffs)),
            })

    all_pred = np.stack(all_pred, axis=0)
    all_gt = np.stack(all_gt, axis=0)

    mee = float(np.mean(np.sqrt(np.sum((all_pred - all_gt) ** 2, axis=2))))
    mean_radial_error = float(np.mean(np.linalg.norm(all_pred - all_gt, axis=2)))
    per_landmark_error = np.mean(np.linalg.norm(all_pred - all_gt, axis=2), axis=0)

    summary = {
        "mee": mee,
        "mean_radial_error": mean_radial_error,
        "per_landmark_error_px": per_landmark_error.tolist(),
        "sample_summaries": sample_summaries,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plt.figure(figsize=(10, 6))
    plt.plot(per_landmark_error, marker="o")
    plt.xticks(range(len(per_landmark_error)))
    plt.xlabel("Landmark index")
    plt.ylabel("Mean error (px)")
    plt.title("Per-landmark mean error")
    plt.tight_layout()
    plt.savefig(output_dir / "per_landmark_error.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(np.concatenate(landmark_errors), bins=30)
    plt.xlabel("Error (px)")
    plt.ylabel("Count")
    plt.title("Distribution of landmark errors")
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png")
    plt.close()

    worst_idx = int(np.argmax(per_landmark_error))
    best_idx = int(np.argmin(per_landmark_error))
    summary["best_landmark"] = best_idx + 1
    summary["worst_landmark"] = worst_idx + 1

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


if __name__ == "__main__":
    evaluate_model()
