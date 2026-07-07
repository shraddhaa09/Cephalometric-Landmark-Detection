from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "Cephalometric dataset"
IMAGE_DIR = DATA_DIR / "cepha400"
TRAIN_CSV = DATA_DIR / "train_senior.csv"
VAL_CSV = DATA_DIR / "test1_senior.csv"
TEST_CSV = DATA_DIR / "test2_senior.csv"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"

IMAGE_SIZE = 512
ORIGINAL_SIZE = (2400, 1935)
NUM_LANDMARKS = 19
NUM_OUTPUTS = NUM_LANDMARKS * 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42
MODEL_PATH = CHECKPOINT_DIR / "resnet_model.pth"
LEGACY_MODEL_PATH = ROOT_DIR / "Main" / "resnet_model.pth"


def ensure_directories() -> None:
    for path in [CHECKPOINT_DIR, OUTPUT_DIR, ASSETS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_model_path(candidate: str | os.PathLike | None = None) -> Path:
    candidates = [Path(candidate)] if candidate is not None else []
    candidates.extend([MODEL_PATH, LEGACY_MODEL_PATH])
    for path in candidates:
        if path and path.exists():
            return path
    return MODEL_PATH
