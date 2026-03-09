import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps
import random
import torchvision.transforms.functional as TF

# Constants
IMAGE_SIZE = 512
ORIGINAL_SIZE = (2400, 1935)  # Original image dimensions (H, W)

class CephalometricDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None, augment=False):
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_filename = row["image_path"].strip()
        img_path = os.path.join(self.image_folder, img_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file '{img_filename}' not found in {self.image_folder}")

        # Load and resize image
        image = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))

        # Load and scale landmarks
        landmarks = row.iloc[1:].values.astype(np.float32)
        x_coords = landmarks[0::2] * (IMAGE_SIZE / ORIGINAL_SIZE[1])
        y_coords = landmarks[1::2] * (IMAGE_SIZE / ORIGINAL_SIZE[0])
        landmarks_scaled = np.stack([x_coords, y_coords], axis=1)

        # --- AUGMENTATION ---
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                landmarks_scaled[:, 0] = IMAGE_SIZE - landmarks_scaled[:, 0]

            if random.random() < 0.3:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.7, 1.3))

            if random.random() < 0.2:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                # Rotation affects landmarks too — skip landmark rotation for now to keep things simple.

        # --- TRANSFORM TO TENSOR ---
        if self.transform:
            image = self.transform(image)  # torchvision transforms
        else:
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)  # (C, H, W)

        landmarks_tensor = torch.tensor(landmarks_scaled, dtype=torch.float32)

        return image, landmarks_tensor

# Debug block
if __name__ == "__main__":
    dataset_folder = os.path.abspath(os.path.join(os.getcwd(), "..", "Cephalometric dataset"))
    image_folder = os.path.join(dataset_folder, "cepha400")
    csv_path = os.path.join(dataset_folder, "train_senior.csv")

    dataset = CephalometricDataset(csv_path, image_folder, augment=True)
    print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
    
    image_tensor, landmarks_tensor = dataset[0]
    print(f"Image shape: {image_tensor.shape}")
    print(f"Landmarks shape: {landmarks_tensor.shape}")
    print("First sample landmarks:\n", landmarks_tensor.numpy())

