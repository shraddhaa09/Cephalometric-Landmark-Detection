import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Constants
IMAGE_SIZE = 512
ORIGINAL_SIZE = (2400, 1935)  # Original image dimensions (H, W)

# Define base project directory
base_dir = os.getcwd()

# Define correct paths
dataset_folder = os.path.join(base_dir, "Cephalometric dataset")  # CSV files
image_folder = os.path.join(dataset_folder, "cepha400")  # Image files (Adjusted path!)

# Load CSV
csv_path = os.path.join(dataset_folder, "train_senior.csv")
df = pd.read_csv(csv_path)

# Debugging: Check if CSV loaded correctly
print(f"CSV successfully loaded. Shape: {df.shape}")

# Select the first row
row = df.iloc[0]
img_filename = row["image_path"].strip()  # Fetch filename exactly as stored
img_path = os.path.join(image_folder, img_filename)

# Debugging: Check the expected image path
print(f"Looking for image at: {img_path}")

# Verify if image exists
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file '{img_filename}' not found in {image_folder}")

# Load and resize image
image = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))

# Extract and scale landmarks
landmarks = row.iloc[1:].values.astype(np.float32)
x_coords = landmarks[0::2] * (IMAGE_SIZE / ORIGINAL_SIZE[1])
y_coords = landmarks[1::2] * (IMAGE_SIZE / ORIGINAL_SIZE[0])
landmarks_scaled = np.stack([x_coords, y_coords], axis=1)

# Plot image with landmarks
plt.imshow(image)
plt.scatter(landmarks_scaled[:, 0], landmarks_scaled[:, 1], c='red', s=10)
plt.title(f"Sample: {img_filename} with Landmarks")
plt.axis('off')
plt.show()