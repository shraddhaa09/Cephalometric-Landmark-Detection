import os
import random
import matplotlib.pyplot as plt
from dataset_loader import CephalometricDataset

dataset_path = os.path.abspath(os.path.join(os.getcwd(), "..", "Cephalometric dataset"))
csv_path = os.path.join(dataset_path, "train_senior.csv")
image_folder = os.path.join(dataset_path, "cepha400")

dataset = CephalometricDataset(csv_path, image_folder, augment=True)

# Pick a fixed index to compare augmentations
index = 0
image_list = []

for i in range(6):  # Show 6 augmented versions
    image_tensor, landmarks = dataset[index]
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_list.append((image_np, landmarks.numpy()))

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (img, lm) in enumerate(image_list):
    ax = axes[i // 3, i % 3]
    ax.imshow(img)
    ax.scatter(lm[:, 0], lm[:, 1], c='r', s=10)
    ax.set_title(f"Augmentation {i+1}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("augmented_samples.png")
plt.show()
