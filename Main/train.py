import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CephalometricDataset
from model import CephalometricCNN
import matplotlib.pyplot as plt

# ------------------ Constants ------------------ #
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 512

# ------------------ Correct Paths ------------------ #
# Go one folder up from "Main" to access the dataset folder
base_path = os.path.abspath(os.path.join(os.getcwd(), "..", "Cephalometric dataset"))
image_folder = os.path.join(base_path, "cepha400")
train_csv_path = os.path.join(base_path, "train_senior.csv")
val_csv_path = os.path.join(base_path, "test1_senior.csv")

# Confirm paths
print("Train CSV Path:", train_csv_path)
print("Val CSV Path:", val_csv_path)
print("Image Folder:", image_folder)

# ------------------ Load Dataset ------------------ #
train_dataset = CephalometricDataset(train_csv_path, image_folder, augment=True)
val_dataset = CephalometricDataset(val_csv_path, image_folder, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------ Model Setup ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CephalometricCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------ Visualize Sample ------------------ #
for i in range(3):
    img, lm = train_dataset[i]
    img_np = img.permute(1, 2, 0).numpy()
    plt.imshow(img_np)
    plt.scatter(lm[:, 0], lm[:, 1], c='r', s=10)
    plt.title(f"Augmented Sample #{i}")
    plt.axis("off")
    plt.show()

# ------------------ Training Loop ------------------ #
best_val_loss = float("inf")

for epoch in range(EPOCHS):
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

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, landmarks in val_loader:
            images, landmarks = images.to(device), landmarks.to(device)
            predictions = model(images)
            loss = criterion(predictions, landmarks.view(landmarks.size(0), -1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Model checkpoint saved!")
        
print("🎉 Training completed!")
