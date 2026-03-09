import numpy as np
import matplotlib.pyplot as plt

# Load saved losses
train_loss = np.load("train_losses.npy")
val_loss = np.load("val_losses.npy")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='s')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and/or show the plot
plt.savefig("loss_plot.png", dpi=300)
plt.show()
