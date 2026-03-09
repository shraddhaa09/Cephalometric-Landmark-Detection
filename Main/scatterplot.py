import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ---------- Step 1: Load the CSV files ----------
gt_df = pd.read_csv("C:/sem1/asep/CLDASEP2_main/CLDASEP2/Cephalometric dataset/test2_senior.csv")
pred_df = pd.read_csv("C:/Users/hp/Downloads/304.jpg_landmarks_mm.csv")

# ---------- Step 2: Filter row for image 301 ----------
gt_row = gt_df[gt_df["image_path"].str.contains("304.jpg", case=False)].reset_index(drop=True)


# ---------- Step 3: Reshape to (19, 2) for (x, y) ----------
gt_coords = gt_row.iloc[0, 1:].values.astype(float).reshape(-1, 2)
pred_coords = pred_df.values.astype(float)        # Assuming no header, shape is (19, 2)

# Scale ground truth from original size to 512x512
x_scale = 512 / 1935
y_scale = 512 / 2400
gt_coords_512 = gt_coords.copy()
gt_coords_512[:, 0] *= x_scale
gt_coords_512[:, 1] *= y_scale

# Convert predicted from mm to pixels in 512x512
pred_coords_px = pred_coords / 0.265

# ---------- Step 4: Plot scatter plot ----------
plt.figure(figsize=(6, 6))
plt.scatter(gt_coords_512[:, 0], gt_coords_512[:, 1], c='blue', label='Ground Truth', marker='x')
plt.scatter(pred_coords_px[:, 0], pred_coords_px[:, 1], c='red', label='Predicted', marker='o')

# ---------- Step 5: Add lines between GT and Pred ----------
for i in range(19):
    plt.plot([gt_coords_512[i, 0], pred_coords_px[i, 0]], [gt_coords_512[i, 1], pred_coords_px[i, 1]], 'gray', linestyle='--', linewidth=0.8)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter Plot: Ground Truth vs Predicted Landmarks (304.jpg)')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()  # Optional: match image orientation
plt.tight_layout()
plt.show()
