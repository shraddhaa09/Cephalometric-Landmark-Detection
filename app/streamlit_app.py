from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageEnhance

from src.cephalometric_landmark_detection.config import IMAGE_SIZE, TEST_CSV, resolve_model_path
from src.cephalometric_landmark_detection.gradcam import generate_gradcam
from src.cephalometric_landmark_detection.model import CephalometricCNN
from src.cephalometric_landmark_detection.utils import mean_euclidean_error, pixel_to_mm

st.set_page_config(page_title="Cephalometric Landmark Detector", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 Cephalometric Landmark Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid gray;'>", unsafe_allow_html=True)


@st.cache_resource
def load_model() -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephalometricCNN().to(device)
    checkpoint_path = resolve_model_path()
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        st.warning("No trained checkpoint found. Train the model first.")
    model.eval()
    return model


@st.cache_data
def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def is_right_facing(image: Image.Image) -> bool:
    gray = image.convert("L")
    img_np = np.array(gray)
    w = img_np.shape[1]
    return img_np[:, w // 2 :].mean() > img_np[:, : w // 2].mean()


def get_ground_truth(filename: str, ground_truth_df: pd.DataFrame) -> np.ndarray | None:
    matches = ground_truth_df[ground_truth_df["image_path"].astype(str).str.contains(filename, case=False)]
    if matches.empty:
        return None
    coords = matches.iloc[0, 1:].values.astype(np.float32).reshape(-1, 2)
    x_scale = IMAGE_SIZE / 1935
    y_scale = IMAGE_SIZE / 2400
    coords[:, 0] *= x_scale
    coords[:, 1] *= y_scale
    return coords


def get_quadrant_counts(landmarks: np.ndarray) -> dict[str, int]:
    center = np.array([IMAGE_SIZE / 2, IMAGE_SIZE / 2])
    q = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for x, y in landmarks:
        if x > center[0] and y < center[1]:
            q["Q1"] += 1
        elif x < center[0] and y < center[1]:
            q["Q2"] += 1
        elif x < center[0] and y > center[1]:
            q["Q3"] += 1
        else:
            q["Q4"] += 1
    return q


model = load_model()
ground_truth_df = load_ground_truth()

st.sidebar.header("🧪 Demo Controls")
st.sidebar.caption("Upload an image and inspect landmark predictions and model explanations.")
uploaded_file = st.sidebar.file_uploader("📤 Upload a Cephalometric X-ray", type=["jpg", "png", "bmp"])
show_grid = st.sidebar.checkbox("🧱 Show Grid Overlay", value=True)
marker_size = st.sidebar.slider("🎯 Landmark Size", 10, 100, 40)
enhance = st.sidebar.checkbox("✨ Enhance Image", value=False)
show_ground_truth = st.sidebar.checkbox("🧩 Show Ground Truth Landmarks", value=True)
show_error_arrows = st.sidebar.checkbox("🎯 Show Error Arrows", value=True)
show_index_labels = st.sidebar.checkbox("🔢 Show Landmark Index", value=True)
show_histogram = st.sidebar.checkbox("🌡 Show Intensity Histogram", value=True)
show_quadrant = st.sidebar.checkbox("📊 Show Quadrant Distribution", value=True)
show_gradcam = st.sidebar.checkbox("🔥 Show Grad-CAM", value=True)

if uploaded_file is not None:
    with st.spinner("Running inference..."):
        image = Image.open(uploaded_file).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        filename = os.path.basename(uploaded_file.name)

        if is_right_facing(image):
            st.sidebar.warning("🔁 Detected right-facing image. Auto-flipping applied.")
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if enhance:
            image = ImageEnhance.Contrast(image).enhance(1.3)
            image = ImageEnhance.Sharpness(image).enhance(2.0)

        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(next(model.parameters()).device)

        with torch.no_grad():
            predicted_landmarks = model(image_tensor).view(-1, 2).cpu().numpy()

    gt_landmarks = get_ground_truth(filename, ground_truth_df) if show_ground_truth else None

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🖼 Input X-ray")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📌 Landmarks Overlay")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)

        for i, (x, y) in enumerate(predicted_landmarks):
            ax.scatter(x, y, c="cyan", s=marker_size, edgecolors="black")
            if show_index_labels:
                ax.text(x + 3, y - 3, str(i + 1), color="black", fontsize=8)

        if gt_landmarks is not None:
            ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c="red", s=marker_size, marker="x", label="Ground Truth")
            if show_error_arrows:
                for i in range(len(predicted_landmarks)):
                    ax.annotate(
                        "",
                        xy=(predicted_landmarks[i, 0], predicted_landmarks[i, 1]),
                        xytext=(gt_landmarks[i, 0], gt_landmarks[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1),
                    )

        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.3)

        ax.axis("off")
        ax.legend(loc="upper right")
        st.pyplot(fig)

        if gt_landmarks is not None:
            mee_px = mean_euclidean_error(predicted_landmarks, gt_landmarks)
            mee_mm = pixel_to_mm(mee_px)
            st.success(f"✅ Mean Euclidean Error (MEE): {mee_px:.2f} px / {mee_mm:.2f} mm")

            diffs = np.linalg.norm(predicted_landmarks - gt_landmarks, axis=1)
            diffs_mm = pixel_to_mm(diffs)
            landmark_data = {
                "Index": np.arange(1, 20),
                "GT X": np.round(gt_landmarks[:, 0], 2),
                "GT Y": np.round(gt_landmarks[:, 1], 2),
                "Pred X": np.round(predicted_landmarks[:, 0], 2),
                "Pred Y": np.round(predicted_landmarks[:, 1], 2),
                "Error (px)": np.round(diffs, 2),
                "Error (mm)": np.round(diffs_mm, 2),
            }
            st.markdown("### 🧾 Per-Landmark Error Summary")
            st.dataframe(pd.DataFrame(landmark_data), use_container_width=True)

    if show_gradcam:
        st.subheader("🔥 Grad-CAM Explanation")
        cam = generate_gradcam(model, image_tensor)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.array(image), cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.35)
        ax.axis("off")
        st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📍 Predicted Landmark Coordinates (in mm)")
    predicted_landmarks_mm = np.round(pixel_to_mm(predicted_landmarks), 2)
    coords_df_mm = pd.DataFrame(predicted_landmarks_mm, columns=["X (mm)", "Y (mm)"])
    st.dataframe(coords_df_mm, use_container_width=True)

    csv_download = coords_df_mm.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Predicted Coordinates (mm)",
        data=csv_download,
        file_name=f"{filename}_landmarks_mm.csv",
        mime="text/csv",
    )

    if show_histogram:
        gray = image.convert("L")
        pixels = np.array(gray).flatten()
        plt.figure(figsize=(6, 2))
        plt.hist(pixels, bins=50, color="gray", edgecolor="black")
        plt.title("Pixel Intensity Histogram")
        st.pyplot(plt)

    if show_quadrant:
        quadrant_counts = get_quadrant_counts(predicted_landmarks)
        plt.figure(figsize=(4, 3))
        plt.bar(quadrant_counts.keys(), quadrant_counts.values(), color=["blue", "green", "orange", "purple"])
        plt.title("Landmark Distribution by Quadrant")
        plt.ylabel("Count")
        st.pyplot(plt)
else:
    st.info("Upload a cephalometric X-ray to begin inference.")
