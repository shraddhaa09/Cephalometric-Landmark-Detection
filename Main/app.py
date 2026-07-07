from app.streamlit_app import *

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((512, 512))
    filename = os.path.basename(uploaded_file.name)

    if is_right_facing(image):
        st.sidebar.warning("🔁 Detected right-facing image. Auto-flipping applied.")
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if enhance:
        image = ImageEnhance.Contrast(image).enhance(1.3)
        image = ImageEnhance.Sharpness(image).enhance(2.0)

    image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_landmarks = model(image_tensor).view(19, 2).cpu().numpy()

    gt_landmarks = get_ground_truth(filename) if show_ground_truth else None

    # 🖼 Image + Prediction Overlay
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🖼 Input X-ray")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📌 Landmarks Overlay")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)

        for i, (x, y) in enumerate(predicted_landmarks):
            ax.scatter(x, y, c='cyan', s=marker_size, edgecolors='black')
            if show_index_labels:
                ax.text(x + 3, y - 3, str(i+1), color='black', fontsize=8)

        if gt_landmarks is not None:
            ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='red', s=marker_size, marker='x', label="Ground Truth")
            if show_error_arrows:
                for i in range(19):
                    arrow = FancyArrowPatch(posA=(gt_landmarks[i, 0], gt_landmarks[i, 1]),
                                            posB=(predicted_landmarks[i, 0], predicted_landmarks[i, 1]),
                                            arrowstyle="->", color="blue", lw=1, mutation_scale=8)
                    ax.add_patch(arrow)

        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.3)

        ax.axis("off")
        ax.legend(loc="upper right")
        st.pyplot(fig)

        # ✅ Display MEE in px and mm
        if gt_landmarks is not None:
            mee_px = mean_euclidean_error(predicted_landmarks, gt_landmarks)
            mee_mm = pixel_to_mm(mee_px)
            st.success(f"✅ Mean Euclidean Error (MEE): {mee_px:.2f} px / {mee_mm:.2f} mm")

            # 📊 Landmark-wise Errors
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

    # 📍 Predicted Coordinates in mm
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📍 Predicted Landmark Coordinates (in mm)")

    predicted_landmarks_mm = np.round(pixel_to_mm(predicted_landmarks), 2)
    coords_df_mm = pd.DataFrame(predicted_landmarks_mm, columns=["X (mm)", "Y (mm)"])
    st.dataframe(coords_df_mm, use_container_width=True)

    # Download option
    csv_download = coords_df_mm.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Predicted Coordinates (mm)", data=csv_download,
                   file_name=f"{filename}_landmarks_mm.csv", mime='text/csv')


    # 🌡 Intensity Histogram
    if show_histogram:
        gray = image.convert("L")
        pixels = np.array(gray).flatten()
        plt.figure(figsize=(6, 2))
        plt.hist(pixels, bins=50, color="gray", edgecolor="black")
        plt.title("Pixel Intensity Histogram")
        st.pyplot(plt)

    # 📊 Quadrant Plot
    if show_quadrant:
        quadrant_counts = get_quadrant_counts(predicted_landmarks)
        plt.figure(figsize=(4, 3))
        plt.bar(quadrant_counts.keys(), quadrant_counts.values(), color=["blue", "green", "orange", "purple"])
        plt.title("Landmark Distribution by Quadrant")
        plt.ylabel("Count")
        st.pyplot(plt)