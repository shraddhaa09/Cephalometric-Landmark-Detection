from pathlib import Path

import numpy as np


def generate_landmark_heatmaps(model, input_tensor, target_layer):
    raise NotImplementedError("Grad-CAM support is not enabled in this version.")


def overlay_heatmap_on_image(image, heatmap):
    return np.array(image)
