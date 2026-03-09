import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RegressionTarget

def generate_landmark_heatmaps(model, input_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    
    heatmaps = []
    for i in range(19):
        targets = [RegressionTarget(i)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)
        heatmaps.append(grayscale_cam)

    return np.array(heatmaps)  # Shape: (19, H, W)

def overlay_heatmap_on_image(image, heatmap):
    image_np = np.array(image).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(image_np, heatmap, use_rgb=True)
    return cam_image
