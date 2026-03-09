import pandas as pd

# csv_files = {
#     "Train": "CLDASEP2/Cephalometric dataset/train_senior.csv",
#     "Test1": "CLDASEP2/Cephalometric dataset/test1_senior.csv",
#     "Test2": "CLDASEP2/Cephalometric dataset/test2_senior.csv"
# }

# summary = []

# for purpose, path in csv_files.items():
#     df = pd.read_csv(path)
#     num_images = len(df)
#     summary.append({"Purpose": purpose, "CSV File": path, "Num Images": num_images})

# # summary_df = pd.DataFrame(summary)
# # print(summary_df)
# from PIL import Image
# import os

# image_folder = "CLDASEP2/Cephalometric dataset/cepha400"
# sample_image_path = os.path.join(image_folder, sample_filename)
# with Image.open(sample_image_path) as img:
#     print(f"Sample image resolution: {img.size}")
from PIL import Image
import numpy as np

def is_left_facing(image):
    # Dummy check: returns True if nose points left (replace with real logic)
    return True

def preprocess(image):
    # Resize and normalize
    image = image.resize((512, 512))
    arr = np.array(image) / 255.0
    return arr

def model_inference(image_arr):
    # Dummy model output: 19 landmarks (x, y) in 512x512 space
    return np.random.rand(19, 2) * 512

def postprocess_landmarks(landmarks, orig_size):
    # Rescale landmarks to original image size
    x_scale = orig_size[0] / 512
    y_scale = orig_size[1] / 512
    return landmarks * [x_scale, y_scale]

def visualize(image, landmarks):
    # Overlay landmarks (dummy visualization)
    print("Landmarks visualized on image.")

# Pipeline
image_path = "CLDASEP2/Cephalometric dataset/cepha400/sample.jpg"
with Image.open("C:\\sem1\\asep\\CLDASEP2_main\\CLDASEP2\\Cephalometric dataset\\cepha400\\301.jpg") as img:
    orig_size = img.size
    if not is_left_facing(img):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_arr = preprocess(img)
    pred_landmarks = model_inference(img_arr)
    final_landmarks = postprocess_landmarks(pred_landmarks, orig_size)
    visualize(img, final_landmarks)