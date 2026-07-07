from src.cephalometric_landmark_detection.dataset import CephalometricDataset
from src.cephalometric_landmark_detection.config import TRAIN_CSV, IMAGE_DIR

if __name__ == "__main__":
    dataset = CephalometricDataset(TRAIN_CSV, IMAGE_DIR, augment=False)
    image, landmarks = dataset[0]
    print(image.shape)
    print(landmarks.shape)
