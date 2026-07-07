import unittest
from pathlib import Path

import torch

from src.cephalometric_landmark_detection.dataset import CephalometricDataset
from src.cephalometric_landmark_detection.model import CephalometricCNN


class PipelineSmokeTests(unittest.TestCase):
    def test_model_output_shape(self):
        model = CephalometricCNN()
        dummy_input = model.dummy_input(batch_size=1)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 38))

    def test_dataset_returns_expected_shapes(self):
        csv_path = Path("Cephalometric dataset/train_senior.csv")
        image_folder = Path("Cephalometric dataset/cepha400")
        dataset = CephalometricDataset(csv_path, image_folder, augment=False)
        image, landmarks = dataset[0]
        self.assertEqual(image.shape[0], 3)
        self.assertEqual(image.shape[1], 512)
        self.assertEqual(image.shape[2], 512)
        self.assertEqual(landmarks.shape, (19, 2))


if __name__ == "__main__":
    unittest.main()
