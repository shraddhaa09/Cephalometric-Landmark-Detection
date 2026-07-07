# Cephalometric Landmark Detection

A practical deep learning project for predicting 19 anatomical landmarks from cephalometric X-ray images using a ResNet-18-based regression model.

## Problem Statement
Cephalometric landmark detection is a common computer vision task in orthodontics and craniofacial analysis. Automatically localizing anatomical landmarks can support downstream analysis pipelines, but it requires careful preprocessing, model design, and evaluation.

## Technical Approach
This project uses a CNN regression pipeline:
- a pretrained ResNet-18 backbone for feature extraction,
- a regression head that predicts 38 values for 19 landmark coordinates,
- coordinate scaling from the original image size to the model input size,
- visualization overlays for predictions and error vectors,
- Grad-CAM-style explanations for interpretability.

## Features
- ResNet-18 landmark regressor
- Config-driven training pipeline
- Reproducible seed handling
- Multiple loss options: MSE, Smooth L1, Wing Loss
- Properly aligned augmentation for image and coordinates
- Evaluation with MEE, mean radial error, and landmark-wise analysis
- Interactive Streamlit demo

## Project Structure
```text
.
├── app/
├── assets/
├── checkpoints/
├── configs/
├── docs/
├── outputs/
├── runs/
├── src/
│   └── cephalometric_landmark_detection/
├── tests/
└── Main/
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data
The repository expects the dataset under:
```text
Cephalometric dataset/
├── train_senior.csv
├── test1_senior.csv
├── test2_senior.csv
└── cepha400/
```

## Training
```bash
python -m src.cephalometric_landmark_detection.train
```

## Evaluation
```bash
python -m src.cephalometric_landmark_detection.evaluate
```

## Streamlit Demo
```bash
streamlit run app/streamlit_app.py
```

## Results
The repository reports the measured metrics produced by the current evaluation workflow, including MEE and mean radial error. No medical or clinical claims are made.

## Limitations
- The project remains a baseline ML portfolio implementation rather than a clinically validated system.
- The dataset size and training setup are modest.
- Landmark geometry is improved but still simple compared with fully geometry-aware medical imaging pipelines.

## Future Work
- Add transformers or attention-based backbones.
- Explore stronger probabilistic or uncertainty-aware models.
- Use more clinically meaningful validation and evaluation protocols.
- Integrate experiment tracking more deeply.

## Assets
Placeholders and generated outputs are stored in the assets directory.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License
This project is licensed under the MIT License.
