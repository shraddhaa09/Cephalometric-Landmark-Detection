import torch
import torch.nn as nn
import torchvision.models as models

class CephalometricCNN(nn.Module):
    def __init__(self):
        super(CephalometricCNN, self).__init__()

        # Load pretrained ResNet18
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


        # Remove the final fully connected layer (used for ImageNet classification)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: (512, 16, 16)

        # Global average pooling to flatten spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (512, 1, 1)

        # Custom regression head: 512 → 38 landmark outputs
        self.fc = nn.Sequential(
            nn.Flatten(),             # (512,)
            nn.Linear(512, 256),      
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 38)        # Final output: 19 (x, y) pairs
        )

    def forward(self, x):
        x = self.backbone(x)          # Extract features
        x = self.global_pool(x)       # Reduce to (B, 512, 1, 1)
        x = self.fc(x)                # Map to landmark coords
        return x

# Test the model
if __name__ == "__main__":
    model = CephalometricCNN()
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expect [1, 38]
