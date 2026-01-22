import torch
from torch import nn


class SpectrogramCNN(nn.Module):
    """CNN for spectrograms that outputs class logits."""

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input with shape (N, C, X, Y), got {tuple(x.shape)}")
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(num_classes: int, in_channels: int = 1, dropout: float = 0.3) -> SpectrogramCNN:
    """Convenience factory for the spectrogram CNN."""
    return SpectrogramCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
