"""A small CNN for 28x28 grayscale images.

Architecture (≈ 0.47 M parameters):

    Normalize -> [Conv3x3(32) ReLU Conv3x3(32) ReLU MaxPool2]
              -> [Conv3x3(64) ReLU Conv3x3(64) ReLU MaxPool2]
              -> Flatten -> FC(128) ReLU -> FC(10)

Design choices
--------------
* Normalisation is the first layer, so the model consumes raw [0, 1] pixels.
  Attacks therefore operate directly in pixel space and epsilon has a clear
  meaning (0.1 = 10 % of the full intensity range, i.e. ~25/255).
* No batch-norm and no dropout. Both make the network behave differently in
  train vs. eval mode, which complicates adversarial training (which mode
  should the inner attack use?) without helping much at this scale.
* Small enough to train for 10 epochs on a laptop CPU in a few minutes,
  while still reaching ~92 % clean accuracy on Fashion-MNIST.
"""

from __future__ import annotations

import torch
from torch import nn

# Fashion-MNIST training-set statistics (pixel mean / std in [0, 1]).
FMNIST_MEAN = 0.2860
FMNIST_STD = 0.3530


class Normalize(nn.Module):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.norm = Normalize(FMNIST_MEAN, FMNIST_STD)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(self.norm(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
