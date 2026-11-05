"""A small convolutional network for MNIST.

Architecture (about 225k parameters):

    conv 3x3, 32  -> ReLU -> max-pool 2
    conv 3x3, 64  -> ReLU -> max-pool 2
    flatten (64 * 5 * 5) -> linear 128 -> ReLU -> linear 10 (logits)

It is deliberately small so that standard training takes well under a minute
on a laptop-class CPU, and PGD adversarial training (about 8x the cost of a
standard epoch) still fits in a cloud session. It is smaller than the network
used by Madry et al. (2018), which matters for robustness: adversarial
training benefits from extra capacity. This trade-off is discussed in the paper.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 28 -> 26
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 26 -> 13
            nn.Conv2d(32, 64, kernel_size=3),  # 13 -> 11
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 11 -> 5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, 28, 28) in [0, 1]. Returns unnormalised logits (N, 10)."""
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
