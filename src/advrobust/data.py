"""MNIST loading and reproducibility helpers.

The whole dataset is small (60k + 10k images of 28x28), so it is loaded once
into in-memory float tensors with pixel values in [0, 1]. No mean/std
normalisation is applied: keeping inputs in raw pixel space means an
L-infinity budget epsilon has its usual meaning ("each pixel may change by at
most epsilon") and the valid input domain is simply the box [0, 1]^784.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

CLASSES = [str(i) for i in range(10)]


def set_seed(seed: int) -> None:
    """Seed every RNG we use and force deterministic CPU kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def load_mnist(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_train, y_train, x_test, y_test); x has shape (N, 1, 28, 28) in [0, 1].

    torchvision downloads the four IDX files (~11 MB) on first use.
    """
    train = datasets.MNIST(root=str(data_dir), train=True, download=True)
    test = datasets.MNIST(root=str(data_dir), train=False, download=True)

    def to_tensors(ds):
        x = ds.data.unsqueeze(1).float().div_(255.0)
        y = ds.targets.long()
        return x, y

    return (*to_tensors(train), *to_tensors(test))


def iterate_minibatches(x: torch.Tensor, y: torch.Tensor, batch_size: int,
                        shuffle: bool = False, generator: torch.Generator | None = None):
    """Yield (xb, yb) mini-batches from in-memory tensors (faster than a DataLoader here)."""
    n = len(x)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y[idx]
