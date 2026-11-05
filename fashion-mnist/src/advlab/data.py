"""Fashion-MNIST loading.

The whole dataset (60k train + 10k test, 28x28 grayscale) is tiny, so it is
loaded once into float tensors in [0, 1]. Keeping pixels in [0, 1] (no
normalisation here) matters: the attack budget epsilon is defined in this
pixel space, and adversarial images must be clipped back into [0, 1]. Input
normalisation is done *inside* the model instead (see ``model.py``).
"""

from __future__ import annotations

import torch
from torchvision import datasets


def _to_tensors(ds) -> tuple[torch.Tensor, torch.Tensor]:
    # ds.data is uint8 [N, 28, 28]; add the channel dim and scale to [0, 1].
    x = ds.data.unsqueeze(1).float().div_(255.0)
    y = ds.targets.clone().long()
    return x, y


def load_fashion_mnist(data_dir: str = "data"):
    """Return (x_train, y_train, x_test, y_test) as CPU tensors."""
    train = datasets.FashionMNIST(data_dir, train=True, download=True)
    test = datasets.FashionMNIST(data_dir, train=False, download=True)
    x_tr, y_tr = _to_tensors(train)
    x_te, y_te = _to_tensors(test)
    return x_tr, y_tr, x_te, y_te


def iterate_minibatches(x, y, batch_size, shuffle=False, generator=None):
    """Yield (xb, yb) mini-batches. Shuffling uses an explicit RNG for reproducibility."""
    n = x.shape[0]
    if shuffle:
        idx = torch.randperm(n, generator=generator)
    else:
        idx = torch.arange(n)
    for start in range(0, n, batch_size):
        b = idx[start : start + batch_size]
        yield x[b], y[b]
