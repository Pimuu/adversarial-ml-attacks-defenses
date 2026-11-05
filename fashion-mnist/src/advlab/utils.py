"""Small helpers: seeding, device, timing, CSV/JSON output."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> torch.Generator:
    """Seed Python, NumPy and PyTorch; return a dedicated torch RNG for data order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@contextmanager
def timer(label: str, sink: dict | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if sink is not None:
        sink[label] = dt
    print(f"[time] {label}: {dt:.1f}s", flush=True)


def write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
