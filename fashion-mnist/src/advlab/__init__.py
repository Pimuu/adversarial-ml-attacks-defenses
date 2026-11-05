"""advlab: a small, reproducible lab for adversarial attacks and defenses.

Modules
-------
config      Experiment configuration (one dataclass, all hyper-parameters).
data        Fashion-MNIST loading into in-memory tensors.
model       The small CNN classifier.
attacks     FGSM and PGD (L-infinity) attacks.
training    Standard, FGSM-adversarial and PGD-adversarial training loops.
evaluation  Clean / robust accuracy, PGD convergence and transfer checks.
plotting    Figures written to results/figures/.
utils       Seeding, device selection, timing and I/O helpers.
"""

__all__ = [
    "config",
    "data",
    "model",
    "attacks",
    "training",
    "evaluation",
    "plotting",
    "utils",
]
