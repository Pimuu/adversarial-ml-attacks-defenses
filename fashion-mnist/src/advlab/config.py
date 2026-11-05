"""Experiment configuration.

Every number that influences the results lives here, so the pipeline is a
pure function of (config, seed). The defaults are the settings used to
produce the committed results in ``results/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    data_dir: str = "data"
    dataset: str = "FashionMNIST"

    # -------------------------------------------------------------- training
    seed: int = 0
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0

    # ---------------------------------------------------- adversarial training
    # Threat model used *during training*: L-inf ball of radius train_eps in
    # [0, 1] pixel space. 0.1 is a common choice for Fashion-MNIST: large
    # enough to break an undefended model, small enough that the images stay
    # recognisable to a human.
    train_eps: float = 0.1
    # PGD adversarial training (Madry et al., 2018): steps and step size.
    train_pgd_steps: int = 7
    train_pgd_alpha: float = 0.025  # = eps / 4, so 7 steps can cross the ball
    # "Fast" FGSM adversarial training (Wong et al., 2020): random start in the
    # ball, then one signed-gradient step of size 1.25 * eps.
    train_fgsm_alpha_scale: float = 1.25

    # ------------------------------------------------------------ evaluation
    # Attack radii evaluated at test time (L-inf, [0, 1] pixel scale).
    eval_eps: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    )
    # PGD evaluation attack: more steps than the training attack, random start.
    eval_pgd_steps: int = 20
    eval_pgd_alpha_scale: float = 2.5  # alpha = 2.5 * eps / steps (Madry et al.)
    eval_batch_size: int = 1000
    # Number of test images used for evaluation (10000 = the full test set).
    n_eval: int = 10000

    # ---------------------------------------------- gradient-masking checks
    check_eps: float = 0.1
    check_n: int = 1000
    check_steps: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    check_restarts: int = 5
    check_restart_steps: int = 20

    # ------------------------------------------------------------- outputs
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    n_example_images: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


# Human-readable names used in tables and figures (fixed order = fixed color).
MODEL_NAMES = {
    "standard": "Standard (undefended)",
    "fgsm_at": "FGSM adversarial training",
    "pgd_at": "PGD adversarial training",
}

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
