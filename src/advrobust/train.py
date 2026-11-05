"""Standard and adversarial training.

Three training modes share one loop:

* "standard" -- minimise cross-entropy on clean images (the undefended baseline).
* "fgsm"     -- FGSM adversarial training in the style of Goodfellow et al.
                (2015): loss = 0.5 * CE(clean) + 0.5 * CE(FGSM(clean)).
                Cheap (2x a standard step) but known to be vulnerable to
                gradient masking: the model can learn to make the *one-step*
                attack weak without becoming robust to stronger attacks.
* "pgd"      -- PGD adversarial training (Madry et al., 2018): approximately
                solve the min-max problem
                    min_theta E[ max_{||delta||_inf <= eps} CE(f(x + delta), y) ]
                by training only on PGD adversarial examples (random start,
                `pgd_steps` steps). About (pgd_steps + 1)x the cost.

For both adversarial modes eps is ramped linearly from 0 to its target over
the first `ramp_epochs` epochs. A small network trained at the full eps = 0.3
from random initialisation can stall near chance accuracy; the warm-up (a
common curriculum trick) avoids that and costs nothing at the end of training.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

from .attacks import fgsm, pgd
from .data import iterate_minibatches


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 2000) -> float:
    """Clean accuracy, computed in batches without gradients."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in iterate_minibatches(x, y, batch_size):
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    return correct / len(x)


def adversarial_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, attack: str, eps: float,
                         seed: int = 0, batch_size: int = 1000, **attack_kwargs) -> float:
    """Accuracy on adversarial examples crafted against `model` itself (white-box)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    correct = 0
    for xb, yb in iterate_minibatches(x, y, batch_size):
        if attack == "fgsm":
            xa = fgsm(model, xb, yb, eps)
        elif attack == "pgd":
            xa = pgd(model, xb, yb, eps, generator=gen, **attack_kwargs)
        else:
            raise ValueError(attack)
        with torch.no_grad():
            correct += (model(xa).argmax(dim=1) == yb).sum().item()
    return correct / len(x)


def train_model(model: nn.Module, x_train, y_train, x_test, y_test, *, mode: str = "standard",
                epochs: int = 5, eps: float = 0.3, lr: float = 1e-3, batch_size: int = 128,
                pgd_steps: int = 7, ramp_epochs: float = 1.0, seed: int = 0,
                monitor_x=None, monitor_y=None, log=print) -> list[dict]:
    """Train `model` in place and return one log row per epoch.

    Optimiser: Adam (lr 1e-3), learning rate divided by 10 for the final
    epoch. `monitor_x/monitor_y` is a small fixed test subset on which
    FGSM and PGD-10 accuracy at the target eps are tracked every epoch.
    """
    if mode not in ("standard", "fgsm", "pgd"):
        raise ValueError(mode)
    gen = torch.Generator().manual_seed(seed)          # batch order
    attack_gen = torch.Generator().manual_seed(seed + 1)  # PGD random starts
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = (len(x_train) + batch_size - 1) // batch_size
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == epochs and epochs > 1:
            for group in opt.param_groups:
                group["lr"] = lr * 0.1
        t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for b, (xb, yb) in enumerate(iterate_minibatches(x_train, y_train, batch_size, shuffle=True, generator=gen)):
            progress = (epoch - 1) + b / n_batches  # fractional epochs completed
            eps_now = eps * min(1.0, progress / ramp_epochs) if ramp_epochs > 0 else eps

            if mode == "standard":
                model.train()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            else:
                # Craft the attack with the current weights (eval mode: no
                # parameter gradients are accumulated by the attack itself).
                model.eval()
                if mode == "fgsm":
                    x_adv = fgsm(model, xb, yb, eps_now)
                    model.train()
                    logits = model(x_adv)
                    loss = 0.5 * F.cross_entropy(model(xb), yb) + 0.5 * F.cross_entropy(logits, yb)
                else:
                    x_adv = pgd(model, xb, yb, eps_now, steps=pgd_steps, generator=attack_gen)
                    model.train()
                    logits = model(x_adv)
                    loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            seen += len(xb)

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            # For adversarial modes this is accuracy on the training-time attack.
            "train_acc": correct / seen,
            "test_clean_acc": accuracy(model, x_test, y_test),
        }
        if monitor_x is not None:
            row["monitor_fgsm_acc"] = adversarial_accuracy(model, monitor_x, monitor_y, "fgsm", eps)
            row["monitor_pgd10_acc"] = adversarial_accuracy(model, monitor_x, monitor_y, "pgd", eps,
                                                            seed=seed, steps=10)
        history.append(row)
        log(f"    [{mode}] epoch {epoch}/{epochs}  loss={row['train_loss']:.4f}  "
            f"train_acc={row['train_acc']:.4f}  clean_test={row['test_clean_acc']:.4f}"
            + (f"  fgsm@{eps}={row['monitor_fgsm_acc']:.3f}  pgd10@{eps}={row['monitor_pgd10_acc']:.3f}"
               if monitor_x is not None else "")
            + f"  ({time.time() - t0:.0f}s)")
    model.eval()
    return history
