"""Robustness evaluation: epsilon sweeps, strong-attack checks, transfer attacks.

The checks follow the advice of Carlini et al. (2019), "On Evaluating
Adversarial Robustness", and Athalye et al. (2018), "Obfuscated Gradients
Give a False Sense of Security":

* iterative attacks must be at least as strong as single-step ones
  (PGD accuracy <= FGSM accuracy);
* accuracy must fall towards chance as eps grows large (an unbounded attack
  should always succeed);
* a much stronger attack (more steps, random restarts) should not reveal a
  large hidden drop;
* black-box transfer attacks should not beat white-box attacks (if they do,
  the white-box gradients are probably masked).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .attacks import run_attack
from .data import iterate_minibatches


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def correct_under_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, attack: str, eps: float,
                         seed: int = 0, batch_size: int = 1000, source_model: nn.Module | None = None,
                         **attack_kwargs) -> torch.Tensor:
    """Boolean vector: is each example still classified correctly after the attack?

    If `source_model` is given, adversarial examples are crafted on it and
    then evaluated on `model` (a black-box transfer attack); otherwise the
    attack is white-box against `model` itself.
    """
    model.eval()
    crafter = source_model if source_model is not None else model
    crafter.eval()
    gen = torch.Generator().manual_seed(seed)
    out = []
    for xb, yb in iterate_minibatches(x, y, batch_size):
        xa = run_attack(attack, crafter, xb, yb, eps, generator=gen, **attack_kwargs)
        with torch.no_grad():
            out.append(model(xa).argmax(dim=1) == yb)
    return torch.cat(out)


def evaluate_sweep(models: dict, x: torch.Tensor, y: torch.Tensor, eps_grid: list[float],
                   pgd_steps: int, seed: int, log=print) -> list[dict]:
    """Accuracy of every model under FGSM and PGD at every epsilon (white-box)."""
    rows = []
    for name, model in models.items():
        for attack in ("fgsm", "pgd"):
            for i, eps in enumerate(eps_grid):
                kwargs = {"steps": pgd_steps} if attack == "pgd" else {}
                ok = correct_under_attack(model, x, y, attack if eps > 0 else "none", eps,
                                          seed=seed + 1000 * i, **kwargs)
                k, n = int(ok.sum()), len(ok)
                lo, hi = wilson_interval(k, n)
                rows.append({"model": name, "attack": attack.upper() if attack == "fgsm" else f"PGD-{pgd_steps}",
                             "epsilon": eps, "accuracy": k / n, "ci95_low": lo, "ci95_high": hi, "n": n})
            log(f"    {name:<10} {attack.upper():<4} " + "  ".join(
                f"{r['epsilon']:.2f}:{r['accuracy']:.3f}" for r in rows if r["model"] == name
                and r["attack"].startswith(attack.upper()[:3])))
    return rows
