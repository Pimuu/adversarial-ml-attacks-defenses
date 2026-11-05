"""White-box L-infinity evasion attacks: FGSM and PGD.

Threat model shared by both attacks
-----------------------------------
* The attacker knows the model's architecture and weights and can compute
  input gradients (white-box).
* The attacker may change every pixel of a test image by at most `eps`
  (L-infinity ball) and the result must remain a valid image in [0, 1].
* The attack is *untargeted*: any wrong label counts as success.

Both functions return adversarial images; they never modify their inputs and
leave the model's parameters, gradients and train/eval mode untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _input_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gradient of the summed cross-entropy loss w.r.t. the input batch.

    The loss is *summed* (not averaged) so that each example's gradient does
    not depend on the batch size; only its sign/direction is used anyway.
    """
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y, reduction="sum")
    (grad,) = torch.autograd.grad(loss, x)
    return grad.detach()


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    One step of size eps in the direction of the sign of the loss gradient:
        x_adv = clip_[0,1]( x + eps * sign(grad_x L(f(x), y)) )
    It is cheap (one forward/backward pass) but only a first-order
    approximation of the worst case inside the eps-ball.
    """
    if eps == 0:
        return x.clone()
    grad = _input_gradient(model, x, y)
    return (x + eps * grad.sign()).clamp(0.0, 1.0).detach()


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int = 40,
        alpha: float | None = None, random_start: bool = True, restarts: int = 1,
        generator: torch.Generator | None = None) -> torch.Tensor:
    """Projected Gradient Descent on the loss (Madry et al., 2018).

    Iterates FGSM-like steps of size `alpha` and projects back onto the
    eps-ball around x (and the [0, 1] box) after every step:
        x_{t+1} = Proj_{B(x, eps) ∩ [0,1]}( x_t + alpha * sign(grad L(f(x_t), y)) )

    * `random_start` begins from a uniform random point in the eps-ball, which
      avoids starting exactly where a defended model's gradient may be
      uninformative.
    * `restarts` > 1 runs the attack several times from different random
      starts and keeps, per example, the first restart that fools the model
      (or the last one otherwise). This makes the attack strictly stronger.
    * Default step size follows Madry et al.: alpha = 2.5 * eps / steps, so
      the iterate can travel across the whole ball and back.
    """
    if eps == 0:
        return x.clone()
    alpha = 2.5 * eps / steps if alpha is None else alpha
    x = x.detach()
    best = x.clone()
    still_correct = torch.ones(len(x), dtype=torch.bool)

    for _ in range(restarts):
        if random_start:
            noise = torch.empty_like(x).uniform_(-eps, eps, generator=generator)
            x_adv = (x + noise).clamp(0.0, 1.0)
        else:
            x_adv = x.clone()
        for _ in range(steps):
            grad = _input_gradient(model, x_adv, y)
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0).detach()

        # Keep this restart's result for examples not yet fooled.
        with torch.no_grad():
            fooled = model(x_adv).argmax(dim=1) != y
        update = still_correct  # examples where we have not yet succeeded
        best[update] = x_adv[update]
        still_correct = still_correct & ~fooled
        if not still_correct.any():
            break
    return best


def run_attack(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
               generator: torch.Generator | None = None, **kwargs) -> torch.Tensor:
    """Dispatch by name ("none", "fgsm", "pgd") so evaluation code stays generic."""
    if name == "none" or eps == 0:
        return x.clone()
    if name == "fgsm":
        return fgsm(model, x, y, eps)
    if name == "pgd":
        return pgd(model, x, y, eps, generator=generator, **kwargs)
    raise ValueError(f"unknown attack {name!r}")
