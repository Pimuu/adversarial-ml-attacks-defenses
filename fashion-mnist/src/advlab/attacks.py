"""White-box L-infinity evasion attacks: FGSM and PGD.

Threat model (see README for the full statement)
-----------------------------------------------
* White-box: the attacker knows the architecture and weights and can compute
  input gradients of the loss.
* Untargeted: success = the model's prediction differs from the true label.
* Perturbation budget: ||x_adv - x||_inf <= eps, and x_adv in [0, 1].

All functions put the model in eval mode, leave the parameters' gradients
untouched (they use ``torch.autograd.grad`` w.r.t. the input only) and return
a detached tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _input_grad(model, x, y):
    """Gradient of the cross-entropy loss w.r.t. the input x."""
    # enable_grad: attacks must work even if called inside a no_grad() block.
    with torch.enable_grad():
        x = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        (grad,) = torch.autograd.grad(loss, x)
    return grad


def _project(x_adv, x, eps):
    """Project onto the intersection of the eps L-inf ball around x and [0, 1]."""
    x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    return x_adv.clamp_(0.0, 1.0)


def fgsm(model, x, y, eps: float):
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    One step of size eps in the direction of the sign of the loss gradient:
        x_adv = clip_[0,1]( x + eps * sign(grad_x L(f(x), y)) )
    """
    if eps == 0:
        return x.clone().detach()
    was_training = model.training
    model.eval()
    grad = _input_grad(model, x, y)
    x_adv = (x + eps * grad.sign()).clamp_(0.0, 1.0).detach()
    model.train(was_training)
    return x_adv


def pgd(
    model,
    x,
    y,
    eps: float,
    alpha: float | None = None,
    steps: int = 20,
    random_start: bool = True,
    restarts: int = 1,
    generator: torch.Generator | None = None,
):
    """Projected Gradient Descent (Madry et al., 2018), L-inf version.

    Iterates FGSM-like steps of size ``alpha`` and projects back onto the
    eps-ball after every step:
        x_{t+1} = Proj_{B_eps(x) ∩ [0,1]}( x_t + alpha * sign(grad_x L(f(x_t), y)) )

    Args:
        alpha: step size; defaults to 2.5 * eps / steps (Madry et al.).
        random_start: start from a uniform random point in the eps-ball.
        restarts: number of independent random starts. For each example the
            first restart that causes a misclassification is kept, so the
            attack with R restarts is at least as strong as with 1.
        generator: optional RNG for the random start (reproducibility).
    """
    if eps == 0:
        return x.clone().detach()
    if alpha is None:
        alpha = 2.5 * eps / steps
    was_training = model.training
    model.eval()

    x = x.detach()
    best = x.clone()
    # Tracks which examples have already been fooled by some restart.
    fooled = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    for _ in range(restarts):
        if random_start:
            noise = torch.rand(x.shape, generator=generator, device="cpu").to(x.device)
            x_adv = _project(x + (2 * noise - 1) * eps, x, eps)
        else:
            x_adv = x.clone()
        for _ in range(steps):
            grad = _input_grad(model, x_adv, y)
            x_adv = _project(x_adv.detach() + alpha * grad.sign(), x, eps)

        with torch.no_grad():
            wrong = model(x_adv).argmax(1) != y
        # Keep this restart's result for examples not fooled so far: if it fools
        # them, great; if not, it is as good as anything we had.
        take = ~fooled
        best[take] = x_adv[take]
        fooled |= wrong

    model.train(was_training)
    return best.detach()


ATTACKS = {"FGSM": fgsm, "PGD": pgd}
