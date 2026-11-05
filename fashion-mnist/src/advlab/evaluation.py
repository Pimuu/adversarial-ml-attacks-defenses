"""Evaluation: clean and robust accuracy, plus sanity checks for the defense.

Robust accuracy at radius eps = fraction of test images that are still
classified correctly after the attack. Every number reported in the paper is
produced by the functions in this file.
"""

from __future__ import annotations

import torch

from .attacks import fgsm, pgd
from .config import Config
from .data import iterate_minibatches


def predict(model, x, device, batch_size=1000) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            out.append(model(x[start:start + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(out)


def robust_accuracy(model, x, y, attack: str, eps: float, cfg: Config, device,
                    generator=None, steps=None, restarts=1, return_adv=False):
    """Accuracy of ``model`` on (x, y) under ``attack`` at radius ``eps``."""
    model.eval()
    correct = 0
    advs = [] if return_adv else None
    steps = steps or cfg.eval_pgd_steps
    for xb, yb in iterate_minibatches(x, y, cfg.eval_batch_size):
        xb, yb = xb.to(device), yb.to(device)
        if attack == "FGSM":
            x_adv = fgsm(model, xb, yb, eps)
        elif attack == "PGD":
            alpha = cfg.eval_pgd_alpha_scale * eps / steps
            x_adv = pgd(model, xb, yb, eps, alpha=alpha, steps=steps,
                        random_start=True, restarts=restarts, generator=generator)
        else:
            raise ValueError(attack)
        # Sanity: the attack must respect the threat model.
        assert (x_adv - xb).abs().max().item() <= eps + 1e-6
        assert x_adv.min().item() >= 0.0 and x_adv.max().item() <= 1.0
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == yb).sum().item()
        if return_adv:
            advs.append(x_adv.cpu())
    acc = correct / x.shape[0]
    if return_adv:
        return acc, torch.cat(advs)
    return acc


def eps_sweep(models: dict, x, y, cfg: Config, device, generator):
    """Accuracy of every model under every attack at every eps (long format)."""
    rows = []
    for name, model in models.items():
        for attack in ("FGSM", "PGD"):
            for eps in cfg.eval_eps:
                acc = robust_accuracy(model, x, y, attack, eps, cfg, device, generator)
                rows.append({"model": name, "attack": attack, "epsilon": eps,
                             "accuracy": round(acc, 4), "n": int(x.shape[0])})
                print(f"  {name:9s} {attack:4s} eps={eps:<6} acc={acc:.4f}", flush=True)
    return rows


def pgd_convergence(models: dict, x, y, cfg: Config, device, generator):
    """Gradient-masking check 1: robust accuracy vs. number of PGD steps.

    A genuinely robust model shows accuracy that *plateaus* as steps increase;
    a model that merely masks gradients typically keeps dropping (or shows
    FGSM being stronger than PGD).
    Also runs PGD with several random restarts (check 2).
    """
    rows = []
    eps = cfg.check_eps
    for name, model in models.items():
        for steps in cfg.check_steps:
            acc = robust_accuracy(model, x, y, "PGD", eps, cfg, device, generator, steps=steps)
            rows.append({"model": name, "eps": eps, "steps": steps, "restarts": 1,
                         "accuracy": round(acc, 4), "n": int(x.shape[0])})
            print(f"  {name:9s} PGD-{steps:<3d} eps={eps} acc={acc:.4f}", flush=True)
        acc = robust_accuracy(model, x, y, "PGD", eps, cfg, device, generator,
                              steps=cfg.check_restart_steps, restarts=cfg.check_restarts)
        rows.append({"model": name, "eps": eps, "steps": cfg.check_restart_steps,
                     "restarts": cfg.check_restarts, "accuracy": round(acc, 4),
                     "n": int(x.shape[0])})
        print(f"  {name:9s} PGD-{cfg.check_restart_steps} x{cfg.check_restarts} restarts "
              f"eps={eps} acc={acc:.4f}", flush=True)
    return rows


def transfer_attack(source_name: str, models: dict, x, y, eps_list, cfg: Config,
                    device, generator):
    """Gradient-masking check 3 / black-box threat: transfer attacks.

    Craft PGD examples on ``source`` and evaluate every model on them. If a
    defended model were robust only because its own gradients are uninformative,
    transferred (black-box) examples would hurt it *more* than white-box ones.
    """
    rows = []
    src = models[source_name]
    for eps in eps_list:
        _, x_adv = robust_accuracy(src, x, y, "PGD", eps, cfg, device, generator,
                                   return_adv=True)
        for name, model in models.items():
            acc = (predict(model, x_adv, device) == y).float().mean().item()
            rows.append({"source": source_name, "target": name, "epsilon": eps,
                         "accuracy": round(acc, 4), "n": int(x.shape[0])})
            print(f"  transfer {source_name}->{name:9s} eps={eps} acc={acc:.4f}", flush=True)
    return rows


def per_class_accuracy(model, x, y, x_adv, device, num_classes=10):
    """Clean and adversarial accuracy per true class."""
    pc = predict(model, x, device)
    pa = predict(model, x_adv, device)
    out = []
    for c in range(num_classes):
        m = y == c
        out.append(((pc[m] == c).float().mean().item(), (pa[m] == c).float().mean().item()))
    return out
