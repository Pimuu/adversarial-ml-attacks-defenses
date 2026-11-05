"""Training loops: standard ERM and two adversarial-training defenses.

All three share one loop; they differ only in how the training batch is
transformed before the gradient step:

* ``standard``  – train on clean images (empirical risk minimisation).
* ``fgsm_at``   – "fast" adversarial training (Wong, Rice & Kolter, 2020):
                  random start in the eps-ball + one FGSM step of 1.25*eps,
                  projected back to the ball. Costs ~2x a standard epoch.
* ``pgd_at``    – PGD adversarial training (Madry et al., 2018): solve the
                  inner maximisation with K-step PGD and train on the result,
                  i.e. minimise  E[ max_{||d||<=eps} L(f(x+d), y) ].
                  Costs ~(K+1)x a standard epoch.

Following Madry et al., adversarial training uses 100 % adversarial batches
(no clean/adversarial mixing).
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .attacks import _input_grad, _project, pgd
from .config import Config
from .data import iterate_minibatches


def fgsm_rs(model, x, y, eps, alpha, generator=None):
    """FGSM with a uniform random start (the "fast AT" inner attack)."""
    was_training = model.training
    model.eval()
    noise = torch.rand(x.shape, generator=generator).to(x.device)
    x0 = _project(x + (2 * noise - 1) * eps, x, eps)
    grad = _input_grad(model, x0, y)
    x_adv = _project(x0 + alpha * grad.sign(), x, eps).detach()
    model.train(was_training)
    return x_adv


def make_batch_transform(method: str, cfg: Config, generator: torch.Generator):
    """Return f(model, x, y) -> x_train for the given training method."""
    eps = cfg.train_eps
    if method == "standard":
        return lambda model, x, y: x
    if method == "fgsm_at":
        alpha = cfg.train_fgsm_alpha_scale * eps
        return lambda model, x, y: fgsm_rs(model, x, y, eps, alpha, generator)
    if method == "pgd_at":
        return lambda model, x, y: pgd(
            model, x, y, eps, alpha=cfg.train_pgd_alpha, steps=cfg.train_pgd_steps,
            random_start=True, generator=generator,
        )
    raise ValueError(f"unknown training method: {method}")


@torch.no_grad()
def clean_accuracy(model, x, y, device, batch_size=1000) -> float:
    model.eval()
    correct = 0
    for xb, yb in iterate_minibatches(x, y, batch_size):
        correct += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item()
    return correct / x.shape[0]


def train_model(model, method, x_train, y_train, x_val, y_val, cfg: Config,
                device, generator: torch.Generator):
    """Train ``model`` in place with the chosen method.

    Returns a list of per-epoch log dicts (loss, train acc on the batches the
    model actually saw, clean validation accuracy, wall-clock time).
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine decay to 0 over the run: stable, no extra hyper-parameters.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    transform = make_batch_transform(method, cfg, generator)
    log = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for xb, yb in iterate_minibatches(x_train, y_train, cfg.batch_size,
                                          shuffle=True, generator=generator):
            xb, yb = xb.to(device), yb.to(device)
            x_in = transform(model, xb, yb)  # clean or adversarial batch
            logits = model(x_in)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_loss += loss.item() * yb.shape[0]
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n += yb.shape[0]
        sched.step()
        dt = time.perf_counter() - t0
        val_acc = clean_accuracy(model, x_val, y_val, device)
        entry = {
            "method": method,
            "epoch": epoch,
            "train_loss": round(tot_loss / tot_n, 5),
            # For AT methods this is accuracy on the *adversarial* training batches.
            "train_acc": round(tot_correct / tot_n, 5),
            "test_clean_acc": round(val_acc, 5),
            "epoch_seconds": round(dt, 2),
        }
        log.append(entry)
        print(f"  [{method}] epoch {epoch:2d}/{cfg.epochs}  loss={entry['train_loss']:.4f}  "
              f"train_acc={entry['train_acc']:.4f}  test_clean={val_acc:.4f}  ({dt:.1f}s)",
              flush=True)
    return log
