"""Unit tests for the attack implementations (fast, no dataset download).

Run with:  python -m pytest -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from advlab.attacks import fgsm, pgd  # noqa: E402
from advlab.model import SmallCNN  # noqa: E402
from advlab.training import fgsm_rs  # noqa: E402


def _setup(n=16, seed=0):
    torch.manual_seed(seed)
    model = SmallCNN().eval()
    x = torch.rand(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return model, x, y


def test_fgsm_respects_budget_and_range():
    model, x, y = _setup()
    for eps in (0.01, 0.1, 0.3):
        x_adv = fgsm(model, x, y, eps)
        assert (x_adv - x).abs().max() <= eps + 1e-6
        assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_pgd_respects_budget_and_range():
    model, x, y = _setup()
    for eps in (0.01, 0.1, 0.3):
        x_adv = pgd(model, x, y, eps, steps=5, restarts=2)
        assert (x_adv - x).abs().max() <= eps + 1e-6
        assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_fgsm_rs_respects_budget():
    model, x, y = _setup()
    x_adv = fgsm_rs(model, x, y, 0.1, 0.125)
    assert (x_adv - x).abs().max() <= 0.1 + 1e-6


def test_eps_zero_is_identity():
    model, x, y = _setup()
    assert torch.equal(fgsm(model, x, y, 0.0), x)
    assert torch.equal(pgd(model, x, y, 0.0), x)


def test_attacks_increase_loss():
    model, x, y = _setup(n=64)
    loss = torch.nn.functional.cross_entropy
    with torch.no_grad():
        base = loss(model(x), y)
        l_fgsm = loss(model(fgsm(model, x, y, 0.1)), y)
        l_pgd = loss(model(pgd(model, x, y, 0.1, steps=10)), y)
    assert l_fgsm > base
    assert l_pgd > base


def test_attacks_do_not_touch_parameter_grads_or_mode():
    model, x, y = _setup()
    model.train()
    pgd(model, x, y, 0.1, steps=3)
    assert model.training  # mode restored
    assert all(p.grad is None for p in model.parameters())


def test_pgd_seeded_is_deterministic():
    model, x, y = _setup()
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    a = pgd(model, x, y, 0.1, steps=3, generator=g1)
    b = pgd(model, x, y, 0.1, steps=3, generator=g2)
    assert torch.equal(a, b)
