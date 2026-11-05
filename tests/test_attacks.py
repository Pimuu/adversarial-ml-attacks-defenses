"""Unit tests for attacks, evaluation helpers and the training loop.

They use a tiny model on synthetic data, so they run in seconds and need no
dataset download.  Run with:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advrobust.attacks import fgsm, pgd, run_attack  # noqa: E402
from advrobust.evaluate import correct_under_attack, wilson_interval  # noqa: E402
from advrobust.model import SmallCNN, count_parameters  # noqa: E402
from advrobust.train import train_model  # noqa: E402

torch.set_num_threads(1)


def _toy_problem(n=256, seed=0):
    """Easily separable 'images': class 1 has a bright left half, class 0 a bright right half."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.rand(n, 1, 28, 28, generator=g) * 0.5
    x[y == 1, :, :, :14] += 0.5
    x[y == 0, :, :, 14:] += 0.5
    return x, y


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        return self.fc(x)


class TestAttacks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TinyNet().eval()
        self.x, self.y = _toy_problem(64)

    def _check_constraints(self, x_adv, eps):
        self.assertEqual(x_adv.shape, self.x.shape)
        self.assertLessEqual((x_adv - self.x).abs().max().item(), eps + 1e-6)
        self.assertGreaterEqual(x_adv.min().item(), 0.0)
        self.assertLessEqual(x_adv.max().item(), 1.0)

    def test_fgsm_respects_ball_and_box(self):
        for eps in (0.05, 0.3):
            self._check_constraints(fgsm(self.model, self.x, self.y, eps), eps)

    def test_pgd_respects_ball_and_box(self):
        for eps in (0.05, 0.3):
            x_adv = pgd(self.model, self.x, self.y, eps, steps=10, generator=torch.Generator().manual_seed(1))
            self._check_constraints(x_adv, eps)

    def test_zero_eps_is_identity(self):
        self.assertTrue(torch.equal(fgsm(self.model, self.x, self.y, 0.0), self.x))
        self.assertTrue(torch.equal(pgd(self.model, self.x, self.y, 0.0), self.x))
        self.assertTrue(torch.equal(run_attack("none", self.model, self.x, self.y, 0.3), self.x))

    def test_attacks_increase_loss(self):
        clean = F.cross_entropy(self.model(self.x), self.y).item()
        for x_adv in (fgsm(self.model, self.x, self.y, 0.1),
                      pgd(self.model, self.x, self.y, 0.1, steps=10, generator=torch.Generator().manual_seed(0))):
            self.assertGreater(F.cross_entropy(self.model(x_adv), self.y).item(), clean)

    def test_attacks_do_not_touch_inputs_or_weights(self):
        x_before = self.x.clone()
        w_before = [p.detach().clone() for p in self.model.parameters()]
        pgd(self.model, self.x, self.y, 0.2, steps=5)
        fgsm(self.model, self.x, self.y, 0.2)
        self.assertTrue(torch.equal(self.x, x_before))
        for p, w in zip(self.model.parameters(), w_before):
            self.assertTrue(torch.equal(p, w))
            self.assertIsNone(p.grad)

    def test_restarts_never_weaker(self):
        # Same seed => the first restart is identical, so extra restarts can only add successes.
        def fooled(restarts):
            x_adv = pgd(self.model, self.x, self.y, 0.05, steps=5, restarts=restarts,
                        generator=torch.Generator().manual_seed(3))
            with torch.no_grad():
                return self.model(x_adv).argmax(1) != self.y
        one, five = fooled(1), fooled(5)
        self.assertTrue(torch.all(five | ~one))  # one => five
        self.assertGreaterEqual(int(five.sum()), int(one.sum()))

    def test_transfer_mode_uses_source_model(self):
        other = TinyNet().eval()
        ok = correct_under_attack(self.model, self.x, self.y, "pgd", 0.1, source_model=other, steps=3)
        self.assertEqual(ok.shape, (len(self.x),))
        self.assertEqual(ok.dtype, torch.bool)


class TestHelpers(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        lo, hi = wilson_interval(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_small_cnn_shapes(self):
        model = SmallCNN()
        self.assertEqual(model(torch.zeros(3, 1, 28, 28)).shape, (3, 10))
        self.assertEqual(count_parameters(model), 225_034)


class TestTraining(unittest.TestCase):
    def test_each_mode_runs_and_learns(self):
        x, y = _toy_problem(512, seed=1)
        xt, yt = _toy_problem(256, seed=2)
        for mode in ("standard", "fgsm", "pgd"):
            torch.manual_seed(0)
            model = TinyNet()
            hist = train_model(model, x, y, xt, yt, mode=mode, epochs=3, eps=0.02, lr=3e-3,
                               batch_size=64, pgd_steps=2, seed=0, log=lambda *_: None)
            self.assertEqual(len(hist), 3)
            self.assertGreater(hist[-1]["test_clean_acc"], 0.75, mode)


if __name__ == "__main__":
    unittest.main()
