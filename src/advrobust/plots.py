"""Figures for results/figures/ (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_LABELS = {
    "baseline": "Baseline (standard training)",
    "fgsm_at": "FGSM adversarial training",
    "pgd_at": "PGD adversarial training",
    "surrogate": "Surrogate (standard, other seed)",
}
COLORS = {"baseline": "#C44E52", "fgsm_at": "#DD8452", "pgd_at": "#4C72B0", "surrogate": "#8C8C8C"}
OK_COLOR, BAD_COLOR = "#2E7D32", "#C62828"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def plot_accuracy_vs_eps(sweep: pd.DataFrame, train_eps: float, path: Path) -> None:
    """Two panels (FGSM, PGD): accuracy vs epsilon for each model, with 95% CIs."""
    attacks = list(dict.fromkeys(sweep.attack))
    fig, axes = plt.subplots(1, len(attacks), figsize=(6 * len(attacks), 4.6), sharey=True)
    for ax, attack in zip(np.atleast_1d(axes), attacks):
        for model in dict.fromkeys(sweep.model):
            d = sweep[(sweep.model == model) & (sweep.attack == attack)].sort_values("epsilon")
            ax.plot(d.epsilon, d.accuracy * 100, marker="o", ms=4, lw=2, color=COLORS[model],
                    label=MODEL_LABELS[model])
            ax.fill_between(d.epsilon, d.ci95_low * 100, d.ci95_high * 100, color=COLORS[model], alpha=0.15, lw=0)
        ax.axvline(train_eps, color="black", ls=":", lw=1)
        ax.text(train_eps, 102, f" training ε = {train_eps}", fontsize=8, va="bottom")
        ax.axhline(10, color="grey", ls="--", lw=0.8)
        ax.text(max(sweep.epsilon), 11, "chance (10%)", fontsize=8, color="grey", ha="right")
        ax.set(title=f"White-box {attack} attack", xlabel="Perturbation budget ε (L∞, pixel scale 0–1)",
               ylim=(0, 108))
    np.atleast_1d(axes)[0].set_ylabel("Test accuracy (%)")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def _image_row(axes, images, preds, labels, row_title):
    for j, ax in enumerate(axes):
        ax.imshow(images[j], cmap="gray", vmin=0, vmax=1)
        ok = preds[j] == labels[j]
        ax.set_title(f"pred {preds[j]}", fontsize=9, color=OK_COLOR if ok else BAD_COLOR, fontweight="bold")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel(row_title, fontsize=9, rotation=0, ha="right", va="center", labelpad=10)


def plot_adversarial_examples(clean, fgsm_x, pgd_x, labels, preds: dict, eps: float, path: Path) -> None:
    """Original vs FGSM vs PGD images (against the baseline) with predicted labels."""
    n = len(clean)
    fig, axes = plt.subplots(4, n, figsize=(1.35 * n + 1.6, 6.4))
    _image_row(axes[0], clean, preds["clean"], labels, f"Original\n(true label\nabove grid)")
    _image_row(axes[1], fgsm_x, preds["fgsm"], labels, f"FGSM\nε = {eps}")
    _image_row(axes[2], pgd_x, preds["pgd"], labels, f"PGD\nε = {eps}")
    diff = pgd_x - clean
    for j, ax in enumerate(axes[3]):
        ax.imshow(diff[j], cmap="RdBu_r", vmin=-eps, vmax=eps)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[3][0].set_ylabel("PGD\nperturbation\n(red +, blue −)", fontsize=9, rotation=0, ha="right", va="center", labelpad=10)
    fig.suptitle("Adversarial examples against the undefended baseline "
                 f"(true labels: {' '.join(str(l) for l in labels)}; green = correct, red = fooled)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_defense_examples(clean, adv_base, adv_robust, labels, preds: dict, eps: float, path: Path) -> None:
    """Same images attacked with PGD at the training eps: baseline vs PGD-trained model."""
    n = len(clean)
    fig, axes = plt.subplots(3, n, figsize=(1.35 * n + 1.6, 4.9))
    _image_row(axes[0], clean, preds["clean_robust"], labels, "Original\n(PGD-AT pred)")
    _image_row(axes[1], adv_base, preds["base"], labels, f"PGD ε={eps}\nvs baseline")
    _image_row(axes[2], adv_robust, preds["robust"], labels, f"PGD ε={eps}\nvs PGD-AT")
    fig.suptitle("Before vs after the defense: PGD-100 at the full training budget "
                 "(each row attacks the model named on the left)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_before_after(summary: pd.DataFrame, path: Path, eps: float) -> None:
    """Grouped bars: each model's accuracy under increasingly strong evaluations."""
    cols = [("clean_acc", "Clean"), (f"fgsm_acc_eps{eps}", f"FGSM ε={eps}"),
            (f"pgd_acc_eps{eps}", f"PGD-20 ε={eps}"), (f"strong_pgd_acc_eps{eps}", f"PGD-100×5 ε={eps}"),
            (f"transfer_acc_eps{eps}", f"Transfer ε={eps}")]
    cols = [(c, lbl) for c, lbl in cols if c in summary.columns]
    models = summary.model.tolist()
    x = np.arange(len(cols))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, m in enumerate(models):
        vals = [summary.set_index("model").loc[m, c] * 100 for c, _ in cols]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, width=w, color=COLORS[m], label=MODEL_LABELS[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [lbl for _, lbl in cols])
    ax.set(ylabel="Accuracy (%)", ylim=(0, 112),
           title="Before vs after defense: clean, white-box, strong white-box and black-box (transfer) accuracy")
    ax.legend(loc="upper right", fontsize=8.5, ncol=3, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save(fig, path)


def plot_training_curves(hist: pd.DataFrame, train_eps: float, path: Path) -> None:
    models = [m for m in dict.fromkeys(hist.model) if m != "surrogate"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for m in models:
        d = hist[hist.model == m]
        axes[0].plot(d.epoch, d.test_clean_acc * 100, marker="o", color=COLORS[m], label=MODEL_LABELS[m])
        axes[1].plot(d.epoch, d.monitor_fgsm_acc * 100, marker="o", color=COLORS[m])
        axes[2].plot(d.epoch, d.monitor_pgd10_acc * 100, marker="o", color=COLORS[m])
    titles = ["Clean test accuracy (10k)", f"FGSM accuracy, ε = {train_eps} (500 imgs)",
              f"PGD-10 accuracy, ε = {train_eps} (500 imgs)"]
    for ax, t in zip(axes, titles):
        ax.set(title=t, xlabel="Epoch", ylabel="Accuracy (%)")
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
