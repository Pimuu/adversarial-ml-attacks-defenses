"""Figures for the paper (static PNGs, light background).

Color assignment follows a fixed categorical order so that each model keeps
the same color in every figure:
    standard -> blue, fgsm_at -> orange, pgd_at -> aqua.
Every multi-series figure has a legend (identity is never color alone), and
models are additionally distinguished by marker shape.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import FASHION_CLASSES, MODEL_NAMES  # noqa: E402

COLORS = {"standard": "#2a78d6", "fgsm_at": "#eb6834", "pgd_at": "#1baf7a"}
MARKERS = {"standard": "o", "fgsm_at": "s", "pgd_at": "D"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
WRONG = "#c62828"  # used only for text on misclassified predictions (+ an "x" mark)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_accuracy_vs_eps(rows, train_eps, path):
    """Two panels (FGSM | PGD): accuracy vs eps for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, attack in zip(axes, ("FGSM", "PGD")):
        for model in MODEL_NAMES:
            pts = sorted((r["epsilon"], r["accuracy"]) for r in rows
                         if r["model"] == model and r["attack"] == attack)
            if not pts:
                continue
            e, a = zip(*pts)
            ax.plot(e, np.array(a) * 100, color=COLORS[model], marker=MARKERS[model],
                    markersize=6, linewidth=2, label=MODEL_NAMES[model])
        ax.axvline(train_eps, color=INK_2, linestyle="--", linewidth=1)
        ax.text(train_eps, 101, f" training ε = {train_eps}", color=INK_2, fontsize=8,
                va="bottom", ha="left")
        ax.axhline(10, color=INK_2, linestyle=":", linewidth=1)
        ax.text(0.003, 11, "chance (10%)", color=INK_2, fontsize=8, ha="left")
        ax.set_title(f"{attack} attack (white-box, L∞)")
        ax.set_xlabel("ε (L∞ budget, pixel scale [0, 1])")
        ax.set_ylim(0, 105)
        _style_axes(ax)
    axes[0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Fashion-MNIST test accuracy under attack", fontweight="bold", y=1.03)
    _save(fig, path)


def plot_defense_comparison(rows, eps_show, path):
    """Grouped bars: before (standard) vs after (AT) defense, PGD-20, selected eps."""
    models = list(MODEL_NAMES)
    width = 0.26
    x = np.arange(len(eps_show))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, model in enumerate(models):
        vals = []
        for eps in eps_show:
            acc = [r["accuracy"] for r in rows if r["model"] == model
                   and r["attack"] == "PGD" and abs(r["epsilon"] - eps) < 1e-9]
            vals.append(acc[0] * 100 if acc else np.nan)
        # 2px-ish white edge separates adjacent bars.
        bars = ax.bar(x + (i - 1) * width, vals, width, color=COLORS[model],
                      edgecolor="white", linewidth=1.5, label=MODEL_NAMES[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, [("clean" if e == 0 else f"ε = {e}") for e in eps_show])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Before vs. after defense: accuracy under PGD-20")
    ax.legend(loc="upper right", ncol=1)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def plot_pgd_convergence(rows, path):
    """Robust accuracy vs PGD iterations at the check eps (log x axis)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model in MODEL_NAMES:
        pts = sorted((r["steps"], r["accuracy"]) for r in rows
                     if r["model"] == model and r["restarts"] == 1)
        if not pts:
            continue
        s, a = zip(*pts)
        ax.plot(s, np.array(a) * 100, color=COLORS[model], marker=MARKERS[model],
                markersize=6, linewidth=2, label=MODEL_NAMES[model])
        rr = [r for r in rows if r["model"] == model and r["restarts"] > 1]
        if rr:
            r = rr[0]
            ax.scatter([r["steps"]], [r["accuracy"] * 100], marker="*", s=140,
                       color=COLORS[model], edgecolor="white", linewidth=1, zorder=5)
    ax.scatter([], [], marker="*", s=140, color=INK_2,
               label="PGD with random restarts (★)")
    ax.set_xscale("log")
    ticks = sorted({r["steps"] for r in rows})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    eps = rows[0]["eps"]
    ax.set_title(f"Attack-strength check: PGD steps at ε = {eps}")
    ax.set_xlabel("PGD iterations (log scale)")
    ax.set_ylabel("Robust accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="center right")
    _style_axes(ax)
    _save(fig, path)


def plot_training_curves(log_rows, path):
    """Two panels: training loss and clean test accuracy per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for model in MODEL_NAMES:
        pts = [r for r in log_rows if r["method"] == model]
        if not pts:
            continue
        ep = [r["epoch"] for r in pts]
        kw = dict(color=COLORS[model], marker=MARKERS[model], markersize=5, linewidth=2,
                  label=MODEL_NAMES[model])
        axes[0].plot(ep, [r["train_loss"] for r in pts], **kw)
        axes[1].plot(ep, [r["test_clean_acc"] * 100 for r in pts], **kw)
    axes[0].set_title("Training loss (on the batches trained on)")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_title("Clean test accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        _style_axes(ax)
    axes[1].legend(loc="lower right")
    _save(fig, path)


def plot_adversarial_examples(x, x_fgsm, x_pgd, preds, y, eps, model_label, path,
                              amp=5.0):
    """Grid: rows = original / FGSM / PGD / amplified PGD perturbation.

    ``preds`` = dict with keys "clean", "FGSM", "PGD" holding predicted labels.
    Wrong predictions are written in red with an "✗" so the cue is not color-only.
    """
    n = x.shape[0]
    rows = [("Original", x, preds["clean"]),
            (f"FGSM ε={eps}", x_fgsm, preds["FGSM"]),
            (f"PGD-20 ε={eps}", x_pgd, preds["PGD"])]
    fig, axes = plt.subplots(4, n, figsize=(1.45 * n, 7.0))
    for j in range(n):
        for i, (name, imgs, pr) in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(imgs[j, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ok = int(pr[j]) == int(y[j])
            label = FASHION_CLASSES[int(pr[j])]
            ax.set_title(("" if ok else "✗ ") + label, fontsize=7.5,
                         color=INK if ok else WRONG, fontweight="normal")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=8.5)
        # Perturbation (PGD - original), amplified and centred on mid-gray.
        ax = axes[3, j]
        delta = (x_pgd[j, 0] - x[j, 0]).numpy()
        ax.imshow(0.5 + amp * delta / 2, cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"true: {FASHION_CLASSES[int(y[j])]}", fontsize=7.5, color=INK_2)
        if j == 0:
            ax.set_ylabel(f"PGD δ (×{amp:g})", fontsize=8.5)
    fig.suptitle(f"Adversarial examples against the {model_label} model "
                 f"(titles = predicted label; ✗ = wrong)", fontweight="bold", y=0.995,
                 fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_transfer(rows, path):
    """Black-box transfer: examples crafted on the standard model, evaluated on all."""
    eps_list = sorted({r["epsilon"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    width = 0.26
    x = np.arange(len(eps_list))
    for i, model in enumerate(MODEL_NAMES):
        vals = [next(r["accuracy"] for r in rows if r["target"] == model
                     and r["epsilon"] == e) * 100 for e in eps_list]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=COLORS[model],
                      edgecolor="white", linewidth=1.5, label=MODEL_NAMES[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, [f"ε = {e}" for e in eps_list])
    ax.set_ylabel("Accuracy on transferred examples (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Transfer attack: PGD-20 examples crafted on the standard model")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    _save(fig, path)
