"""End-to-end experiment: train -> attack -> defend -> evaluate -> results/.

Called by the top-level script:  python run_pipeline.py [--quick] [--reuse-models]
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch

from . import plots
from .attacks import fgsm, pgd
from .data import load_mnist, set_seed
from .evaluate import correct_under_attack, evaluate_sweep, wilson_interval
from .model import SmallCNN, count_parameters
from .train import accuracy, train_model


@dataclass
class Config:
    seed: int = 42
    threads: int = 4
    train_eps: float = 0.3          # L-inf budget used for adversarial training and headline numbers
    eps_grid: list = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    epochs_standard: int = 5
    epochs_fgsm_at: int = 5
    epochs_pgd_at: int = 6
    pgd_train_steps: int = 7
    ramp_epochs: float = 1.0
    pgd_eval_steps: int = 20        # PGD used for the epsilon sweep
    strong_steps: int = 100         # strong-attack check
    strong_restarts: int = 5
    n_eval: int = 2000              # random test subset for attack sweeps
    n_strong: int = 1000            # subset (of the above) for the strong check
    n_monitor: int = 500            # subset tracked during training
    transfer_eps: list = field(default_factory=lambda: [0.1, 0.2, 0.3])
    example_eps: float = 0.2
    n_examples: int = 8

    @classmethod
    def quick(cls) -> "Config":
        """Tiny configuration for smoke tests (~2 minutes); results are not meaningful."""
        return cls(epochs_standard=1, epochs_fgsm_at=1, epochs_pgd_at=1, pgd_train_steps=3,
                   eps_grid=[0.0, 0.1, 0.3], pgd_eval_steps=5, strong_steps=10, strong_restarts=2,
                   n_eval=300, n_strong=200, n_monitor=100, transfer_eps=[0.3])


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rounded(obj, digits: int = 6):
    if isinstance(obj, dict):
        return {k: _rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v, digits) for v in obj]
    if isinstance(obj, float):
        return round(obj, digits)
    return obj


def run(cfg: Config, out_dir: Path, data_dir: Path, reuse_models: bool = False) -> dict:
    torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    fig_dir, model_dir = out_dir / "figures", out_dir / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # ------------------------------------------------------------------ data
    _log("Loading MNIST")
    x_train, y_train, x_test, y_test = load_mnist(data_dir)
    perm = torch.randperm(len(x_test), generator=torch.Generator().manual_seed(cfg.seed))
    eval_idx = perm[: cfg.n_eval]
    x_eval, y_eval = x_test[eval_idx], y_test[eval_idx]
    x_mon, y_mon = x_eval[: cfg.n_monitor], y_eval[: cfg.n_monitor]

    # ---------------------------------------------------------------- models
    specs = {
        # name: (training mode, epochs, seed offset)
        "baseline": ("standard", cfg.epochs_standard, 0),
        "surrogate": ("standard", cfg.epochs_standard, 100),  # independent copy for transfer attacks
        "fgsm_at": ("fgsm", cfg.epochs_fgsm_at, 200),
        "pgd_at": ("pgd", cfg.epochs_pgd_at, 300),
    }
    models, histories = {}, []
    for name, (mode, epochs, offset) in specs.items():
        ckpt = model_dir / f"{name}.pt"
        torch.manual_seed(cfg.seed + offset)  # weight initialisation
        model = SmallCNN()
        if reuse_models and ckpt.exists():
            _log(f"Loading {name} from {ckpt}")
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            model.eval()
        else:
            _log(f"Training {name} (mode={mode}, epochs={epochs})")
            t0 = time.time()
            hist = train_model(model, x_train, y_train, x_test, y_test, mode=mode, epochs=epochs,
                               eps=cfg.train_eps, pgd_steps=cfg.pgd_train_steps, ramp_epochs=cfg.ramp_epochs,
                               seed=cfg.seed + offset, monitor_x=x_mon, monitor_y=y_mon, log=print)
            timings[f"train_{name}_s"] = round(time.time() - t0, 1)
            histories += [{"model": name, "mode": mode, **h} for h in hist]
            torch.save(model.state_dict(), ckpt)
        models[name] = model
    if histories:
        pd.DataFrame(histories).to_csv(out_dir / "training_log.csv", index=False, float_format="%.4f")
    main_models = {k: models[k] for k in ("baseline", "fgsm_at", "pgd_at")}

    # ----------------------------------------------------- clean accuracy
    clean = {name: accuracy(m, x_test, y_test) for name, m in models.items()}
    _log("Clean test accuracy (10k): " + ", ".join(f"{k}={v:.4f}" for k, v in clean.items()))

    # --------------------------------------------------------- eps sweep
    _log(f"White-box sweep: FGSM and PGD-{cfg.pgd_eval_steps} on {cfg.n_eval} test images")
    t0 = time.time()
    sweep = pd.DataFrame(evaluate_sweep(main_models, x_eval, y_eval, cfg.eps_grid, cfg.pgd_eval_steps,
                                        seed=cfg.seed, log=print))
    timings["sweep_s"] = round(time.time() - t0, 1)
    sweep.to_csv(out_dir / "metrics.csv", index=False, float_format="%.4f")
    wide = sweep.pivot_table(index=["model", "attack"], columns="epsilon", values="accuracy", sort=False)
    wide.columns = [f"eps_{c:g}" for c in wide.columns]
    wide.reset_index().to_csv(out_dir / "accuracy_table.csv", index=False, float_format="%.4f")

    # ------------------------------------------------------- strong check
    eps = cfg.train_eps
    _log(f"Strong-attack check at eps={eps}: PGD-{cfg.strong_steps} x {cfg.strong_restarts} restarts "
         f"on {cfg.n_strong} images")
    x_s, y_s = x_eval[: cfg.n_strong], y_eval[: cfg.n_strong]
    strong_rows = []
    t0 = time.time()
    for name, m in main_models.items():
        for label, kwargs in [(f"PGD-{cfg.pgd_eval_steps}", {"steps": cfg.pgd_eval_steps}),
                              (f"PGD-{cfg.strong_steps} x{cfg.strong_restarts}",
                               {"steps": cfg.strong_steps, "restarts": cfg.strong_restarts})]:
            ok = correct_under_attack(m, x_s, y_s, "pgd", eps, seed=cfg.seed + 7, **kwargs)
            k, n = int(ok.sum()), len(ok)
            lo, hi = wilson_interval(k, n)
            strong_rows.append({"model": name, "attack": label, "epsilon": eps, "accuracy": k / n,
                                "ci95_low": lo, "ci95_high": hi, "n": n})
            print(f"    {name:<10} {label:<14} acc={k / n:.4f}")
    timings["strong_check_s"] = round(time.time() - t0, 1)
    strong = pd.DataFrame(strong_rows)
    strong.to_csv(out_dir / "strong_attack_check.csv", index=False, float_format="%.4f")

    # ---------------------------------------------------- transfer attack
    _log(f"Black-box transfer: PGD-{cfg.pgd_eval_steps} crafted on the surrogate")
    tr_rows = []
    for t_eps in cfg.transfer_eps:
        for name, m in {**main_models, "surrogate": models["surrogate"]}.items():
            ok = correct_under_attack(m, x_eval, y_eval, "pgd", t_eps, seed=cfg.seed + 11,
                                      source_model=models["surrogate"], steps=cfg.pgd_eval_steps)
            k, n = int(ok.sum()), len(ok)
            lo, hi = wilson_interval(k, n)
            tr_rows.append({"source": "surrogate", "target": name, "epsilon": t_eps, "accuracy": k / n,
                            "ci95_low": lo, "ci95_high": hi, "n": n,
                            "note": "white-box (source = target)" if name == "surrogate" else "black-box transfer"})
    transfer = pd.DataFrame(tr_rows)
    transfer.to_csv(out_dir / "transfer_attack.csv", index=False, float_format="%.4f")
    print(transfer.pivot(index="target", columns="epsilon", values="accuracy").round(4).to_string())

    # ----------------------------------------------------- summary table
    def sweep_acc(model, attack_prefix, e):
        d = sweep[(sweep.model == model) & sweep.attack.str.startswith(attack_prefix) & (sweep.epsilon == e)]
        return float(d.accuracy.iloc[0]) if len(d) else float("nan")

    summary_rows = []
    for name in main_models:
        row = {"model": name, "clean_acc": clean[name]}
        for e in sorted({0.1, 0.2, eps} & set(cfg.eps_grid)):
            row[f"fgsm_acc_eps{e}"] = sweep_acc(name, "FGSM", e)
            row[f"pgd_acc_eps{e}"] = sweep_acc(name, "PGD", e)
        row[f"strong_pgd_acc_eps{eps}"] = float(strong[(strong.model == name) &
                                                       strong.attack.str.contains("x")].accuracy.iloc[0])
        t = transfer[(transfer.target == name) & (transfer.epsilon == eps)]
        if len(t):
            row[f"transfer_acc_eps{eps}"] = float(t.accuracy.iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_table.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------- example images
    _log("Rendering figures")
    base, robust = models["baseline"], models["pgd_at"]
    with torch.no_grad():
        ok_both = (base(x_eval).argmax(1) == y_eval) & (robust(x_eval).argmax(1) == y_eval)
    # First correctly-classified example of each digit 0..7 (deterministic).
    pick = []
    for digit in range(cfg.n_examples):
        idx = torch.nonzero((y_eval == digit) & ok_both).flatten()
        if len(idx):
            pick.append(int(idx[0]))
    xs, ys = x_eval[pick], y_eval[pick]
    g = torch.Generator().manual_seed(cfg.seed + 99)
    ex_eps = cfg.example_eps
    x_f = fgsm(base, xs, ys, ex_eps)
    x_p = pgd(base, xs, ys, ex_eps, steps=cfg.strong_steps, generator=g)
    x_pb = pgd(base, xs, ys, eps, steps=cfg.strong_steps, generator=g)
    x_pr = pgd(robust, xs, ys, eps, steps=cfg.strong_steps, generator=g)
    with torch.no_grad():
        preds = {k: v(x).argmax(1).tolist() for k, (v, x) in {
            "clean": (base, xs), "fgsm": (base, x_f), "pgd": (base, x_p), "clean_robust": (robust, xs),
            "base": (base, x_pb), "robust": (robust, x_pr)}.items()}
    sq = lambda t: t.squeeze(1).numpy()  # noqa: E731
    labels = ys.tolist()
    plots.plot_adversarial_examples(sq(xs), sq(x_f), sq(x_p), labels, preds, ex_eps,
                                    fig_dir / "adversarial_examples.png")
    plots.plot_defense_examples(sq(xs), sq(x_pb), sq(x_pr), labels, preds, eps, fig_dir / "defense_examples.png")
    plots.plot_accuracy_vs_eps(sweep, eps, fig_dir / "accuracy_vs_epsilon.png")
    plots.plot_before_after(summary, fig_dir / "before_after_defense.png", eps)
    if histories:
        plots.plot_training_curves(pd.DataFrame(histories), eps, fig_dir / "training_curves.png")
    # Mean L-inf / L2 size of the example perturbations, for the paper.
    pert = {"pgd_vs_baseline_eps": eps,
            "mean_l2_pgd_vs_baseline": float((x_pb - xs).flatten(1).norm(dim=1).mean()),
            "mean_l2_pgd_vs_pgd_at": float((x_pr - xs).flatten(1).norm(dim=1).mean()),
            "example_predictions": preds, "example_labels": labels}

    result = {
        "config": asdict(cfg),
        "n_parameters": count_parameters(base),
        "clean_test_accuracy": clean,
        "summary": {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in summary_rows},
        "examples": pert,
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(_rounded(result), fh, indent=2)
    # Wall-clock times differ between machines and runs; kept out of metrics.json
    # so that the deterministic outputs stay byte-identical. With --reuse-models
    # no training happens, so earlier training times are kept.
    timing_file = out_dir / "timings.json"
    if reuse_models and timing_file.exists():
        timings = {**json.loads(timing_file.read_text()), **timings}
    with open(timing_file, "w") as fh:
        json.dump(timings, fh, indent=2)
    _log(f"Done. Results written to {out_dir}/")
    return result
