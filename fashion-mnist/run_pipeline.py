#!/usr/bin/env python3
"""End-to-end reproducible pipeline.

    python run_pipeline.py            # full run (≈ 1.5–2 h on a 4-core CPU)
    python run_pipeline.py --quick    # smoke test (≈ 3 min): 1 epoch, 500 test images

Stages
------
1. Load Fashion-MNIST (downloaded to ./data on first run).
2. Train three models with the same architecture and seed:
     standard  (no defense), fgsm_at (fast adversarial training),
     pgd_at    (PGD adversarial training, the main defense).
   Checkpoints go to ./checkpoints and are reused on re-runs unless --retrain.
3. Evaluate clean accuracy and robust accuracy under FGSM and PGD-20 at every
   epsilon in the config, on the test set.
4. Sanity checks against gradient masking: PGD steps sweep + random restarts,
   and black-box transfer from the standard model.
5. Write CSV/JSON results and figures to ./results.

Use --plots-only to regenerate figures from existing CSVs.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch  # noqa: E402

from advlab import plotting  # noqa: E402
from advlab.config import MODEL_NAMES, Config  # noqa: E402
from advlab.data import load_fashion_mnist  # noqa: E402
from advlab.evaluation import (eps_sweep, pgd_convergence, predict,  # noqa: E402
                               robust_accuracy, transfer_attack)
from advlab.model import SmallCNN, count_parameters  # noqa: E402
from advlab.training import train_model  # noqa: E402
from advlab.utils import get_device, set_seed, timer, write_csv, write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-eval", type=int, default=None, help="number of test images")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--retrain", action="store_true", help="ignore existing checkpoints")
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config(seed=args.seed)
    if args.quick:
        cfg.epochs = 1
        cfg.n_eval = 500
        cfg.eval_batch_size = 500
        cfg.eval_eps = [0.0, 0.05, 0.1, 0.2]
        cfg.check_n = 200
        cfg.check_steps = [1, 10, 20]
        cfg.check_restarts = 2
        cfg.check_restart_steps = 10
        cfg.results_dir = "results_quick"
        cfg.checkpoint_dir = "checkpoints_quick"
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.n_eval is not None:
        cfg.n_eval = args.n_eval
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    return cfg


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if any(c in v for c in ".e") else int(v)
            except ValueError:
                pass
    return rows


def make_plots(cfg: Config):
    rd, fd = cfg.results_dir, os.path.join(cfg.results_dir, "figures")
    sweep = read_csv(os.path.join(rd, "metrics_long.csv"))
    plotting.plot_accuracy_vs_eps(sweep, cfg.train_eps,
                                  os.path.join(fd, "accuracy_vs_epsilon.png"))
    eps_show = [e for e in (0.0, 0.05, 0.1, 0.15, 0.2) if e in cfg.eval_eps]
    plotting.plot_defense_comparison(sweep, eps_show,
                                     os.path.join(fd, "defense_comparison.png"))
    plotting.plot_pgd_convergence(read_csv(os.path.join(rd, "pgd_convergence.csv")),
                                  os.path.join(fd, "pgd_convergence.png"))
    plotting.plot_training_curves(read_csv(os.path.join(rd, "training_log.csv")),
                                  os.path.join(fd, "training_curves.png"))
    plotting.plot_transfer(read_csv(os.path.join(rd, "transfer_attack.csv")),
                           os.path.join(fd, "transfer_attack.png"))


def wide_table(sweep, models):
    """One row per model: clean acc, then acc under each attack at each eps."""
    out = []
    for m in models:
        row = {"model": m, "description": MODEL_NAMES[m]}
        clean = [r for r in sweep if r["model"] == m and r["epsilon"] == 0]
        row["clean_acc"] = clean[0]["accuracy"]
        for attack in ("FGSM", "PGD"):
            for r in sweep:
                if r["model"] == m and r["attack"] == attack and r["epsilon"] > 0:
                    row[f"{attack}_eps{r['epsilon']}"] = r["accuracy"]
        out.append(row)
    return out


def main():
    args = parse_args()
    cfg = build_config(args)
    if args.plots_only:
        make_plots(cfg)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    device = get_device(args.device)
    print(f"device={device} torch={torch.__version__} threads={torch.get_num_threads()}")
    timings = {}
    t_start = time.perf_counter()

    # ------------------------------------------------------------ 1. data
    x_tr, y_tr, x_te, y_te = load_fashion_mnist(cfg.data_dir)
    x_ev, y_ev = x_te[:cfg.n_eval], y_te[:cfg.n_eval]
    print(f"train={tuple(x_tr.shape)} test={tuple(x_te.shape)} eval_subset={cfg.n_eval}")

    # -------------------------------------------------------- 2. training
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    models, train_log = {}, []
    log_path = os.path.join(cfg.results_dir, "training_log.csv")
    old_log = read_csv(log_path) if os.path.exists(log_path) else []
    for method in MODEL_NAMES:
        # Same seed for every method -> identical initialisation and data order.
        gen = set_seed(cfg.seed)
        model = SmallCNN().to(device)
        ckpt = os.path.join(cfg.checkpoint_dir, f"{method}_seed{cfg.seed}.pt")
        if os.path.exists(ckpt) and not args.retrain:
            model.load_state_dict(torch.load(ckpt, map_location=device))
            train_log += [r for r in old_log if r["method"] == method]
            print(f"loaded {ckpt}")
        else:
            print(f"training {method} ({count_parameters(model):,} params)")
            with timer(f"train_{method}", timings):
                train_log += train_model(model, method, x_tr, y_tr, x_te, y_te, cfg,
                                         device, gen)
            torch.save(model.state_dict(), ckpt)
        model.eval()
        models[method] = model
    write_csv(log_path, train_log)

    # ------------------------------------------------------ 3. eps sweep
    print("evaluating FGSM / PGD across epsilons")
    gen = set_seed(cfg.seed + 1)
    with timer("eval_sweep", timings):
        sweep = eps_sweep(models, x_ev, y_ev, cfg, device, gen)
    write_csv(os.path.join(cfg.results_dir, "metrics_long.csv"), sweep)
    write_csv(os.path.join(cfg.results_dir, "metrics_table.csv"),
              wide_table(sweep, list(models)))

    # ----------------------------------------- 4. gradient-masking checks
    print("PGD convergence / restarts check")
    gen = set_seed(cfg.seed + 2)
    with timer("eval_convergence", timings):
        conv = pgd_convergence(models, x_te[:cfg.check_n], y_te[:cfg.check_n], cfg,
                               device, gen)
    write_csv(os.path.join(cfg.results_dir, "pgd_convergence.csv"), conv)

    print("transfer attack from the standard model")
    gen = set_seed(cfg.seed + 3)
    t_eps = [e for e in (0.05, 0.1, 0.2) if e in cfg.eval_eps]
    with timer("eval_transfer", timings):
        transfer = transfer_attack("standard", models, x_ev, y_ev, t_eps, cfg, device, gen)
    write_csv(os.path.join(cfg.results_dir, "transfer_attack.csv"), transfer)

    # ------------------------------------------------ 5. example images
    gen = set_seed(cfg.seed + 4)
    n_img = cfg.n_example_images
    # Pick the first image of each of the first n_img classes for variety.
    idx = torch.stack([(y_te == c).nonzero()[0, 0] for c in range(n_img)])
    xs, ys = x_te[idx], y_te[idx]
    eps_img = cfg.train_eps
    for name in ("standard", "pgd_at"):
        m = models[name]
        _, xf = robust_accuracy(m, xs, ys, "FGSM", eps_img, cfg, device, return_adv=True)
        _, xp = robust_accuracy(m, xs, ys, "PGD", eps_img, cfg, device, gen,
                                return_adv=True)
        preds = {"clean": predict(m, xs, device), "FGSM": predict(m, xf, device),
                 "PGD": predict(m, xp, device)}
        label = "undefended" if name == "standard" else "PGD-adversarially-trained"
        plotting.plot_adversarial_examples(
            xs, xf, xp, preds, ys, eps_img, label,
            os.path.join(cfg.results_dir, "figures", f"adversarial_examples_{name}.png"))

    # ------------------------------------------------------- 6. summary
    timings["total"] = time.perf_counter() - t_start
    write_json(os.path.join(cfg.results_dir, "run_info.json"), {
        "config": cfg.to_dict(),
        "n_parameters": count_parameters(models["standard"]),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
    })
    make_plots(cfg)
    print(f"done in {timings['total'] / 60:.1f} min -> {cfg.results_dir}/")


if __name__ == "__main__":
    main()
