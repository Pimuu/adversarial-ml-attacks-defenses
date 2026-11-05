#!/usr/bin/env python3
"""Reproduce every result in results/ with one command.

    python run_pipeline.py                 # full experiment (~30-35 min on 4 CPU cores)
    python run_pipeline.py --reuse-models  # skip training, re-evaluate saved checkpoints
    python run_pipeline.py --quick --out results_quick   # ~2-minute smoke test

Outputs are deterministic for a given --seed and --threads on the same
software stack (see README, "Reproducibility").
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from advrobust.pipeline import Config, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    parser.add_argument("--reuse-models", action="store_true",
                        help="load checkpoints from <out>/models/ instead of training")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    cfg = Config.quick() if args.quick else Config()
    cfg.seed, cfg.threads = args.seed, args.threads
    run(cfg, out_dir=args.out, data_dir=ROOT / "data", reuse_models=args.reuse_models)


if __name__ == "__main__":
    main()
