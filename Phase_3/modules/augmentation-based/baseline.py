"""No-defence baseline: the Phase 1 recipe, trained with the same loop as the defences.

Used as the reference curve in the loss report, and as the frozen model that
median smoothing is calibrated on when the Phase 1 checkpoint is not available.

  python modules/augmentation-based/baseline.py
"""
import argparse

import _trainer as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="override Phase 1 epochs (quick test)")
    args = parser.parse_args()

    config = T.load_config()
    if args.epochs:
        config["epochs"] = args.epochs
    T.setup_gpu()
    tuning_csv = T.TUNING / "baseline_none" / "history.csv"
    T.finished_run("none", config, tuning_csv) or T.train_one("none", T.plain_step, config, tuning_csv)


if __name__ == "__main__":
    main()
