"""Calibrate the preprocessing defences (D1-D3) on clean val, then freeze them.

No attack data or attack knowledge is used (PRD_defences.md):
- Only the clean val/ split is read. check_tuning_split() refuses anything
  else, in particular every *attacked* folder.
- The grids (modules/preprocessing/defences.py GRIDS) come from general
  image-processing ranges, not from any attack budget.
- Rule, fixed before running: for each defence, the strongest setting whose
  clean-val accuracy is at most TOLERANCE below no defence. If none qualifies,
  the mildest setting is used and flagged.

Outputs (both folders are created if missing, same layout as the augmentation
defences' checkpoints/augmentation/ + tuning_logs/augmentation/):
  checkpoints/preprocessing/<defence>_grid.csv   grid of each tuning run (+ logs/)
  tuning_logs/preprocessing/<defence>_grid.csv   copy kept with the committed tuning record
  tuning_logs/preprocessing/summary.csv          one row per defence: chosen setting
and the chosen settings in tuned_defences.yaml (jpeg_compression,
gaussian_smoothing, median_smoothing).

  python tune_preprocessing.py                          # Phase 1 model from config.yaml
  python tune_preprocessing.py --model path/to/best.keras
  python tune_preprocessing.py --defence median_smoothing
"""
import argparse
import csv
import sys
from pathlib import Path

PHASE3 = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE3 / "modules" / "augmentation-based"))

import _trainer as T  # noqa: E402
from modules.evaluation import load_model, predict_split  # noqa: E402
from modules.preprocessing import defences as D  # noqa: E402

TUNING_SPLIT = "val"
TOLERANCE = 0.01  # max clean-val accuracy drop (1 pt), fixed a priori
CHECKPOINTS = PHASE3 / "checkpoints" / "preprocessing"  # tuning runs (grids, logs), gitignored
TUNING = PHASE3 / "tuning_logs" / "preprocessing"       # tuning record (grids, summary), committed


def check_tuning_split(data_root, split):
    """Refuse to tune on anything but the clean val split (no leakage)."""
    path = Path(data_root) / split
    if split != TUNING_SPLIT or "attacked" in str(path).lower():
        raise ValueError(f"Preprocessing defences are tuned on clean '{TUNING_SPLIT}/' only, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Clean val split not found: {path}")
    return path


def relative(path):
    """Path relative to src/ when possible, so tuned_defences.yaml is machine independent."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PHASE3.parent).as_posix()
    except ValueError:
        return path.as_posix()


def choose(rows, base_accuracy):
    """Strongest grid setting within TOLERANCE of no defence (rows are mildest first)."""
    ok = [r for r in rows if base_accuracy - r["accuracy"] <= TOLERANCE]
    return (ok[-1], True) if ok else (rows[0], False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="frozen model (default: phase1_model in config.yaml)")
    parser.add_argument("--data", help="data root holding val/ (default: data in config.yaml)")
    parser.add_argument("--defence", nargs="+", choices=list(D.GRIDS), default=list(D.GRIDS))
    args = parser.parse_args()

    config = T.load_config()
    if args.data:
        config["data"] = str(Path(args.data).resolve())
    check_tuning_split(config["data"], TUNING_SPLIT)
    for folder in (CHECKPOINTS, TUNING):
        folder.mkdir(parents=True, exist_ok=True)
    T.setup_gpu()
    model_path = Path(args.model).resolve() if args.model else Path(config["phase1_model"])
    model = load_model(model_path)
    val_ds = T.load_split(TUNING_SPLIT, config, shuffle=False).batch(int(config["batch"]))

    base = T.metrics(*predict_split(D.make_predictor(D.NO_DEFENCE, model), val_ds))
    print(f"no defence: val_acc={base['accuracy']:.4f} val_macro_f1={base['macro_f1']:.4f}", flush=True)

    summary = []
    for name in args.defence:
        rows = []
        for params in D.GRIDS[name]:
            m = T.metrics(*predict_split(D.make_predictor(name, model, params), val_ds))
            rows.append({**params, **m, "drop": round(base["accuracy"] - m["accuracy"], 4)})
            print(f"{name} {params}: val_acc={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f} "
                  f"drop={rows[-1]['drop']:+.4f}", flush=True)
        no_defence_row = {**{k: "" for k in D.GRIDS[name][0]}, **base, "drop": 0.0}
        for folder in (CHECKPOINTS, TUNING):
            T.write_csv(folder / f"{name}_grid.csv", [no_defence_row] + rows)

        chosen, within = choose(rows, base["accuracy"])
        if not within:
            print(f"{name}: no setting within tolerance, using the mildest one.", flush=True)
        chosen_params = {k: chosen[k] for k in D.GRIDS[name][0]}
        if name == D.GAUSSIAN:
            chosen_params.update(samples=D.SMOOTHING_SAMPLES, seed=D.SMOOTHING_SEED)
        T.save_choice(name, {
            "type": "preprocessing (test time, frozen Phase 1 model)",
            "params": chosen_params,
            "calibration_model": relative(model_path),
            "val_accuracy": round(chosen["accuracy"], 4),
            "val_macro_f1": round(chosen["macro_f1"], 4),
            "no_defence_val_accuracy": round(base["accuracy"], 4),
            "within_tolerance": within,
            "selected_on": f"strongest setting with clean val accuracy drop <= {TOLERANCE}",
        })
        summary.append({"defence": name, "params": chosen_params,
                        "val_accuracy": round(chosen["accuracy"], 4),
                        "val_macro_f1": round(chosen["macro_f1"], 4),
                        "no_defence_val_accuracy": round(base["accuracy"], 4),
                        "drop": chosen["drop"], "within_tolerance": within})
        print(f"Chosen {name}: {chosen_params}", flush=True)

    # keep rows of defences not re-tuned in this call (e.g. --defence median_smoothing)
    summary_csv = TUNING / "summary.csv"
    if summary_csv.exists():
        with summary_csv.open(encoding="utf-8") as f:
            done = {r["defence"] for r in summary}
            summary = [r for r in csv.DictReader(f) if r["defence"] not in done] + summary
    T.write_csv(summary_csv, sorted(summary, key=lambda r: list(D.GRIDS).index(r["defence"])))
    print(f"Tuning runs in {CHECKPOINTS}, tuning record in {TUNING}", flush=True)


if __name__ == "__main__":
    main()
