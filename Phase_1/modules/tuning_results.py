"""Save a small leaderboard and the best tuning parameters."""

import csv
import math
import os
from datetime import datetime

import wandb
import yaml


def make_output_directory(tuning_directory):
    """Make one timestamp folder for this sweep, return the path."""
    # one folder per sweep so live updates overwrite the same files
    output_directory = tuning_directory / "summaries" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def clean_params(params):
    """Turn numpy numbers into plain python numbers for csv and yaml."""
    clean = {}
    for k, v in dict(params).items():
        if hasattr(v, "item"):
            try:
                clean[k] = v.item()
            except Exception:
                clean[k] = v
        else:
            clean[k] = v
    return clean


def load_old_rows(output_directory):
    """Read back the leaderboard we saved last time, or return empty list."""
    path = output_directory / "leaderboard.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # csv stores everything as string, turn score back to float
            try:
                row["validation_macro_f1"] = float(row.get("validation_macro_f1", 0))
            except Exception:
                continue
            rows.append(row)
    return rows


def load_old_best(output_directory):
    """Read back the best score we saved last time, or return None."""
    path = output_directory / "best_run.yaml"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        best = yaml.safe_load(f)
    if not best:
        return None
    try:
        return float(best.get("validation_macro_f1"))
    except Exception:
        return None


def write_leaderboard(output_directory, rows):
    """Sort rows by score and overwrite leaderboard.csv, return best row."""
    # best run goes first
    rows.sort(key=lambda r: float(r["validation_macro_f1"]), reverse=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (output_directory / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return rows[0]


def write_best(output_directory, best):
    """Overwrite best_run.yaml and best_hyperparameters.yaml."""
    with (output_directory / "best_run.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(best, f, sort_keys=False)
    # best_hyperparameters is the same but without the run info
    params = best.copy()
    params.pop("wandb_run_id", None)
    params.pop("wandb_run_name", None)
    params.pop("validation_macro_f1", None)
    with (output_directory / "best_hyperparameters.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)


def update_live_results(output_directory, run_id, run_name, score, params):
    """Update the tuning summary after one trial finishes.

    leaderboard.csv is updated every time.
    best files are only overwritten when the new score is better.
    """
    # skip bad scores so they dont break the leaderboard
    if score is None:
        print("No score for this trial, skipping live update.")
        return
    try:
        score = float(score)
    except Exception:
        print("Bad score for this trial, skipping live update.")
        return
    if not math.isfinite(score):
        print("Bad score for this trial, skipping live update.")
        return

    # load what we saved last time and add this new trial
    rows = load_old_rows(output_directory)
    # if same run id somehow runs again, drop the old row first
    rows = [r for r in rows if r.get("wandb_run_id") != run_id]
    new_row = {
        "wandb_run_id": run_id,
        "wandb_run_name": run_name,
        "validation_macro_f1": score,
        **clean_params(params),
    }
    rows.append(new_row)

    best = write_leaderboard(output_directory, rows)

    # only overwrite best files when we beat the old best
    old_best = load_old_best(output_directory)
    if old_best is None or score > old_best:
        write_best(output_directory, best)
        print(f"\nNew best! val_macro_f1: {score:.4f} by {run_name}")
    else:
        print(f"\nTrial done: {score:.4f}, best stays: {old_best:.4f}")
    print(f"Tuning summary updated at: {output_directory}")


def save_tuning_results(tuning_directory):
    output_directory = make_output_directory(tuning_directory)
    rows = []

    if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        print("W&B offline, skipping tuning summary.")
        return

    project = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
    group = os.environ.get("WANDB_RUN_GROUP")
    if not group:
        print("No WANDB_RUN_GROUP set, skipping tuning summary.")
        return

    runs = wandb.Api().runs(project, filters={"group": group}, order="+created_at")
    for run in runs:
        try:
            score = run.summary.get("val_macro_f1")
        except Exception:
            score = None
        if score is None or not math.isfinite(float(score)):
            continue

        # run.config has swept params, convert numpy numbers before yaml
        params = {}
        for k, v in dict(run.config).items():
            params[k] = v.item() if hasattr(v, "item") else v
        rows.append({
            "wandb_run_id": run.id,
            "wandb_run_name": run.name,
            "validation_macro_f1": float(score),
            **params,
        })

    if not rows:
        print("No successful tuning trials with a valid validation macro F1 to save.")
        return

    best = write_leaderboard(output_directory, rows)
    write_best(output_directory, best)

    print(f"\nBest validation macro F1: {best['validation_macro_f1']:.4f}")
    print(f"Tuning summary saved to: {output_directory}")
