"""Save a small leaderboard and the best tuning parameters."""

import csv
from datetime import datetime
import math

import yaml


def save_tuning_results(results, tuning_directory):
    output_directory = tuning_directory / "summaries" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []

    for trial in results:
        accuracy = trial.metrics.get("metrics/accuracy_top1")
        if trial.error or accuracy is None or not math.isfinite(accuracy):
            continue

        # Ray can return NumPy numbers, so convert them before saving YAML.
        parameters = {
            name: value.item() if hasattr(value, "item") else value
            for name, value in trial.config.items()
        }
        rows.append({
            "ray_trial_directory": str(trial.path),
            "top1_accuracy": float(accuracy),
            **parameters,
        })

    if not rows:
        print("No successful tuning trials with a valid top-1 accuracy to save.")
        return

    rows.sort(key=lambda row: row["top1_accuracy"], reverse=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output_directory / "leaderboard.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    best_run = rows[0]
    with (output_directory / "best_run.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(best_run, file, sort_keys=False)

    parameters = best_run.copy()
    parameters.pop("ray_trial_directory")
    parameters.pop("top1_accuracy")
    with (output_directory / "best_hyperparameters.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(parameters, file, sort_keys=False)

    print(f"\nBest validation top-1 accuracy: {best_run['top1_accuracy']:.4f}")
    print(f"Tuning summary saved to: {output_directory}")
