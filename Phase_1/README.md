Folder for Phase 1 of the project.

- Goal is to just train a simple classifier on data_processed
- Must get the metrics

## Run W&B-tracked tuning in Docker

1. From `Phase_1`, copy `.env.example` to `.env` and set `WANDB_API_KEY` to an API key created in your Weights & Biases account. Keep the key only in `.env`; do not embed or commit secrets.
2. Start the container workflow:

   ```bash
   docker compose up --build
   ```

3. View the resulting runs at https://wandb.ai/hongyaok-hongyaok/adversarial-cv-speed-limit-sign.

Each Ray trial is recorded as a W&B run, while Ray Tune and Optuna remain responsible for optimization. The default configuration schedules 200 GPU trials; lower `iterations` first for a smoke test.

## How the code works

- `train.py` reads both YAML files and checks `tuner` in `tuner.yaml`.
- With `tuner: true`, it prepares the search space and calls `model.tune()`.
- With `tuner: false`, it calls `model.train()` using `config.yaml`, then prints test top-1 and top-5 accuracy. W&B and Ray reporting are disabled in this mode.
- `modules/tuning_config.py` converts the YAML `uniform` and `choice` entries into Ray search values.
- `modules/wandb_config.py` checks the W&B settings and sets the project and entity environment variables.

## W&B setup

Keep the API key in `.env` for Docker Compose and the project and entity under `wandb` in `tuner.yaml`. You can also supply the entity through `WANDB_ENTITY`. When running Python directly, set `WANDB_API_KEY` in your shell first; Python does not automatically read `.env`.

Ultralytics provides a Python function for enabling its integrations:

```python
from ultralytics import settings

settings.update({"wandb": True, "raytune": True})
```

`train.py` does this before creating the model, using the selected training mode. This updates the same persistent settings as `yolo settings`, without launching a subprocess. See the [Ultralytics settings documentation](https://docs.ultralytics.com/usage/settings/).

W&B reads `WANDB_API_KEY` automatically, and the built-in integration manages the runs. No extra `wandb.login()`, `wandb.init()`, custom callback, or W&B Sweep is needed for this workflow. See the [W&B environment variable documentation](https://docs.wandb.ai/models/track/environment-variables).

## Find the best tuning run

Run `train.py` as usual with `tuner: true`. Tuning sets Ultralytics' persistent `runs_dir` to `Phase_1/tuning`. With the existing relative W&B project name, Ray logs and individual training outputs are saved beneath that folder. The W&B project stays the same. Normal training still uses the explicit output project in `config.yaml`.

After each search returns, `modules/tuning_results.py` saves a new summary:

```text
Phase_1/tuning/summaries/<date_time>/
    leaderboard.csv
    best_run.yaml
    best_hyperparameters.yaml
```

- Open `leaderboard.csv` to compare trials, ordered by validation top-1 accuracy with the best first. Each parameter has its own column.
- Open `best_run.yaml` for the winning score, parameters, and Ray trial directory. This directory contains Ray logs; it is not necessarily the folder containing YOLO's `best.pt`.
- Use `best_hyperparameters.yaml` to copy the selected values into `config.yaml` for a normal training run. It contains the sampled parameters and Ray's dataset argument, not every fixed training setting.

Ranking uses each trial's final reported `metrics/accuracy_top1`, not test accuracy or its highest historical epoch score. Failed trials and non-finite or missing scores are excluded. Trials stopped early by Ray's scheduler can still appear. Summaries are written after the search returns, not updated live, and each search has a separate dated folder.

The existing Docker Compose mount makes `/app/tuning` available as `Phase_1/tuning` on your computer. The console also prints the best score and summary location. This uses Ray's [returned trial results](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.ResultGrid.html), without another tracking service.

## Numbered W&B run names

The `name` in `tuner.yaml` is the prefix. After tuning finishes and the local summary is saved, `rename_wandb_runs()` names the W&B runs in creation order:

```text
yolo26_speed_limit_tune_run_id_1
yolo26_speed_limit_tune_run_id_2
yolo26_speed_limit_tune_run_id_3
```

Each search gets a unique W&B group, and numbering starts at 1 within that group. Filter by **Group** in W&B to see one search. All runs in that group are numbered, including failed runs that reached W&B; the number is not an accuracy ranking.

During training, names still use Ultralytics' original Ray trial IDs. The short naming function uses the [W&B API to rename runs](https://docs.wandb.ai/models/runs/run-identifiers) after the search returns. It changes the displayed run names; W&B IDs, URLs, local folders, and the original `name` stored in each run's configuration stay the same. Older search groups are not renamed.

Renaming requires an online W&B connection and permission to update these runs. Offline or disabled W&B skips renaming. If the search is interrupted before returning, the runs keep their original names. Your local tuning summary is saved before the renaming step.
