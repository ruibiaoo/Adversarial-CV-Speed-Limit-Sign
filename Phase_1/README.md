Folder for Phase 1 of the project.

- Goal is to just train a simple classifier on data_processed
- Must get the metrics

### !!BEST MODEL IS NOW `"/app/runs/tf_resnet50_speed_limit_aug/best/best.keras"`!!

```
              precision    recall  f1-score   support

      10_MPH   1.000000  1.000000  1.000000         7
      20_MPH   1.000000  1.000000  1.000000        28
      30_MPH   0.968750  1.000000  0.984127        31
      40_MPH   1.000000  1.000000  1.000000        46
      50_MPH   1.000000  0.888889  0.941176         9
      60_MPH   1.000000  1.000000  1.000000         5
      70_MPH   1.000000  1.000000  1.000000         5

    accuracy                       0.992366       131
   macro avg   0.995536  0.984127  0.989329       131
weighted avg   0.992605  0.992366  0.992202       131


Confusion matrix

Actual \ Predicted      10_MPH      20_MPH      30_MPH      40_MPH      50_MPH      60_MPH      70_MPH
10_MPH                 7           0           0           0           0           0           0
20_MPH                 0          28           0           0           0           0           0
30_MPH                 0           0          31           0           0           0           0
40_MPH                 0           0           0          46           0           0           0
50_MPH                 0           0           1           0           8           0           0
60_MPH                 0           0           0           0           0           5           0
70_MPH                 0           0           0           0           0           0           5

```

## Others

Archived YOLO baseline is in `archive/yolo/`.
It has the old `train.py`, `config.yaml`, `tuner.yaml`,
`modules/`, `Dockerfile`, `runs/` and `tuning/summaries/`.
Kept so Phase 2 can still compare against YOLO.

## Run W&B-tracked training in Docker

1. From `Phase_1`, copy `.env.example` to `.env` and set `WANDB_API_KEY` to an API key created in your Weights & Biases account. Keep the key only in `.env`; do not embed or commit secrets.

2. Edit `config.yaml` if needed. To change model, just change `model_name` (default `resnet50`). Make sure `run.sh` is calling `train.py`. 

3. Start the container workflow:

   ```bash
   docker compose up
   ```

4. View the resulting runs at https://wandb.ai/hongyaok-hongyaok/adversarial-cv-speed-limit-sign.

## Test a trained model

Testing is separate from training so the held-out test set is used only when you explicitly run `test.py`.

1. Edit `test.yaml` and set `model_path` to the required `best.keras`. Set `image_size` to the same image size used to train that model. Keep `class_names` in the same order as training. Modify `run.sh` to call `test.py`.

2. From `Phase_1`, run:

   ```bash
   docker compose up
   ```

3. Open the configured `output_dir` (default: `Phase_1/test_results`). It contains:

   ```text
   confusion_matrix.txt
   confusion_matrix_normalized.txt
   test_metrics.txt
   classification_report.txt
   predictions.csv
   ```

`test_metrics.txt` contains loss, top-1/3/5 accuracy, balanced accuracy, macro/micro/weighted precision, recall and F1, Matthews correlation coefficient, Cohen's kappa, ROC AUC, average precision, and per-class precision, recall, specificity, F1 and support. `predictions.csv` contains every image's true class, predicted class, confidence, and all class probabilities.

Set `tuner: false` in `tuner.yaml` for one normal training run.
Set `tuner: true` to run a W&B sweep instead.
Lower `iterations` first for a smoke test.

## How the code works

- `train.py` reads both YAML files and checks `tuner` in `tuner.yaml`.
- With `tuner: true`, it builds a sweep from `tuner.yaml` and starts it with the Python `wandb.agent()` function. After every trial, the W&B run is finished and old TensorFlow models are cleared from memory.
- With `tuner: false`, it calls `model.fit()` using `config.yaml`, uses the validation set during training, and saves the checkpoint with the best validation macro F1 as `best.keras`. `ReduceLROnPlateau` lowers the learning rate when macro F1 stops improving, and early stopping ends training after the configured patience. Training and validation metrics are tracked in W&B. The test set is not loaded during training.
- `modules/augmentations.py` builds the on-the-fly augmentations from the same values as the old YOLO config. Only applied on train split.
- `modules/data.py` loads only the `train` and `val` folders with `image_dataset_from_directory`.
- `modules/model.py` builds resnet50 (or another name from `config.yaml`) with imagenet weights and a new 7-class head.
- `modules/tuning_config.py` converts the YAML `uniform` and `choice` entries into W&B sweep parameters.
- `modules/wandb_config.py` checks the W&B settings and sets the project and entity environment variables.

## W&B setup

Keep the API key in `.env` for Docker Compose and the project and entity under `wandb` in `tuner.yaml` for tuning and `config.yaml` for training. You can also supply the entity through `WANDB_ENTITY`. When running Python directly, set `WANDB_API_KEY` in your shell first; Python does not automatically read `.env`.

No extra `wandb.login()` is needed in the code. Training calls `wandb.init()` once. Sweeps use `wandb.sweep()` and `wandb agent`, then finish the W&B run and clear Keras memory after every trial, including failed trials. See the [W&B environment variable documentation](https://docs.wandb.ai/models/track/environment-variables).

## Find the best tuning run

Run `train.py` as usual with `tuner: true`. After the sweep finishes, `modules/tuning_results.py` saves a new summary:

```text
Phase_1/tuning/summaries/<date_time>/
    leaderboard.csv
    best_run.yaml
    best_hyperparameters.yaml
```

- Open `leaderboard.csv` to compare trials, ordered by validation macro F1 with the best first.
- Open `best_run.yaml` for the winning score and parameters.
- Use `best_hyperparameters.yaml` to copy the selected values into `config.yaml` for a normal training run.

Ranking uses validation macro F1. Failed trials and missing scores are excluded.

## Augmentations

`config.yaml`, applied on the fly in `tf.data` (train split only; validation uses resize + resnet preprocess only):

- `hsv_h/s/v` -> HueSaturationValue
- `degrees`, `translate`, `scale`, `shear` -> one Affine transform
- `perspective` -> Perspective
- `bgr` -> ChannelShuffle with same prob
- `grayscale` -> ToGray with same prob
- `erasing` -> CoarseDropout with same prob

Order is: decode -> resize to `image_size` -> albumentations aug -> resnet `preprocess_input`.
