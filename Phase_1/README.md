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
