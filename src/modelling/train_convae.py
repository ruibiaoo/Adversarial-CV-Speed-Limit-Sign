"""Train and save the Convolutional Autoencoder (ConvAE)."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
import yaml

if __package__:
    from .modules.augmentations import build_augmenter
    from .modules.convae import _is_true, build_autoencoder
    from .modules.model import get_optimizer
    from .modules.wandb_config import prepare_wandb_training
else:
    from modules.augmentations import build_augmenter
    from modules.convae import _is_true, build_autoencoder
    from modules.model import get_optimizer
    from modules.wandb_config import prepare_wandb_training


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(value: str | Path, base_directory: Path) -> Path:
    """Resolve relative paths against ``src/modelling`` and map ``/app/...`` when running locally."""
    candidate = Path(value)
    if candidate.exists():
        return candidate

    normalized = str(value).replace("\\", "/")
    if normalized == "/app/data" or normalized.startswith("/app/data/"):
        repo_dir = base_directory.parents[1]
        rel = normalized[len("/app/data") :].lstrip("/")
        local_data = repo_dir / "data" / "data_updated"
        if (local_data / rel).exists():
            return local_data / rel
    if normalized.startswith("/app/"):
        rel_from_app = normalized[len("/app/") :]
        return base_directory / rel_from_app

    if candidate.is_absolute():
        return candidate
    return base_directory / candidate


def clear_training_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


class WandbAutoencoderLog(tf.keras.callbacks.Callback):
    """Log ConvAE training and validation metrics to W&B after each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        learning_rate = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "loss": logs.get("loss"),
                "mse": logs.get("mse"),
                "mae": logs.get("mae"),
                "psnr": logs.get("psnr_metric"),
                "val_loss": logs.get("val_loss"),
                "val_mse": logs.get("val_mse"),
                "val_mae": logs.get("val_mae"),
                "val_psnr": logs.get("val_psnr_metric"),
            }
        )


def build_convae_datasets(config: dict, base_directory: Path):
    """Load ``train`` and ``val`` image folders as ``(input_rgb01, target_rgb01)`` pairs."""
    data_dir = resolve_path(config["data"], base_directory)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training split directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation split directory not found: {val_dir}")

    image_size = int(config.get("image_size", 256))
    batch = int(config.get("batch", 32))
    seed = int(config.get("seed", 4266))
    workers = int(config.get("workers", 0) or 0)
    input_noise_std = float(config.get("input_noise_std", 0.0) or 0.0)

    parallel = tf.data.AUTOTUNE if workers > 0 else 1

    augmenter = build_augmenter(config, image_size=image_size)
    if augmenter is None:
        print("ConvAE augmentations OFF: train gets resize + [0, 1] scaling only")
    else:
        print("ConvAE augmentations ON: train gets on-the-fly aug")

    def aug_fn(image):
        def _aug(img):
            img = np.array(img).astype("uint8")
            out = augmenter(image=img)["image"]
            return out.astype("uint8")

        aug_img = tf.numpy_function(_aug, [image], tf.uint8)
        aug_img.set_shape((image_size, image_size, 3))
        return aug_img

    def make_train_pair(image):
        clean = tf.clip_by_value(tf.cast(image, tf.float32) / 255.0, 0.0, 1.0)
        if input_noise_std > 0.0:
            noise = tf.random.normal(
                tf.shape(clean), mean=0.0, stddev=input_noise_std, dtype=tf.float32
            )
            noisy = tf.clip_by_value(clean + noise, 0.0, 1.0)
            return noisy, clean
        return clean, clean

    def make_val_pair(image):
        clean = tf.clip_by_value(tf.cast(image, tf.float32) / 255.0, 0.0, 1.0)
        return clean, clean

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels=None,
        label_mode=None,
        image_size=(image_size, image_size),
        batch_size=None,
        shuffle=True,
        seed=seed,
    )
    if augmenter is not None:
        train_ds = train_ds.map(aug_fn, num_parallel_calls=parallel)
    train_ds = train_ds.map(make_train_pair, num_parallel_calls=parallel)
    train_ds = (
        train_ds.shuffle(1000, seed=seed).batch(batch).prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels=None,
        label_mode=None,
        image_size=(image_size, image_size),
        batch_size=None,
        shuffle=False,
    )
    val_ds = val_ds.map(make_val_pair, num_parallel_calls=parallel)
    val_ds = val_ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def save_reconstruction_grid(
    model: tf.keras.Model, val_ds, output_path: Path, num_samples: int = 8
) -> None:
    """Save a side-by-side grid of original vs reconstructed validation images."""
    if num_samples <= 0:
        return

    originals = []
    for batch_inputs, batch_targets in val_ds:
        originals.append(batch_targets.numpy())
        if sum(arr.shape[0] for arr in originals) >= num_samples:
            break

    if not originals:
        return

    samples = np.concatenate(originals, axis=0)[:num_samples]
    reconstructed = model.predict(samples, verbose=0)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    cols = samples.shape[0]
    fig, axes = plt.subplots(2, cols, figsize=(2.5 * cols, 5.2))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(cols):
        axes[0, i].imshow(samples[i])
        axes[0, i].set_title(f"Original #{i + 1}", fontsize=9)
        axes[0, i].axis("off")

        mse_i = float(np.mean((samples[i] - reconstructed[i]) ** 2))
        axes[1, i].imshow(reconstructed[i])
        axes[1, i].set_title(f"Recon (MSE={mse_i:.4f})", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle("ConvAE Validation Reconstructions", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved validation reconstruction preview to: {output_path}")


def train_convae(cfg: dict, base_directory: Path, track: bool = False) -> dict:
    """Train the ConvAE from a config dictionary and save checkpoints + diagnostics."""
    seed = int(cfg.get("seed", 4266))
    tf.keras.utils.set_random_seed(seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    train_ds, val_ds = build_convae_datasets(cfg, base_directory)

    image_size = int(cfg.get("image_size", 256))
    hidden_dim = int(cfg.get("hidden_dim", 1024))
    latent_dim = int(cfg.get("latent_dim", 256))
    noise_std = float(cfg.get("noise_std", 0.2))

    model = build_autoencoder(
        img_size=image_size,
        channels=3,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        noise_std=noise_std,
    )

    loss_name = str(cfg.get("loss", "mse")).strip().lower()
    if loss_name in ("mse", "mean_squared_error"):
        loss_fn = "mse"
    elif loss_name in ("mae", "mean_absolute_error", "l1"):
        loss_fn = "mae"
    else:
        raise ValueError(f"Unsupported ConvAE loss: {loss_name!r} (expected 'mse' or 'mae')")

    optimizer = get_optimizer(cfg)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["mse", "mae", psnr_metric],
    )

    project_dir = resolve_path(cfg.get("project", "/app/runs/convae"), base_directory)
    run_name = str(cfg.get("name", "") or "").strip()
    save_dir = project_dir / run_name if run_name else project_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configuration used for reproducibility.
    with (save_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    epochs = int(cfg.get("epochs", 200))
    reduce_lr_factor = float(cfg.get("reduce_lr_factor", 0.5))
    reduce_lr_patience = int(cfg.get("reduce_lr_patience", 5))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 15))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 1e-5))
    min_learning_rate = float(cfg.get("min_learning_rate", 1e-6))

    best_checkpoint_path = save_dir / "best.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(best_checkpoint_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    if _is_true(cfg.get("save_latest_best", True), default=True):
        checkpoint.monitor_op = lambda current, best: current <= best

    callbacks = [
        checkpoint,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_learning_rate,
            min_delta=early_stopping_min_delta,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(save_dir / "history.csv")),
    ]

    if track:
        callbacks.append(WandbAutoencoderLog())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final state as well (with restore_best_weights=True, this matches best epoch).
    final_path = save_dir / "final.keras"
    model.save(str(final_path))
    if not best_checkpoint_path.exists():
        model.save(str(best_checkpoint_path))

    eval_results = model.evaluate(val_ds, verbose=0, return_dict=True)
    val_Losses = history.history.get("val_loss", [])
    best_val_loss = float(min(val_Losses)) if val_Losses else float(eval_results.get("loss", 0.0))

    summary = {
        "best_val_loss": best_val_loss,
        "val_loss": float(eval_results.get("loss", 0.0)),
        "val_mse": float(eval_results.get("mse", 0.0)),
        "val_mae": float(eval_results.get("mae", 0.0)),
        "val_psnr": float(eval_results.get("psnr_metric", 0.0)),
        "epochs_ran": len(history.history.get("loss", [])),
        "best_checkpoint": str(best_checkpoint_path),
    }

    with (save_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("CONVAE VALIDATION METRICS\n")
        f.write("=========================\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

    save_samples = int(cfg.get("save_samples", 8) or 0)
    if save_samples > 0:
        save_reconstruction_grid(
            model,
            val_ds,
            save_dir / "reconstructions.png",
            num_samples=save_samples,
        )

    print("\n--- CONVAE TRAINING COMPLETE ---")
    print(f"Best validation loss ({loss_fn}): {best_val_loss:.6f}")
    print(f"Validation MSE:  {summary['val_mse']:.6f}")
    print(f"Validation MAE:  {summary['val_mae']:.6f}")
    print(f"Validation PSNR: {summary['val_psnr']:.2f} dB")
    print(f"Best ConvAE saved to: {best_checkpoint_path}")

    return summary


def parse_args():
    phase_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the Convolutional Autoencoder")
    parser.add_argument(
        "--config",
        default=str(phase_directory / "configs" / "train_convae.yaml"),
        help="Path to the ConvAE training YAML configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phase_directory = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = phase_directory / config_path

    config = load_config(config_path)

    wandb_cfg = config.get("wandb") or {}
    wandb_enabled = _is_true(wandb_cfg.get("enabled", True), default=True)
    track = prepare_wandb_training(config) if wandb_enabled else False
    if not wandb_enabled:
        config.pop("wandb", None)

    if track:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=config.get("name"),
            group=os.environ.get("WANDB_RUN_GROUP"),
            config=config,
            job_type="train_convae",
        )

    try:
        train_convae(config, base_directory=phase_directory, track=track)
    finally:
        if track:
            wandb.finish()
        clear_training_memory()


if __name__ == "__main__":
    main()
