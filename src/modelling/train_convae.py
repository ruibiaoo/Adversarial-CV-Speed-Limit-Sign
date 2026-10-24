"""Train and save the Convolutional Autoencoder (ConvAE)."""
from __future__ import annotations

import argparse
import gc
import os
import re
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ADV_PREFIX_RE = re.compile(r"^\d{4}_")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(value: str | Path, base_directory: Path) -> Path:
    """Resolve relative paths against ``src/modelling`` and map ``/app/...`` when running locally."""
    candidate = Path(value)
    if candidate.exists():
        return candidate

    normalized = str(value).replace("\\", "/")
    repo_dir = base_directory.parents[1] if len(base_directory.parents) > 1 else None
    if repo_dir is not None and (
        normalized == "/app/data" or normalized.startswith("/app/data/")
    ):
        rel = normalized[len("/app/data") :].lstrip("/")
        local_data = repo_dir / "data" / "data_updated"
        if (local_data / rel).exists():
            return local_data / rel
    if normalized.startswith("/app/"):
        rel_from_app = normalized[len("/app/") :]
        # Map /app/<data_folder>[/...] to <repo>/data/<data_folder>[/...] if running locally
        if repo_dir is not None and Path(rel_from_app).parts:
            data_candidate = repo_dir / "data" / rel_from_app
            if data_candidate.exists():
                return data_candidate
        return base_directory / rel_from_app

    if candidate.is_absolute():
        return candidate
    if repo_dir is not None and (repo_dir / candidate).exists():
        return repo_dir / candidate
    return base_directory / candidate


def clear_training_memory():
    tf.keras.backend.clear_session()
    gc.collect()


def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


@tf.keras.utils.register_keras_serializable(package="ConvAE")
class L1SSIMLoss(tf.keras.losses.Loss):
    """Combined L1 + SSIM loss: alpha * (1 - SSIM) + (1 - alpha) * L1."""

    def __init__(
        self,
        alpha: float = 0.84,
        max_val: float = 1.0,
        name: str = "l1_ssim_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.max_val = float(max_val)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # L1 error per sample in the batch: shape (batch_size,)
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])

        # SSIM per sample in the batch: shape (batch_size,)
        ssim_val = tf.image.ssim(y_true, y_pred, max_val=self.max_val)
        ssim_loss = 1.0 - ssim_val

        return self.alpha * ssim_loss + (1.0 - self.alpha) * l1

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"alpha": self.alpha, "max_val": self.max_val})
        return config


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
                "ssim": logs.get("ssim_metric"),
                "val_ssim": logs.get("val_ssim_metric"),
            }
        )


def _list_image_files(split_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_pairs(
    clean_split_dir: Path, adv_split_dirs: list[Path]
) -> tuple[list[tuple[str, str]], list[str]]:
    """Pair each adversarial image with its matching clean image in ``clean_split_dir``.

    Clean images are indexed by ``(class_folder_name, stem)``. Adversarial filenames
    may carry a leading ``NNNN_`` index prefix (e.g. ``0000_<stem>.png``) which is
    stripped before matching.

    Returns:
        (adv_to_clean_pairs, clean_paths)
    """
    clean_files = _list_image_files(clean_split_dir)
    if not clean_files:
        raise FileNotFoundError(f"No clean images found in: {clean_split_dir}")

    clean_index: dict[tuple[str, str], Path] = {}
    for clean_path in clean_files:
        key = (clean_path.parent.name, clean_path.stem)
        if key in clean_index:
            raise ValueError(
                f"Duplicate clean stem {key} in {clean_split_dir}: "
                f"{clean_index[key]} vs {clean_path}"
            )
        clean_index[key] = clean_path

    adv_to_clean: list[tuple[str, str]] = []
    unmatched: list[str] = []

    for adv_split_dir in adv_split_dirs:
        adv_files = _list_image_files(adv_split_dir)
        if not adv_files:
            raise FileNotFoundError(f"No adversarial images found in: {adv_split_dir}")
        for adv_path in adv_files:
            cls_name = adv_path.parent.name
            raw_stem = adv_path.stem
            stripped_stem = ADV_PREFIX_RE.sub("", raw_stem)
            clean_match = clean_index.get((cls_name, stripped_stem)) or clean_index.get(
                (cls_name, raw_stem)
            )
            if clean_match is None:
                unmatched.append(str(adv_path))
            else:
                adv_to_clean.append((str(adv_path), str(clean_match)))

    if unmatched:
        examples = ", ".join(unmatched[:5])
        raise ValueError(
            f"Could not match {len(unmatched)} adversarial image(s) to clean targets "
            f"in {clean_split_dir}. Examples: {examples}"
        )

    clean_paths = [str(p) for p in clean_files]
    return adv_to_clean, clean_paths


def _parse_adv_dirs(config: dict, base_directory: Path) -> list[Path]:
    raw_adv = config.get("adv_data")
    if raw_adv is None:
        raw_list = ["/app/data_pgdpixle"]
    elif isinstance(raw_adv, (str, Path)):
        raw_list = [raw_adv] if str(raw_adv).strip() else []
    elif isinstance(raw_adv, (list, tuple)):
        raw_list = [item for item in raw_adv if str(item).strip()]
    else:
        raise ValueError(f"Unsupported adv_data config value: {raw_adv!r}")
    return [resolve_path(item, base_directory) for item in raw_list]


def build_convae_datasets(config: dict, base_directory: Path):
    """Load paired ``(input_rgb01, target_rgb01)`` datasets for ``train`` and ``val``."""
    clean_root = resolve_path(config["data"], base_directory)
    clean_train_dir = clean_root / "train"
    clean_val_dir = clean_root / "val"
    if not clean_train_dir.exists():
        raise FileNotFoundError(f"Clean training split directory not found: {clean_train_dir}")
    if not clean_val_dir.exists():
        raise FileNotFoundError(f"Clean validation split directory not found: {clean_val_dir}")

    adv_roots = _parse_adv_dirs(config, base_directory)
    for adv_root in adv_roots:
        if not (adv_root / "train").exists():
            raise FileNotFoundError(f"Adversarial training split not found: {adv_root / 'train'}")
        if not (adv_root / "val").exists():
            raise FileNotFoundError(f"Adversarial validation split not found: {adv_root / 'val'}")

    image_size = int(config.get("image_size", 256))
    batch = int(config.get("batch", 32))
    seed = int(config.get("seed", 4266))
    workers = int(config.get("workers", 0) or 0)
    input_noise_std = float(config.get("input_noise_std", 0.0) or 0.0)
    clean_pair_ratio = float(config.get("clean_pair_ratio", 0.5) or 0.0)
    if not (0.0 <= clean_pair_ratio < 1.0):
        raise ValueError(f"clean_pair_ratio must be in [0.0, 1.0), got {clean_pair_ratio}")

    parallel = tf.data.AUTOTUNE if workers > 0 else 1

    train_adv_pairs, train_clean_paths = collect_pairs(
        clean_train_dir, [r / "train" for r in adv_roots]
    )
    val_adv_pairs, val_clean_paths = collect_pairs(
        clean_val_dir, [r / "val" for r in adv_roots]
    )

    if train_adv_pairs and clean_pair_ratio > 0.0:
        n_clean = int(round(len(train_adv_pairs) * clean_pair_ratio / (1.0 - clean_pair_ratio)))
        rng = np.random.default_rng(seed)
        if n_clean <= len(train_clean_paths):
            chosen_idx = rng.choice(len(train_clean_paths), size=n_clean, replace=False)
        else:
            chosen_idx = rng.choice(len(train_clean_paths), size=n_clean, replace=True)
        train_clean_pairs = [(train_clean_paths[i], train_clean_paths[i]) for i in chosen_idx]
    elif not train_adv_pairs:
        train_clean_pairs = [(p, p) for p in train_clean_paths]
    else:
        train_clean_pairs = []

    train_pairs = train_adv_pairs + train_clean_pairs
    val_pairs = val_adv_pairs if val_adv_pairs else [(p, p) for p in val_clean_paths]
    val_clean_pairs = [(p, p) for p in val_clean_paths]

    print(
        f"ConvAE train pairs: {len(train_pairs)} total "
        f"({len(train_adv_pairs)} adv->clean + {len(train_clean_pairs)} clean->clean)"
    )
    print(
        f"ConvAE val pairs:   {len(val_pairs)} adv->clean "
        f"(plus {len(val_clean_pairs)} clean->clean for evaluation)"
    )

    augmenter = build_augmenter(
        config, image_size=image_size, additional_targets={"target": "image"}
    )
    if augmenter is None:
        print("ConvAE augmentations OFF: train gets resize + [0, 1] scaling only")
    else:
        print("ConvAE augmentations ON: identical transform applied to input and target")

    def decode_and_resize(path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (image_size, image_size), method="bilinear")
        img.set_shape((image_size, image_size, 3))
        return tf.cast(tf.round(img), tf.uint8)

    def load_pair(in_path: tf.Tensor, tgt_path: tf.Tensor):
        return decode_and_resize(in_path), decode_and_resize(tgt_path)

    def aug_pair_fn(in_img: tf.Tensor, tgt_img: tf.Tensor):
        def _aug(inp, tgt):
            inp_np = np.asarray(inp, dtype=np.uint8)
            tgt_np = np.asarray(tgt, dtype=np.uint8)
            out = augmenter(image=inp_np, target=tgt_np)
            return out["image"].astype(np.uint8), out["target"].astype(np.uint8)

        aug_in, aug_tgt = tf.numpy_function(
            _aug, [in_img, tgt_img], [tf.uint8, tf.uint8]
        )
        aug_in.set_shape((image_size, image_size, 3))
        aug_tgt.set_shape((image_size, image_size, 3))
        return aug_in, aug_tgt

    def scale_train_pair(in_img: tf.Tensor, tgt_img: tf.Tensor):
        inp = tf.clip_by_value(tf.cast(in_img, tf.float32) / 255.0, 0.0, 1.0)
        tgt = tf.clip_by_value(tf.cast(tgt_img, tf.float32) / 255.0, 0.0, 1.0)
        if input_noise_std > 0.0:
            noise = tf.random.normal(
                tf.shape(inp), mean=0.0, stddev=input_noise_std, dtype=tf.float32
            )
            inp = tf.clip_by_value(inp + noise, 0.0, 1.0)
        return inp, tgt

    def scale_val_pair(in_img: tf.Tensor, tgt_img: tf.Tensor):
        inp = tf.clip_by_value(tf.cast(in_img, tf.float32) / 255.0, 0.0, 1.0)
        tgt = tf.clip_by_value(tf.cast(tgt_img, tf.float32) / 255.0, 0.0, 1.0)
        return inp, tgt

    train_in = [p[0] for p in train_pairs]
    train_tgt = [p[1] for p in train_pairs]
    train_ds = tf.data.Dataset.from_tensor_slices((train_in, train_tgt))
    train_ds = train_ds.shuffle(len(train_pairs), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(load_pair, num_parallel_calls=parallel)
    if augmenter is not None:
        train_ds = train_ds.map(aug_pair_fn, num_parallel_calls=parallel)
    train_ds = train_ds.map(scale_train_pair, num_parallel_calls=parallel)
    train_ds = train_ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    val_in = [p[0] for p in val_pairs]
    val_tgt = [p[1] for p in val_pairs]
    val_ds = tf.data.Dataset.from_tensor_slices((val_in, val_tgt))
    val_ds = val_ds.map(load_pair, num_parallel_calls=parallel)
    val_ds = val_ds.map(scale_val_pair, num_parallel_calls=parallel)
    val_ds = val_ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    val_clean_in = [p[0] for p in val_clean_pairs]
    val_clean_tgt = [p[1] for p in val_clean_pairs]
    val_clean_ds = tf.data.Dataset.from_tensor_slices((val_clean_in, val_clean_tgt))
    val_clean_ds = val_clean_ds.map(load_pair, num_parallel_calls=parallel)
    val_clean_ds = val_clean_ds.map(scale_val_pair, num_parallel_calls=parallel)
    val_clean_ds = val_clean_ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    dataset_info = {
        "train_total_pairs": len(train_pairs),
        "train_adv_pairs": len(train_adv_pairs),
        "train_clean_pairs": len(train_clean_pairs),
        "val_adv_pairs": len(val_pairs),
        "val_clean_pairs": len(val_clean_pairs),
    }
    return train_ds, val_ds, val_clean_ds, dataset_info


def compute_identity_baseline(ds) -> dict:
    """Compute MSE, MAE, and PSNR of passing the input directly through as the prediction."""
    mses = []
    maes = []
    psnrs = []
    for batch_inputs, batch_targets in ds:
        diff = batch_inputs - batch_targets
        mse_b = tf.reduce_mean(tf.square(diff), axis=[1, 2, 3]).numpy()
        mae_b = tf.reduce_mean(tf.abs(diff), axis=[1, 2, 3]).numpy()
        psnr_b = tf.image.psnr(batch_targets, batch_inputs, max_val=1.0).numpy()
        mses.append(mse_b)
        maes.append(mae_b)
        psnrs.append(psnr_b)
    if not mses:
        return {"mse": 0.0, "mae": 0.0, "psnr": 0.0}
    mse_all = np.concatenate(mses)
    mae_all = np.concatenate(maes)
    psnr_all = np.concatenate(psnrs)
    finite_psnr = psnr_all[np.isfinite(psnr_all)]
    return {
        "mse": float(np.mean(mse_all)),
        "mae": float(np.mean(mae_all)),
        "psnr": float(np.mean(finite_psnr)) if finite_psnr.size else float("inf"),
    }


def save_reconstruction_grid(
    model: tf.keras.Model, val_ds, output_path: Path, num_samples: int = 3
) -> None:
    """Save a 3-column grid (Adversarial Input | Reconstructed | Clean Target) with ``num_samples`` rows."""
    if num_samples <= 0:
        return

    all_inputs = []
    all_targets = []
    for batch_inputs, batch_targets in val_ds:
        all_inputs.append(batch_inputs.numpy())
        all_targets.append(batch_targets.numpy())

    if not all_inputs:
        return

    inputs_arr = np.concatenate(all_inputs, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    total = inputs_arr.shape[0]
    rows = min(int(num_samples), total)
    # Spread sample indices evenly across validation set so different classes are shown
    indices = np.linspace(0, total - 1, num=rows, dtype=int)
    adv_samples = inputs_arr[indices]
    clean_samples = targets_arr[indices]

    reconstructed = model.predict(adv_samples, verbose=0)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_headers = ["Adversarial Input", "Reconstructed (ConvAE)", "Clean Target"]
    for i in range(rows):
        adv_i = adv_samples[i]
        rec_i = reconstructed[i]
        cln_i = clean_samples[i]

        adv_mse = float(np.mean((adv_i - cln_i) ** 2))
        rec_mse = float(np.mean((rec_i - cln_i) ** 2))
        adv_psnr = float(tf.image.psnr(cln_i, adv_i, max_val=1.0).numpy())
        rec_psnr = float(tf.image.psnr(cln_i, rec_i, max_val=1.0).numpy())

        axes[i, 0].imshow(adv_i)
        axes[i, 0].set_title(
            f"{col_headers[0]} #{i + 1}\nMSE={adv_mse:.4f} | PSNR={adv_psnr:.1f}dB",
            fontsize=9,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rec_i)
        axes[i, 1].set_title(
            f"{col_headers[1]} #{i + 1}\nMSE={rec_mse:.4f} | PSNR={rec_psnr:.1f}dB",
            fontsize=9,
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(cln_i)
        axes[i, 2].set_title(f"{col_headers[2]} #{i + 1}", fontsize=9)
        axes[i, 2].axis("off")

    fig.suptitle(
        "ConvAE Validation Samples: Adversarial vs Reconstructed vs Clean",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved {rows}x{cols} validation reconstruction grid to: {output_path}")


def train_convae(cfg: dict, base_directory: Path, track: bool = False) -> dict:
    """Train the ConvAE from a config dictionary and save checkpoints + diagnostics."""
    seed = int(cfg.get("seed", 4266))
    tf.keras.utils.set_random_seed(seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    train_ds, val_ds, val_clean_ds, dataset_info = build_convae_datasets(
        cfg, base_directory
    )

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
    elif loss_name in ("l1_ssim", "ssim_l1", "l1+ssim"):
        ssim_alpha = float(cfg.get("ssim_alpha", 0.84))
        loss_fn = L1SSIMLoss(alpha=ssim_alpha, max_val=1.0)
    else:
        raise ValueError(f"Unsupported ConvAE loss: {loss_name!r} (expected 'mse' or 'mae' or 'ls_ssim')")

    optimizer = get_optimizer(cfg)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["mse", "mae", psnr_metric, ssim_metric],
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
    clean_eval_results = model.evaluate(val_clean_ds, verbose=0, return_dict=True)
    identity_metrics = compute_identity_baseline(val_ds)
    val_Losses = history.history.get("val_loss", [])
    best_val_loss = float(min(val_Losses)) if val_Losses else float(eval_results.get("loss", 0.0))

    summary = {
        "best_val_loss": best_val_loss,
        "val_loss": float(eval_results.get("loss", 0.0)),
        "val_mse": float(eval_results.get("mse", 0.0)),
        "val_mae": float(eval_results.get("mae", 0.0)),
        "val_psnr": float(eval_results.get("psnr_metric", 0.0)),
        "val_ssim": float(eval_results.get("ssim_metric", 0.0)),
        "val_clean_ssim": float(clean_eval_results.get("ssim_metric", 0.0)),
        "identity_baseline_mse": identity_metrics["mse"],
        "identity_baseline_mae": identity_metrics["mae"],
        "identity_baseline_psnr": identity_metrics["psnr"],
        "val_clean_mse": float(clean_eval_results.get("mse", 0.0)),
        "val_clean_mae": float(clean_eval_results.get("mae", 0.0)),
        "val_clean_psnr": float(clean_eval_results.get("psnr_metric", 0.0)),
        **dataset_info,
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

    save_samples = int(cfg.get("save_samples", 3) or 0)
    if save_samples > 0:
        save_reconstruction_grid(
            model,
            val_ds,
            save_dir / "reconstructions.png",
            num_samples=save_samples,
        )

    print("\n--- CONVAE TRAINING COMPLETE ---")
    print(f"Best validation loss ({loss_name}): {best_val_loss:.6f}")
    print(f"Validation MSE (adv->clean):       {summary['val_mse']:.6f}")
    print(f"Validation MAE (adv->clean):       {summary['val_mae']:.6f}")
    print(f"Validation PSNR (adv->clean):      {summary['val_psnr']:.2f} dB")
    print(f"Identity baseline PSNR (raw adv):  {summary['identity_baseline_psnr']:.2f} dB")
    print(f"Validation PSNR (clean->clean):    {summary['val_clean_psnr']:.2f} dB")
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
