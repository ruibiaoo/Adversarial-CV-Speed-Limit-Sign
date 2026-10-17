"""Shared helpers for the scripts in this folder (not a defence itself).

The folder name has a hyphen, so it is not an importable package. Each script
in here is run directly (python modules/augmentation-based/<name>.py) and
imports this file as a sibling module.

train_grid() trains one model per parameter setting with the Phase 1 recipe
(same model, data, albumentations, optimizer, class weights, seed, LR schedule
and early stopping as Phase 1). Only the per-batch defence changes.
Only train/ is used for training and clean val/ for checkpoint and parameter
choice. No Phase 2 image is read.
"""
import csv
import gc
import json
import sys
import time
from pathlib import Path

PHASE3 = Path(__file__).resolve().parents[2]
if str(PHASE3) not in sys.path:
    sys.path.insert(0, str(PHASE3))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from tensorflow.keras.applications.resnet import preprocess_input  # noqa: E402

from modules.augmentations import build_augmenter  # noqa: E402
from modules.config_loader import load_config  # noqa: E402,F401
from modules.data import compute_class_weights  # noqa: E402
from modules.model import build_model, get_optimizer  # noqa: E402

# Frozen settings live next to the module that writes them: augmentation choices
# in modules/augmentation-based/, preprocessing choices in modules/preprocessing/.
TUNED_AUGMENTATION = PHASE3 / "modules" / "augmentation-based" / "tuned_defences.yaml"
TUNED_PREPROCESSING = PHASE3 / "modules" / "preprocessing" / "tuned_defences.yaml"
TUNING = PHASE3 / "tuning_logs" / "augmentation"  # per-epoch histories + loss report


def setup_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_split(split, config, shuffle):
    """Images as uint8 0-255 (no preprocess_input yet), unbatched."""
    size = int(config["image_size"])
    return tf.keras.utils.image_dataset_from_directory(
        Path(config["data"]) / split,
        labels="inferred",
        label_mode="int",
        class_names=config["class_names"],
        image_size=(size, size),
        batch_size=None,
        shuffle=shuffle,
        seed=int(config["seed"]),
    )


def make_train_ds(config):
    """Train set with the Phase 1 albumentations, batched, still float 0-255."""
    size = int(config["image_size"])
    augmenter = build_augmenter(config, image_size=size)

    def aug(image, label):
        def _aug(img):
            return augmenter(image=np.array(img).astype("uint8"))["image"].astype("uint8")

        out = tf.numpy_function(_aug, [image], tf.uint8)
        out.set_shape((size, size, 3))
        return out, label

    ds = load_split(config["splits"]["train"], config, shuffle=True)
    if augmenter is not None:
        ds = ds.map(aug, num_parallel_calls=1)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.shuffle(1000, seed=int(config["seed"])).batch(int(config["batch"]))


def val_batches(config):
    """Val split (from data.yaml), float 0-255, batched, fixed order."""
    return load_split(config["splits"]["val"], config, shuffle=False).map(
        lambda x, y: (tf.cast(x, tf.float32), y)
    ).batch(int(config["batch"]))


def evaluate(model, ds, transform=None):
    """Loss, accuracy and macro F1 on a float 0-255 dataset. transform: optional uint8 -> uint8 fn.

    loss is plain (unweighted) cross entropy on the clean labels, so it is
    comparable across defences, unlike each defence's own training loss.
    """
    true, probs = [], []
    for x, y in ds:
        x = x.numpy()
        if transform is not None:
            x = np.stack([transform(im.astype(np.uint8)) for im in x]).astype(np.float32)
        probs.append(model.predict_on_batch(preprocess_input(x)))
        true.extend(y.numpy())
    return metrics(true, np.concatenate(probs))


def metrics(true, probs):
    """Loss, accuracy and macro F1 from integer labels and BxC class scores."""
    true = np.asarray(true)
    pred = np.argmax(probs, axis=1)
    return {
        "loss": float(np.mean(-np.log(np.clip(probs[np.arange(len(true)), true], 1e-7, 1.0)))),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
    }


def class_weight_vector(config, num_classes):
    cw = compute_class_weights(config, config["class_names"]) or {}
    return tf.constant([cw.get(i, 1.0) for i in range(num_classes)], tf.float32)


def train_one(name, make_step, config, tuning_csv=None):
    """Train one model. make_step(model, optimizer, class_weight, num_classes) -> tf.function(x, y) -> loss.

    Saves checkpoints/augmentation/<name>/best.keras (best clean-val macro F1) and history.csv,
    and a copy of the history at tuning_csv if given.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(int(config["seed"]))

    num_classes = len(config["class_names"])
    train_ds = make_train_ds(config)
    val_ds = val_batches(config)
    model = build_model(config, num_classes)
    optimizer = get_optimizer(config)
    train_step = make_step(model, optimizer, class_weight_vector(config, num_classes), num_classes)

    save_dir = Path(config["checkpoints_dir"]) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best, best_epoch, bad_epochs, lr_bad = -1.0, 0, 0, 0
    for epoch in range(int(config["epochs"])):
        start = time.time()
        losses = [float(train_step(x, y)) for x, y in train_ds]
        val = evaluate(model, val_ds)
        lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"], "lr": lr,
                        "seconds": round(time.time() - start, 1)})
        print(f"[{name}] epoch {epoch + 1}: train_loss={np.mean(losses):.4f} val_loss={val['loss']:.4f} "
              f"val_acc={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f} "
              f"lr={lr:.2e} ({time.time() - start:.0f}s)", flush=True)

        # same rules as the Phase 1 callbacks
        if val["macro_f1"] > best:
            best, best_epoch, bad_epochs, lr_bad = val["macro_f1"], epoch + 1, 0, 0
            model.save(save_dir / "best.keras")
            continue
        bad_epochs += 1
        lr_bad += 1
        if lr_bad >= int(config["reduce_lr_patience"]):
            new_lr = max(lr * float(config["reduce_lr_factor"]), float(config["min_learning_rate"]))
            tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
            lr_bad = 0
        if bad_epochs >= int(config["early_stopping_patience"]):
            print(f"[{name}] early stopping", flush=True)
            break

    write_csv(save_dir / "history.csv", history)
    if tuning_csv is not None:
        write_csv(tuning_csv, history)
    best_row = history[best_epoch - 1]
    print(f"[{name}] best epoch {best_epoch}: val_loss={best_row['val_loss']:.4f} "
          f"val_acc={best_row['val_accuracy']:.4f} val_macro_f1={best:.4f}", flush=True)
    return {"run": name, "best_epoch": best_epoch, "val_loss": best_row["val_loss"],
            "val_accuracy": best_row["val_accuracy"], "val_macro_f1": best, "checkpoint": str((save_dir / "best.keras").relative_to(PHASE3))}


def finished_run(name, config, tuning_csv=None):
    """Result of an already finished run (history.csv is written last), so a crashed grid can resume."""
    save_dir = Path(config["checkpoints_dir"]) / name
    if not ((save_dir / "history.csv").exists() and (save_dir / "best.keras").exists()):
        return None
    with (save_dir / "history.csv").open(encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    if "val_loss" not in history[0]:  # older run without loss logging: train again
        return None
    if tuning_csv is not None:
        write_csv(tuning_csv, history)
    best_row = max(history, key=lambda r: float(r["val_macro_f1"]))  # first max, as in train_one
    print(f"[{name}] already trained, reusing best epoch {best_row['epoch']}", flush=True)
    return {"run": name, "best_epoch": int(best_row["epoch"]),
            "val_loss": float(best_row["val_loss"]),
            "val_accuracy": float(best_row["val_accuracy"]),
            "val_macro_f1": float(best_row["val_macro_f1"]),
            "checkpoint": str((save_dir / "best.keras").relative_to(PHASE3))}


def train_grid(defence, grid, make_step_for, config):
    """Train one model per params dict in grid, pick the best clean-val macro F1.

    Ties go to the earlier (milder) setting. Writes checkpoints/augmentation/<defence>_grid.csv
    (copied with the histories to tuning_logs/augmentation/<defence>/) and
    freezes the winner in modules/augmentation-based/tuned_defences.yaml.
    """
    results = []
    for params in grid:
        setting = "_".join(f"{k}{v}" for k, v in params.items())
        name = f"{defence}_{setting}"
        tuning_csv = TUNING / defence / f"{setting}_history.csv"
        print(f"\n===== {name} {params} =====", flush=True)
        row = (finished_run(name, config, tuning_csv)
               or train_one(name, make_step_for(params), config, tuning_csv))
        results.append({**params, **row})
        write_csv(Path(config["checkpoints_dir"]) / f"{defence}_grid.csv", results)
        write_csv(TUNING / defence / "grid.csv", results)

    chosen = max(results, key=lambda r: r["val_macro_f1"])  # first max wins ties
    save_choice(defence, {
        "params": {k: chosen[k] for k in grid[0]},
        "checkpoint": chosen["checkpoint"],
        "val_accuracy": round(chosen["val_accuracy"], 4),
        "val_macro_f1": round(chosen["val_macro_f1"], 4),
        "selected_on": "clean val macro F1",
    }, TUNED_AUGMENTATION)
    print(f"\nChosen {defence}: {json.dumps(chosen)}", flush=True)
    return chosen


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_choice(defence, entry, path):
    """Freeze one defence's chosen parameters in its group's tuned_defences.yaml.

    path is TUNED_AUGMENTATION or TUNED_PREPROCESSING; other entries already in
    that file are kept.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = {}
    if path.exists():
        configs = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    configs[defence] = entry
    path.write_text(yaml.safe_dump(configs, sort_keys=False), encoding="utf-8")
