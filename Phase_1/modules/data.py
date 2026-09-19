"""Load train/val folders, aug only on train, on the fly."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

from .augmentations import build_augmenter


def compute_class_weights(config, class_names):
    """Count train images per class and make balanced weights.

    weight = total / (num_classes * count)
    So rare classes get bigger weight. Returns None if turned off.
    """
    mode = str(config.get("class_weight", "none")).lower()
    if mode in ("none", "off", "false", ""):
        return None

    train_dir = Path(config["data"]) / "train"
    counts = []
    for name in class_names:
        folder = train_dir / name
        if not folder.exists():
            counts.append(0)
            continue
        # count image files only
        n = sum(1 for path in folder.iterdir() if path.is_file())
        counts.append(n)

    total = sum(counts)
    num_classes = len(class_names)
    weights = {}
    for i, n in enumerate(counts):
        if n > 0:
            weights[i] = total / (num_classes * n)
        else:
            weights[i] = 1.0

    print(f"Train counts: {dict(zip(class_names, counts))}")
    print(f"Class weights: {weights}")
    return weights


def build_datasets(config):
    """Make tf.data datasets from data/train and data/val folders."""
    data_dir = config["data"]
    image_size = int(config.get("image_size", 224))
    batch = int(config.get("batch", 32))
    seed = int(config.get("seed", 4266))
    workers = int(config.get("workers", 0) or 0)

    parallel = tf.data.AUTOTUNE if workers > 0 else 1

    augmenter = build_augmenter(config, image_size=image_size)
    if augmenter is None:
        print("Augmentations OFF: train gets resize + preprocess only")
    else:
        print("Augmentations ON: train gets on-the-fly aug")

    def aug_fn(image, label):
        def _aug(img):
            img = np.array(img).astype("uint8")
            out = augmenter(image=img)["image"]
            return out.astype("uint8")

        aug_img = tf.numpy_function(_aug, [image], tf.uint8)
        aug_img.set_shape((image_size, image_size, 3))
        return aug_img, label

    def prep_fn(image, label):
        image = preprocess_input(tf.cast(image, tf.float32))
        return image, label

    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=None,
        shuffle=True,
        seed=seed,
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # train gets aug on the fly, then preprocess
    # augment false = resize + preprocess only, same as val
    if augmenter is not None:
        train_ds = train_ds.map(aug_fn, num_parallel_calls=parallel)
    train_ds = train_ds.map(prep_fn, num_parallel_calls=parallel)
    train_ds = train_ds.shuffle(1000, seed=seed).batch(batch).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=None,
        shuffle=False,
    )
    val_ds = val_ds.map(prep_fn, num_parallel_calls=parallel)
    val_ds = val_ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes
