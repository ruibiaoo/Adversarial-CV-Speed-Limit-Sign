"""Evaluate Pixle on the Phase 1 speed-limit-sign classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # Progress display is optional for the attack evaluator.
    tqdm = None

try:
    from .pixle_attack import (
        PixleAttack,
        PixleConfig,
        attack_with_original_l0,
        count_l0,
        parse_patch_size,
    )
except ImportError:  # Allows direct execution from the repository root.
    from pixle_attack import (
        PixleAttack,
        PixleConfig,
        attack_with_original_l0,
        count_l0,
        parse_patch_size,
    )


CLASS_NAMES = [
    "10_MPH",
    "20_MPH",
    "30_MPH",
    "40_MPH",
    "50_MPH",
    "60_MPH",
    "70_MPH",
]


class KerasProbabilityOracle:
    """Adapt a TensorFlow/Keras model to Pixle's black-box interface."""

    def __init__(self, model_path: Path, image_size: int, preprocess: str) -> None:
        import tensorflow as tf

        self.tf = tf
        self.model = tf.keras.models.load_model(model_path)
        self.image_size = image_size
        self.preprocess = preprocess

    def __call__(self, image: np.ndarray) -> np.ndarray:
        batch = self.tf.convert_to_tensor(image[None], dtype=self.tf.float32)
        if self.preprocess == "resnet":
            from keras.applications.resnet import preprocess_input

            batch = preprocess_input(batch)
        elif self.preprocess != "none":
            raise ValueError(f"Unknown preprocessing mode: {self.preprocess}")
        probabilities = self.model(batch, training=False).numpy()
        return probabilities[0]


def load_image(path: Path, image_size: int) -> np.ndarray:
    """Load an image using the same path as Phase 1's test pipeline.

    ``tf.keras.utils.image_dataset_from_directory`` decodes images with
    TensorFlow and resizes them with bilinear interpolation by default. Keep
    those details here so Phase 2 evaluates the model on the same pixel values
    as Phase 1 before applying ResNet preprocessing.
    """
    import tensorflow as tf

    encoded = tf.io.read_file(str(path))
    image = tf.image.decode_image(
        encoded,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(
        image,
        (image_size, image_size),
        method="bilinear",
    )
    image.set_shape((image_size, image_size, 3))
    return image.numpy().astype(np.float32)


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an attack result using the format indicated by ``path``."""
    values = np.asarray(image)
    if np.issubdtype(values.dtype, np.floating):
        if values.min() >= 0 and values.max() <= 1:
            values = values * 255.0
        values = np.clip(values, 0, 255)
    output = Image.fromarray(values.astype(np.uint8), mode="RGB")
    if path.suffix.lower() == ".png":
        output.save(path, format="PNG", optimize=True)
    else:
        output.save(path, format="JPEG", quality=95)


def iter_test_images(data_dir: Path, class_names: list[str]):
    test_dir = data_dir / "test"
    for label, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        # Allow focused evaluations, e.g. a test directory containing only
        # ``50_MPH``. Missing class folders contribute no images.
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                yield path, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/data_updated"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--preprocess", choices=["resnet", "none"], default="resnet")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("src/attack/pixle/results")
    )
    parser.add_argument(
        "--adversarial-dir",
        type=Path,
        default=Path("data/data_updated/pixle_attacked"),
        help="Directory for class-organized adversarial examples.",
    )
    parser.add_argument("--restarts", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--patch-size", type=parse_patch_size, default=3)
    parser.add_argument(
        "--mapping",
        choices=[
            "random",
            "similarity",
            "distance",
            "similarity_distribution",
            "distance_distribution",
        ],
        default="random",
    )
    parser.add_argument(
        "--algorithm",
        choices=["restart-iterative", "iterative"],
        default="restart-iterative",
    )
    parser.add_argument("--operation", choices=["overwrite", "swap"], default="overwrite")
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Deprecated: attacked images are now always saved.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the per-image model-query progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    adversarial_dir = args.adversarial_dir
    adversarial_dir.mkdir(parents=True, exist_ok=True)

    oracle = KerasProbabilityOracle(args.model, args.image_size, args.preprocess)
    config = PixleConfig(
        restarts=args.restarts,
        iterations=args.iterations,
        patch_size=args.patch_size,
        mapping=args.mapping,
        algorithm=args.algorithm,
        operation=args.operation,
        seed=args.seed,
    )
    attacker = PixleAttack(oracle, config)

    rows = []
    attempted = 0
    initially_correct = 0
    successes = 0
    for index, (path, label) in enumerate(iter_test_images(args.data, CLASS_NAMES)):
        if args.max_images is not None and index >= args.max_images:
            break
        image = load_image(path, args.image_size)
        initial_probabilities = oracle(image)
        initial_label = int(np.argmax(initial_probabilities))
        initial_confidence = float(initial_probabilities[initial_label])

        # if prediction is already wrong, we skip the attack
        if initial_label != label:
            rows.append(
                {
                    "file_path": str(path),
                    "actual_class": CLASS_NAMES[label],
                    "initial_class": CLASS_NAMES[initial_label],
                    "adversarial_class": CLASS_NAMES[initial_label],
                    "initial_confidence": initial_confidence,
                    "adversarial_confidence": initial_confidence,
                    "success": False,
                    "status": "skipped_initially_misclassified",
                    "queries": 1,
                    "l0": 0,
                    "saved_image_path": "",
                    "saved_image_type": "not_attacked",
                    "saved_class": CLASS_NAMES[initial_label],
                    "saved_confidence": initial_confidence,
                    "saved_l0": 0,
                }
            )
            continue

        initially_correct += 1
        attempted += 1
        progress_bar = None
        if tqdm is not None and not args.no_progress:
            progress_bar = tqdm(
                total=1 + args.restarts * args.iterations,
                initial=1,
                desc=path.name,
                unit="query",
                leave=False,
            )

        def update_progress(queries: int, _query_limit: int) -> None:
            if progress_bar is None:
                return
            progress_bar.n = queries
            progress_bar.refresh()

        try:
            result = attack_with_original_l0(
                attacker,
                image,
                label,
                initial_probabilities=initial_probabilities,
                progress_callback=update_progress,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        successes += int(result.success)

        if result.success:
            saved_image = result.adversarial_image
            saved_class = result.adversarial_label
            saved_confidence = result.adversarial_confidence
            saved_image_type = "successful_adversarial"
            saved_l0 = result.l0
        else:
            # Preserve the final candidate queried by Pixle, even when it was
            # not the best-scoring candidate and the attack did not succeed.
            saved_image = result.last_perturbed_image
            saved_class = result.last_perturbed_label
            saved_confidence = result.last_perturbed_confidence
            saved_image_type = "last_perturbation"
            saved_l0 = count_l0(image, saved_image)

        class_output_dir = adversarial_dir / CLASS_NAMES[label]
        class_output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{index:04d}_{path.stem}.png"
        output_path = class_output_dir / output_name
        save_image(output_path, saved_image)

        row = {
            "file_path": str(path),
            "actual_class": CLASS_NAMES[label],
            "initial_class": CLASS_NAMES[result.original_label],
            "adversarial_class": CLASS_NAMES[result.adversarial_label],
            "initial_confidence": result.original_confidence,
            "adversarial_confidence": result.adversarial_confidence,
            "success": result.success,
            "status": "attacked",
            "queries": result.queries,
            "l0": result.l0,
            "saved_image_path": str(output_path),
            "saved_image_type": saved_image_type,
            "saved_class": CLASS_NAMES[saved_class],
            "saved_confidence": saved_confidence,
            "saved_l0": saved_l0,
        }
        rows.append(row)

        print(
            f"[{index + 1}] {path.name}: "
            f"{CLASS_NAMES[label]} -> {CLASS_NAMES[result.adversarial_label]} | "
            f"success={result.success} queries={result.queries} L0={result.l0}"
        )

    csv_path = args.results_dir / "pixle_results.csv"
    fieldnames = [
        "file_path",
        "actual_class",
        "initial_class",
        "adversarial_class",
        "initial_confidence",
        "adversarial_confidence",
        "success",
        "status",
        "queries",
        "l0",
        "saved_image_path",
        "saved_image_type",
        "saved_class",
        "saved_confidence",
        "saved_l0",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    attempted_rows = [row for row in rows if row["status"] == "attacked"]
    metrics = {
        "total_images_seen": len(rows),
        "initially_correct": initially_correct,
        "attacked_images": attempted,
        "successful_attacks": successes,
        "attack_success_rate": successes / attempted if attempted else 0.0,
        "mean_queries": (
            float(np.mean([row["queries"] for row in attempted_rows]))
            if attempted_rows
            else 0.0
        ),
        "mean_l0": (
            float(np.mean([row["l0"] for row in attempted_rows]))
            if attempted_rows
            else 0.0
        ),
        "config": {
            "restarts": args.restarts,
            "iterations": args.iterations,
            "patch_size": args.patch_size,
            "mapping": args.mapping,
            "algorithm": args.algorithm,
            "operation": args.operation,
            "seed": args.seed,
        },
    }
    (args.results_dir / "pixle_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
