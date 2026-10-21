"""Evaluate Pixle on the speed-limit-sign classifier."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil
import threading

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

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        gpus = tf.config.list_physical_devices("GPU")
        print(f"Detected GPUs: {gpus if gpus else 'None (falling back to CPU)'}")

        self.tf = tf
        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        self.image_size = image_size
        self.preprocess = preprocess

        if preprocess == "resnet":
            @tf.function(input_signature=[tf.TensorSpec(shape=(1, image_size, image_size, 3), dtype=tf.float32)])
            def _predict_fn(batch):
                proc = tf.keras.applications.resnet.preprocess_input(batch)
                return self.model(proc, training=False)
        elif preprocess == "none":
            @tf.function(input_signature=[tf.TensorSpec(shape=(1, image_size, image_size, 3), dtype=tf.float32)])
            def _predict_fn(batch):
                return self.model(batch, training=False)
        else:
            raise ValueError(f"Unknown preprocessing mode: {preprocess}")

        self._predict_fn = _predict_fn

        # Warm up execution
        dummy = tf.zeros((1, image_size, image_size, 3), dtype=tf.float32)
        _ = self._predict_fn(dummy)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        batch = self.tf.convert_to_tensor(image[None], dtype=self.tf.float32)
        probabilities = self._predict_fn(batch).numpy()
        return probabilities[0]


def load_image(path: Path, image_size: int) -> np.ndarray:
    """Load an image using the same path as Phase 1's test pipeline."""
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


def iter_split_images(data_dir: Path, split: str, class_names: list[str]):
    split_dir = data_dir / split
    for label, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
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


def iter_test_images(data_dir: Path, class_names: list[str]):
    return iter_split_images(data_dir, "test", class_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("src/modelling/runs/best_baseline.keras"),
        help="Path to Keras model checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/data_updated"),
        help="Root directory of dataset containing train/val/test splits.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process, e.g. --splits train val test",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--preprocess", choices=["resnet", "none"], default="resnet")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("src/attack/pixle/results"),
        help="Directory to save metrics and results CSV.",
    )
    parser.add_argument(
        "--adversarial-dir",
        type=Path,
        default=Path("data/data_updated_pixle"),
        help="Root directory for class-organized adversarial examples.",
    )
    parser.add_argument(
        "--skip-misclassified",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip images already misclassified by the model. Default is False (generate all).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads for parallel Pixle attacks (default: 8).",
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


def attack_worker(
    index: int,
    path: Path,
    label: int,
    args: argparse.Namespace,
    oracle: KerasProbabilityOracle,
    split_adversarial_dir: Path,
    print_lock: threading.Lock,
    progress_state: dict,
    total_images: int,
) -> dict:
    image = load_image(path, args.image_size)
    initial_probabilities = oracle(image)
    initial_label = int(np.argmax(initial_probabilities))
    initial_confidence = float(initial_probabilities[initial_label])

    class_output_dir = split_adversarial_dir / CLASS_NAMES[label]
    output_name = f"{index:04d}_{path.stem}.png"
    output_path = class_output_dir / output_name

    # If prediction is already wrong
    if initial_label != label:
        if args.skip_misclassified:
            with print_lock:
                progress_state["completed"] += 1
                count = progress_state["completed"]
                print(
                    f"[{count}/{total_images}] {path.name}: "
                    f"{CLASS_NAMES[label]} (already predicted as {CLASS_NAMES[initial_label]}) -> SKIPPED"
                )
            return {
                "index": index,
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
                "is_attacked": False,
            }
        else:
            # Generate all: save the clean image as it already fools the model
            save_image(output_path, image)
            with print_lock:
                progress_state["completed"] += 1
                count = progress_state["completed"]
                print(
                    f"[{count}/{total_images}] {path.name}: "
                    f"{CLASS_NAMES[label]} (initially wrong -> {CLASS_NAMES[initial_label]}) | saved clean"
                )
            return {
                "index": index,
                "file_path": str(path),
                "actual_class": CLASS_NAMES[label],
                "initial_class": CLASS_NAMES[initial_label],
                "adversarial_class": CLASS_NAMES[initial_label],
                "initial_confidence": initial_confidence,
                "adversarial_confidence": initial_confidence,
                "success": True,
                "status": "initially_misclassified",
                "queries": 1,
                "l0": 0,
                "saved_image_path": str(output_path),
                "saved_image_type": "clean_already_misclassified",
                "saved_class": CLASS_NAMES[initial_label],
                "saved_confidence": initial_confidence,
                "saved_l0": 0,
                "is_attacked": True,
            }

    # Derive unique seed for thread-safety and reproducibility
    image_seed = (args.seed + index * 10007) % (2**31 - 1)
    config = PixleConfig(
        restarts=args.restarts,
        iterations=args.iterations,
        patch_size=args.patch_size,
        mapping=args.mapping,
        algorithm=args.algorithm,
        operation=args.operation,
        seed=image_seed,
    )
    attacker = PixleAttack(oracle, config)

    result = attack_with_original_l0(
        attacker,
        image,
        label,
        initial_probabilities=initial_probabilities,
    )

    if result.success:
        saved_image = result.adversarial_image
        saved_class = result.adversarial_label
        saved_confidence = result.adversarial_confidence
        saved_image_type = "successful_adversarial"
        saved_l0 = result.l0
    else:
        saved_image = result.last_perturbed_image
        saved_class = result.last_perturbed_label
        saved_confidence = result.last_perturbed_confidence
        saved_image_type = "last_perturbation"
        saved_l0 = count_l0(image, saved_image)

    save_image(output_path, saved_image)

    with print_lock:
        progress_state["completed"] += 1
        count = progress_state["completed"]
        print(
            f"[{count}/{total_images}] {path.name}: "
            f"{CLASS_NAMES[label]} -> {CLASS_NAMES[result.adversarial_label]} | "
            f"success={result.success} queries={result.queries} L0={result.l0}"
        )

    return {
        "index": index,
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
        "is_attacked": True,
    }


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.adversarial_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print("Pixle Adversarial Generation Setup")
    print(f"  Model: {args.model}")
    print(f"  Data source: {args.data}")
    print(f"  Adversarial output: {args.adversarial_dir}")
    print(f"  Splits: {args.splits}")
    print(f"  Skip misclassified switch: {args.skip_misclassified} (generate all: {not args.skip_misclassified})")
    print(f"  Workers (threads): {args.workers}")
    print(f"  Restarts: {args.restarts}, Iterations: {args.iterations}, Patch size: {args.patch_size}")
    print("==========================================================\n")

    print("Loading model and warming up KerasProbabilityOracle...")
    oracle = KerasProbabilityOracle(args.model, args.image_size, args.preprocess)

    for split in args.splits:
        print(f"\n================ Processing split: {split} ================")
        split_adversarial_dir = args.adversarial_dir / split
        split_results_dir = args.results_dir / split
        split_adversarial_dir.mkdir(parents=True, exist_ok=True)
        split_results_dir.mkdir(parents=True, exist_ok=True)

        for name in CLASS_NAMES:
            (split_adversarial_dir / name).mkdir(parents=True, exist_ok=True)

        candidate_items = list(iter_split_images(args.data, split, CLASS_NAMES))
        if args.max_images is not None:
            candidate_items = candidate_items[: args.max_images]

        total_images = len(candidate_items)
        print(f"Found {total_images} candidate images in split '{split}'.")
        if total_images == 0:
            print(f"Warning: No images found for split '{split}'. Skipping.")
            continue

        print_lock = threading.Lock()
        progress_state = {"completed": 0}

        rows = []
        if args.workers > 1:
            print(f"Launching ThreadPoolExecutor with {args.workers} workers...")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        attack_worker,
                        index,
                        path,
                        label,
                        args,
                        oracle,
                        split_adversarial_dir,
                        print_lock,
                        progress_state,
                        total_images,
                    )
                    for index, (path, label) in enumerate(candidate_items)
                ]
                for future in as_completed(futures):
                    rows.append(future.result())
        else:
            for index, (path, label) in enumerate(candidate_items):
                row = attack_worker(
                    index,
                    path,
                    label,
                    args,
                    oracle,
                    split_adversarial_dir,
                    print_lock,
                    progress_state,
                    total_images,
                )
                rows.append(row)

        # Sort back into original order
        rows.sort(key=lambda r: r["index"])
        for r in rows:
            r.pop("index", None)
            is_attacked = r.pop("is_attacked", True)

        csv_path = split_results_dir / "pixle_results.csv"
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

        attacked_rows = [row for row in rows if row["status"] in {"attacked", "initially_misclassified"}]
        fully_attacked = [row for row in rows if row["status"] == "attacked"]
        successful_attacks = sum(1 for row in rows if row["success"])

        metrics = {
            "split": split,
            "total_images_seen": len(rows),
            "attacked_or_generated_images": len(attacked_rows),
            "successful_attacks": successful_attacks,
            "attack_success_rate": successful_attacks / len(attacked_rows) if attacked_rows else 0.0,
            "mean_queries": (
                float(np.mean([row["queries"] for row in fully_attacked]))
                if fully_attacked
                else 0.0
            ),
            "mean_l0": (
                float(np.mean([row["l0"] for row in fully_attacked]))
                if fully_attacked
                else 0.0
            ),
            "config": {
                "workers": args.workers,
                "restarts": args.restarts,
                "iterations": args.iterations,
                "patch_size": args.patch_size,
                "mapping": args.mapping,
                "algorithm": args.algorithm,
                "operation": args.operation,
                "seed": args.seed,
                "skip_misclassified": args.skip_misclassified,
            },
        }
        metrics_path = split_results_dir / "pixle_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"\nResults for split '{split}' saved to {csv_path}")
        print(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")

    # Copy data.yaml to output directory
    source_yaml = args.data / "data.yaml"
    if source_yaml.is_file():
        shutil.copy2(source_yaml, args.adversarial_dir / "data.yaml")
        print(f"\nCopied {source_yaml} -> {args.adversarial_dir / 'data.yaml'}")

    print("\nAll requested splits successfully generated with Pixle!")


if __name__ == "__main__":
    main()
