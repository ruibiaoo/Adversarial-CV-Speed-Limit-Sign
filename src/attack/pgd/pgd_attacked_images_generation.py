import argparse
import csv
from pathlib import Path
import random
import shutil

import numpy as np
import tensorflow as tf

from pgd_tuning import (
    generate_best_restart,
    generate_sweep_parameters,
    load_art_classifier,
    load_config,
    pgd_attack,
    resolve_repo_path,
    validate_attack_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PGD adversarial examples for dataset splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "pgd.yaml",
        help="Path to PGD YAML config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to Keras model checkpoint (overrides config if given).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Dataset splits to attack, e.g. --splits train val test.",
    )
    parser.add_argument(
        "--skip-misclassified",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip images already misclassified by the model. Default is False (generate all).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory for attacked images (default: from config or data/data_updated_pgd).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to attack per split (for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for PGD attack.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs in output directory.",
    )
    return parser.parse_args()


def validate_mapping(config: dict, source_root: Path, splits: list[str]) -> None:
    names = config["data"]["class_names"]
    expected = [f"{speed}_MPH" for speed in range(10, 71, 10)]
    if names != expected:
        raise ValueError(f"Invalid class order: expected {expected}, got {names}")
    metadata = load_config(source_root / "data.yaml")
    mapped = metadata.get("names")
    if isinstance(mapped, dict):
        try:
            mapped = [mapped[index] if index in mapped else mapped[str(index)] for index in range(len(names))]
        except KeyError as error:
            raise ValueError("data.yaml has missing class indices") from error
    if metadata.get("nc") != len(names) or mapped != names:
        raise ValueError("Processed data.yaml does not match the PGD class mapping")
    for split in splits:
        split_dir = source_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory missing: {split_dir}")
        for name in names:
            if not (split_dir / name).is_dir():
                raise FileNotFoundError(f"Class directory missing: {split_dir / name}")


def selected_parameters(selected_path: Path, sweep_config: dict) -> dict:
    selected = load_config(selected_path)
    required = {"iterations", "epsilon_255", "alpha_ratio", "restarts"}
    if set(selected) != required:
        raise ValueError(f"{selected_path} must contain exactly {sorted(required)}")
    try:
        parameters = {
            "iterations": int(selected["iterations"]),
            "epsilon_255": float(selected["epsilon_255"]),
            "alpha_ratio": float(selected["alpha_ratio"]),
            "restarts": int(selected["restarts"]),
        }
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid PGD values in {selected_path}") from error
    if (parameters["iterations"] != selected["iterations"]
            or parameters["restarts"] != selected["restarts"]):
        raise ValueError("Selected iterations and restarts must be integers")
    choices = list(generate_sweep_parameters(sweep_config))
    if parameters not in choices:
        raise ValueError(f"Selected PGD configuration is not one of the validation sweep choices: {parameters}")
    return parameters


def list_split_images(source_root: Path, split: str, class_names: list[str], max_images: int | None = None):
    split_root = source_root / split
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    selected = []
    seen = set()

    for class_index, class_name in enumerate(class_names):
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in valid_suffixes:
                continue
            output_filename = image_path.stem + ".png"
            output_key = (class_name, output_filename)
            if output_key in seen:
                raise ValueError(f"Duplicate output filename in split {split}: {output_key}")
            seen.add(output_key)
            selected.append((image_path, class_index, (class_name, output_filename)))
            if max_images is not None and len(selected) >= max_images:
                return selected

    if not selected:
        raise ValueError(f"No images found for split {split} in {split_root}")
    return selected


def decode_model_input(image_bytes: bytes | tf.Tensor, image_size: int) -> np.ndarray:
    """Decode bytes exactly as test images are decoded for Phase 2 inference."""
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_rgb_image(path: Path, image_size: int) -> np.ndarray:
    try:
        return decode_model_input(tf.io.read_file(str(path)), image_size)
    except (tf.errors.OpError, ValueError) as error:
        raise ValueError(f"Could not decode image {path}: {error}") from error


def encode_bounded_png(
    path: Path, original: np.ndarray, attacked: np.ndarray, epsilon: float,
    image_size: int,
) -> tuple[bytes, float]:
    original_64 = original.astype(np.float64)
    lower = np.ceil(np.maximum(0.0, original_64 - epsilon) * 255.0)
    upper = np.floor(np.minimum(1.0, original_64 + epsilon) * 255.0)
    if np.any(lower > upper):
        raise RuntimeError(f"No uint8 pixel value is within epsilon for {path}")
    pixels = np.clip(np.rint(attacked * 255.0), lower, upper).astype(np.uint8)
    for _ in range(3):
        png_bytes = tf.io.encode_png(tf.convert_to_tensor(pixels)).numpy()
        decoded = decode_model_input(png_bytes, image_size)
        difference = decoded.astype(np.float64) - original_64
        observed_maximum = float(np.max(np.abs(difference)))
        if observed_maximum <= epsilon:
            return png_bytes, observed_maximum

        too_high = difference > epsilon
        too_low = difference < -epsilon
        corrected = pixels.astype(np.int16)
        corrected[too_high] -= 1
        corrected[too_low] += 1
        if np.any((corrected < 0) | (corrected > 255)):
            break
        pixels = corrected.astype(np.uint8)

    raise RuntimeError(
        f"Saved PNG would exceed epsilon={epsilon:.8f} for {path}: "
        f"L-inf={observed_maximum:.8f}. No attacked images were saved."
    )


def main() -> None:
    args = parse_args()
    phase_directory = Path(__file__).resolve().parent
    repo_directory = phase_directory.parents[2]  # src/attack/pgd -> repository root
    config = load_config(args.config)
    validate_attack_settings(config)
    parameters = selected_parameters(phase_directory / "configs" / "pgd_selected.yaml", config)

    generation_config = config.get("generation", {})

    # Determine splits
    if args.splits:
        splits = args.splits
    elif "splits" in generation_config:
        splits = generation_config["splits"]
    else:
        splits = ["train", "val", "test"]

    # Determine skip_misclassified switch
    if args.skip_misclassified is not None:
        skip_misclassified = args.skip_misclassified
    elif "skip_misclassified" in generation_config:
        skip_misclassified = bool(generation_config["skip_misclassified"])
    else:
        skip_misclassified = False

    # Determine output root
    if args.output_root:
        output_root = resolve_repo_path(args.output_root, repo_directory)
    elif "output_root" in generation_config:
        output_root = resolve_repo_path(generation_config["output_root"], repo_directory)
    else:
        output_root = resolve_repo_path("data/data_updated_pgd", repo_directory)

    # Determine model override
    if args.model:
        config["model"]["path"] = args.model

    source_root = resolve_repo_path(config["data"]["root"], repo_directory)
    validate_mapping(config, source_root, splits)

    class_names = config["data"]["class_names"]
    image_size = int(config["model"]["image_size"])
    batch_size = args.batch_size or int(config["data"]["batch_size"])
    epsilon = parameters["epsilon_255"] / 255.0

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"Loading model for PGD from {config['model']['path']}...")
    classifier = load_art_classifier(config, repo_directory)
    attack = pgd_attack(classifier, config, parameters)

    output_root.mkdir(parents=True, exist_ok=True)
    results_dir = resolve_repo_path(config["experiment"]["output_dir"], repo_directory) / "generation"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPGD Generation Configuration:")
    print(f"  Model: {config['model']['path']}")
    print(f"  Source Root: {source_root}")
    print(f"  Output Root: {output_root}")
    print(f"  Splits: {splits}")
    print(f"  Skip misclassified switch: {skip_misclassified} (generate all: {not skip_misclassified})")
    print(f"  Batch size: {batch_size}")
    print(f"  Parameters: {parameters}\n")

    for split in splits:
        print(f"\n================ Processing split: {split} ================")
        split_output_root = output_root / split
        if split_output_root.is_dir() and any(path.is_file() for path in split_output_root.rglob("*")):
            if not args.overwrite:
                raise FileExistsError(
                    f"Attacked image directory already contains files: {split_output_root}. "
                    "Use --overwrite or remove existing files before proceeding."
                )

        selected_images = list_split_images(source_root, split, class_names, args.max_images)
        print(f"Found {len(selected_images)} candidate images in split '{split}'.")

        manifest_rows = []
        encoded_images = []
        maximum_saved_linf = 0.0

        clean_correct_total = 0
        successful_attacks_total = 0
        total_processed = 0

        for start in range(0, len(selected_images), batch_size):
            batch = selected_images[start:start + batch_size]
            images = np.stack([load_rgb_image(path, image_size) for path, _, _ in batch])
            labels = np.asarray([label for _, label, _ in batch], dtype=np.int64)

            # Predict on clean images
            clean_preds_proba = classifier.predict(images)
            clean_preds = np.argmax(clean_preds_proba, axis=1)
            is_correct_mask = (clean_preds == labels)

            # Check filtering switch
            if skip_misclassified:
                eligible_indices = np.where(is_correct_mask)[0]
                if len(eligible_indices) == 0:
                    for (source, label_idx, (name, filename)), clean_p in zip(batch, clean_preds):
                        manifest_rows.append({
                            "file_path": str(source),
                            "output_path": "",
                            "actual_class": class_names[label_idx],
                            "clean_predicted": class_names[clean_p],
                            "adv_predicted": class_names[clean_p],
                            "clean_correct": False,
                            "attack_success": False,
                            "saved_linf": 0.0,
                            "status": "skipped_misclassified",
                        })
                    continue
                active_batch = [batch[i] for i in eligible_indices]
                active_images = images[eligible_indices]
                active_labels = labels[eligible_indices]
                active_clean_preds = clean_preds[eligible_indices]
            else:
                active_batch = batch
                active_images = images
                active_labels = labels
                active_clean_preds = clean_preds

            attacked, adv_preds_proba, _ = generate_best_restart(
                attack, classifier, active_images, active_labels, parameters, config
            )
            adv_preds = np.argmax(adv_preds_proba, axis=1)

            for item, clean, adversarial, clean_pred, adv_pred, label_val in zip(
                active_batch, active_images, attacked, active_clean_preds, adv_preds, active_labels
            ):
                source, label_idx, (name, filename) = item
                maximum_change = float(np.max(np.abs(adversarial - clean)))
                if maximum_change > epsilon + 1e-5 or not np.isfinite(adversarial).all():
                    raise RuntimeError(f"Invalid PGD output for {source}: L∞={maximum_change}")
                
                destination = split_output_root / name / filename
                image_bytes, saved_linf = encode_bounded_png(
                    destination, clean, adversarial, epsilon, image_size
                )
                encoded_images.append((destination, image_bytes))
                maximum_saved_linf = max(maximum_saved_linf, saved_linf)

                clean_correct = bool(clean_pred == label_val)
                attack_success = bool(adv_pred != label_val)

                if clean_correct:
                    clean_correct_total += 1
                if attack_success:
                    successful_attacks_total += 1
                total_processed += 1

                manifest_rows.append({
                    "file_path": str(source),
                    "output_path": str(destination),
                    "actual_class": class_names[label_idx],
                    "clean_predicted": class_names[clean_pred],
                    "adv_predicted": class_names[adv_pred],
                    "clean_correct": clean_correct,
                    "attack_success": attack_success,
                    "saved_linf": saved_linf,
                    "status": "attacked",
                })

            print(f"Processed {min(start + batch_size, len(selected_images))}/{len(selected_images)} images in split '{split}'")

        # Create output directories and write files
        for name in class_names:
            (split_output_root / name).mkdir(parents=True, exist_ok=True)
        for destination, image_bytes in encoded_images:
            destination.write_bytes(image_bytes)

        # Write manifest CSV
        manifest_path = results_dir / f"generation_{split}_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "file_path", "output_path", "actual_class", "clean_predicted",
                "adv_predicted", "clean_correct", "attack_success", "saved_linf", "status"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        print(f"\n--- Split '{split}' Complete ---")
        print(f"Saved {len(encoded_images)} images to {split_output_root}")
        print(f"Max saved L-inf: {maximum_saved_linf:.8f} (epsilon: {epsilon:.8f})")
        if total_processed > 0:
            clean_acc = clean_correct_total / total_processed
            asr = successful_attacks_total / total_processed
            print(f"Clean Accuracy: {clean_acc:.2%} ({clean_correct_total}/{total_processed})")
            print(f"Attack Success Rate (adv != actual): {asr:.2%} ({successful_attacks_total}/{total_processed})")
        print(f"Manifest written to: {manifest_path}")

    # Copy data.yaml to output_root
    source_yaml = source_root / "data.yaml"
    if source_yaml.is_file():
        shutil.copy2(source_yaml, output_root / "data.yaml")
        print(f"\nCopied {source_yaml} -> {output_root / 'data.yaml'}")

    print("\nAll requested splits successfully generated with PGD!")


if __name__ == "__main__":
    main()
