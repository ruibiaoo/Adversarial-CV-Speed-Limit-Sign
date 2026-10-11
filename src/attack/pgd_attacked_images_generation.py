import csv
from pathlib import Path
import random

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


def parse_correct(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"Invalid value in predictions.csv correct column: {value!r}")


def validate_mapping(config: dict, source_root: Path) -> None:
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
    for name in names:
        if not (source_root / "test" / name).is_dir():
            raise FileNotFoundError(f"Test class directory missing: {source_root / 'test' / name}")


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


def predictions_file(repo_directory: Path) -> Path:
    csv_path = repo_directory / "src" / "modelling" / "results" / "test_new" / "predictions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")
    return csv_path


def eligible_test_images(csv_path: Path, test_root: Path, class_names: list[str]):
    selected = []
    seen = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"file_path", "actual_class", "correct"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, row in enumerate(reader, start=2):
            if not parse_correct(row["correct"]):
                continue
            name = row["actual_class"].strip()
            if name not in class_names:
                raise ValueError(f"Row {row_number}: unknown actual_class {name!r}")
            parts = Path(row["file_path"].replace("\\", "/")).parts
            if len(parts) < 3 or parts[-3] != "test" or parts[-2] != name:
                raise ValueError(f"Row {row_number}: file_path does not match test/{name}: {row['file_path']}")
            image_path = test_root / name / parts[-1]
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise FileNotFoundError(f"Row {row_number}: missing or empty test image: {image_path}")
            output_key = (name, image_path.stem + ".png")
            if output_key in seen:
                raise ValueError(f"Row {row_number}: duplicate output filename {output_key}")
            seen.add(output_key)
            selected.append((image_path, class_names.index(name), output_key))
    if not selected:
        raise ValueError("No correctly classified test images found in predictions.csv")
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
        raise ValueError(f"Could not decode test image {path}: {error}") from error


def encode_bounded_png(
    path: Path, original: np.ndarray, attacked: np.ndarray, epsilon: float,
    image_size: int,
) -> tuple[bytes, float]:
    # Quantize within the PGD budget, then verify the encoded image through
    # the exact decoder and resize path used for model inference.
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

        # Float32 normalization can put a boundary pixel a few nanounits
        # beyond epsilon. Move only the offending channels one uint8 level
        # toward the clean image, then check the encoded PNG again.
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
    phase_directory = Path(__file__).resolve().parent
    repo_directory = phase_directory.parents[1]
    config = load_config(phase_directory / "configs" / "pgd.yaml")
    validate_attack_settings(config)
    parameters = selected_parameters(phase_directory / "configs" / "pgd_selected.yaml", config)
    source_root = resolve_repo_path(config["data"]["root"], repo_directory)
    validate_mapping(config, source_root)
    csv_path = predictions_file(repo_directory)
    class_names = config["data"]["class_names"]
    selected = eligible_test_images(csv_path, source_root / "test", class_names)
    output_root = source_root / "pgd_attacked"
    if output_root.is_dir() and any(path.is_file() for path in output_root.rglob("*")):
        raise FileExistsError(
            f"Attacked image directory already contains files: {output_root}. "
            "Move the previous JPG outputs aside before generating verified PNGs."
        )
    for _, _, (name, filename) in selected:
        if (output_root / name / filename).exists():
            raise FileExistsError(f"Output image already exists: {output_root / name / filename}")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    classifier = load_art_classifier(config, repo_directory)
    attack = pgd_attack(classifier, config, parameters)
    image_size = int(config["model"]["image_size"])
    batch_size = int(config["data"]["batch_size"])
    epsilon = parameters["epsilon_255"] / 255.0

    # Reject stale Phase 1 CSVs before writing any attacked image.
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        images = np.stack([load_rgb_image(path, image_size) for path, _, _ in batch])
        labels = np.asarray([label for _, label, _ in batch], dtype=np.int64)
        if not np.array_equal(np.argmax(classifier.predict(images), axis=1), labels):
            raise ValueError("predictions.csv disagrees with the current checkpoint on a selected test image")

    encoded_images = []
    maximum_saved_linf = 0.0
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        images = np.stack([load_rgb_image(path, image_size) for path, _, _ in batch])
        labels = np.asarray([label for _, label, _ in batch], dtype=np.int64)
        attacked, _, _ = generate_best_restart(attack, classifier, images, labels, parameters, config)
        for (source, _, (name, filename)), clean, adversarial in zip(batch, images, attacked):
            maximum_change = float(np.max(np.abs(adversarial - clean)))
            if maximum_change > epsilon + 1e-5 or not np.isfinite(adversarial).all():
                raise RuntimeError(f"Invalid PGD output for {source}: L∞={maximum_change}")
            destination = output_root / name / filename
            image_bytes, saved_linf = encode_bounded_png(
                destination, clean, adversarial, epsilon, image_size
            )
            encoded_images.append((destination, image_bytes))
            maximum_saved_linf = max(maximum_saved_linf, saved_linf)
        print(f"Prepared {min(start + batch_size, len(selected))}/{len(selected)} attacked test images")

    for name in class_names:
        (output_root / name).mkdir(parents=True, exist_ok=True)
    for destination, image_bytes in encoded_images:
        destination.write_bytes(image_bytes)
    print(f"Saved {len(selected)} attacked test images to {output_root}")
    print(
        f"TensorFlow-decoded PNG L-inf maximum: {maximum_saved_linf:.8f} "
        f"(epsilon={epsilon:.8f})"
    )


if __name__ == "__main__":
    main()
