import csv
from pathlib import Path
import random

import numpy as np
from PIL import Image
import tensorflow as tf
import yaml

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


def validate_mapping(config: dict, source_root: Path) -> dict:
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
    return metadata


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
    candidates = (
        repo_directory / "src" / "Phase_1" / "results" / "predictions.csv",
        repo_directory / "src" / "Phase_1" / "results" / "test" / "predictions.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Phase 1 predictions.csv not found at {candidates}")


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


def load_rgb_image(path: Path, image_size: int) -> np.ndarray:
    try:
        data = tf.io.read_file(str(path))
        image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size), method="bilinear")
        return np.asarray(image, dtype=np.float32) / 255.0
    except (tf.errors.OpError, ValueError) as error:
        raise ValueError(f"Could not decode test image {path}: {error}") from error


def save_bounded_png(path: Path, original: np.ndarray, attacked: np.ndarray, epsilon: float) -> None:
    # Quantization must not put the saved 8-bit image outside the epsilon ball.
    lower = np.ceil(np.maximum(0.0, original - epsilon) * 255.0)
    upper = np.floor(np.minimum(1.0, original + epsilon) * 255.0)
    pixels = np.clip(np.rint(attacked * 255.0), lower, upper).astype(np.uint8)
    if np.max(np.abs(pixels.astype(np.float32) / 255.0 - original)) > epsilon + 1e-6:
        raise RuntimeError(f"PNG quantization exceeded epsilon for {path}")
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def main() -> None:
    phase_directory = Path(__file__).resolve().parent
    repo_directory = phase_directory.parents[1]
    config = load_config(phase_directory / "configs" / "pgd.yaml")
    validate_attack_settings(config)
    parameters = selected_parameters(phase_directory / "configs" / "pgd_selected.yaml", config)
    source_root = resolve_repo_path(config["data"]["root"], repo_directory)
    metadata = validate_mapping(config, source_root)
    csv_path = predictions_file(repo_directory)
    class_names = config["data"]["class_names"]
    selected = eligible_test_images(csv_path, source_root / "test", class_names)
    output_root = repo_directory / "data" / "data_pgd_attacked"
    output_test = output_root / "test"
    if output_test.is_dir() and any(path.is_file() for path in output_test.rglob("*")):
        raise FileExistsError(f"Attacked test directory already contains files: {output_test}")
    for _, _, (name, filename) in selected:
        if (output_root / "test" / name / filename).exists():
            raise FileExistsError(f"Output image already exists: {output_root / 'test' / name / filename}")

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
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

    for name in class_names:
        (output_root / "test" / name).mkdir(parents=True, exist_ok=True)
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        images = np.stack([load_rgb_image(path, image_size) for path, _, _ in batch])
        labels = np.asarray([label for _, label, _ in batch], dtype=np.int64)
        attacked, _, _ = generate_best_restart(attack, classifier, images, labels, parameters, config)
        for (source, _, (name, filename)), clean, adversarial in zip(batch, images, attacked):
            maximum_change = float(np.max(np.abs(adversarial - clean)))
            if maximum_change > epsilon + 1e-5 or not np.isfinite(adversarial).all():
                raise RuntimeError(f"Invalid PGD output for {source}: L∞={maximum_change}")
            save_bounded_png(output_root / "test" / name / filename, clean, adversarial, epsilon)
        print(f"Saved {min(start + batch_size, len(selected))}/{len(selected)} attacked test images")

    attacked_metadata = dict(metadata)
    attacked_metadata.update({
        "path": ".",
        "train": "../data_processed/train",
        "val": "../data_processed/val",
        "test": "test",
    })
    with (output_root / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(attacked_metadata, file, sort_keys=False)
    print(f"Saved {len(selected)} attacked test images and {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
