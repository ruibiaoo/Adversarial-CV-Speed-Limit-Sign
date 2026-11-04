"""Compare Grad-CAM and Ablation-CAM, or explain one evaluated pipeline.

This is a post-hoc analysis, not a training or attack script. It uses the
``pre_none__model_*__post_none`` predictions written by ``test_all.py`` and
explains the ground-truth class on a few matched images. Both methods use the
same checkpoint, ResNet feature layer and pre-softmax class score per image.

Inside the modelling Docker container, for example::

    python -u ablationcam.py
    python -u ablationcam.py --datasets pgd_test pixle_test pgdpixle_test
    python -u ablationcam.py --model /app/runs/best_baseline.keras \
        --predictions /app/final_final_results_i_promise/tests/pre_none__model_baseline__post_none/pgd_test/predictions.csv \
        --image 20s1_png.rf.21e5ce2e19a562face23463639c38966.png --target true

The saved checkpoint must reproduce its predictions.csv probabilities before
any explanation is generated. See Desai and Ramaswamy, WACV 2020, Ablation-CAM.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import yaml

from gradcam import LAST_CONV_LAYER, _locate_backbone_and_stages, load_defence_pipeline
from test import load_classifier


DEFAULT_MODELS = ("baseline", "adversarial_retrained", "grl", "convae_rn50")
DEFAULT_CASES = ("60_MPH/60_42", "40_MPH/mph40_049")


def parse_args() -> argparse.Namespace:
    modelling_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare true-class Ablation-CAM maps on matched test images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=modelling_dir / "configs" / "test_all.yaml",
        help="test_all.yaml with model paths, datasets, class order and results path.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Override the test_all.py output directory containing tests/<combination>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=modelling_dir / "results" / "ablation_cam",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--datasets", nargs="+", default=["pgd_test"])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help="Original image stems, optionally prefixed by class (e.g. 60_MPH/60_42).",
    )
    parser.add_argument("--layer", default=LAST_CONV_LAYER)
    parser.add_argument("--ablation-batch-size", type=int, default=32)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--model", type=Path, help="Checkpoint for single-image mode.")
    parser.add_argument("--predictions", type=Path, help="Matching test_all.py predictions.csv.")
    parser.add_argument("--image", help="Exact image filename or stem in predictions.csv.")
    parser.add_argument("--image-size", type=int, help="Defaults to test_all.yaml image_size.")
    parser.add_argument(
        "--target", choices=("true", "predicted"), default="true",
        help="Class to explain in single-image mode (default: true class).",
    )
    parser.add_argument(
        "--preprocessing", choices=("none", "convae_jpeg"), default="none",
        help="Apply the ConvAE then JPEG chain from test_all.yaml when requested.",
    )
    parser.add_argument(
        "--postprocessing", choices=("none", "highconf"), default="none",
        help="Verify High-Confidence Filtering against the CSV; CAM explains the classifier.",
    )
    return parser.parse_args()


def resolve_project_path(raw_path: str | Path, modelling_dir: Path) -> Path:
    """Resolve Docker /app paths against this checkout when run outside Docker."""
    path = Path(raw_path)
    if path.exists():
        return path
    normalized = str(raw_path).replace("\\", "/")
    repo_dir = modelling_dir.parents[1]
    aliases = {
        "/app/eval_data": repo_dir / "data" / "data_updated",
        "/app/data_pgdpixle": repo_dir / "data" / "data_pgdpixle",
        "/app/data_updated_pgd": repo_dir / "data" / "data_updated_pgd",
        "/app/data_updated_pixle": repo_dir / "data" / "data_updated_pixle",
        "/app/data": repo_dir / "data" / "data_combined",
        "/app": modelling_dir,
    }
    for prefix, local_root in aliases.items():
        if normalized == prefix or normalized.startswith(prefix + "/"):
            remainder = normalized[len(prefix) :].lstrip("/")
            return local_root / remainder
    if path.is_absolute():
        return path
    return modelling_dir / path


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    for key in ("models", "datasets", "class_names", "image_size", "output_dir"):
        if key not in config:
            raise ValueError(f"Missing {key!r} in {path}")
    class_names = config["class_names"]
    if not isinstance(class_names, list) or not all(
        isinstance(name, str) and name.strip() for name in class_names
    ):
        raise ValueError("class_names must contain nonempty class-name strings")
    if len(class_names) != len(set(class_names)):
        raise ValueError("class_names must be unique")
    return config


def original_stem(filename: str) -> str:
    """Drop the four-digit attack-image prefix, preserving the original stem."""
    return re.sub(r"^\d{4}_", "", Path(filename).stem)


def read_predictions(path: Path, class_names: list[str]) -> dict[tuple[str, str], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    expected = {"file_path", "actual_class", "predicted_class", "correct"}
    expected.update(f"probability_{name}" for name in class_names)
    rows: dict[tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns {sorted(missing)} in {path}")
        for row in reader:
            actual = row["actual_class"]
            predicted = row["predicted_class"]
            if actual not in class_names or predicted not in class_names:
                raise ValueError(f"Invalid class mapping in {path}: {actual}, {predicted}")
            key = (actual, original_stem(Path(row["file_path"]).name))
            if key in rows:
                raise ValueError(f"Ambiguous original image {key} in {path}")
            if row["correct"].strip().lower() not in {"true", "false"}:
                raise ValueError(f"Invalid correct flag for {key} in {path}")
            if (actual == predicted) != (row["correct"].strip().lower() == "true"):
                raise ValueError(f"Inconsistent correct flag for {key} in {path}")
            rows[key] = row
    return rows


def select_case(selector: str, available: dict[tuple[str, str], dict]) -> tuple[str, str]:
    normalized = selector.replace("\\", "/")
    if "/" in normalized:
        class_name, stem = normalized.split("/", 1)
        key = (class_name, stem)
        if key not in available:
            raise ValueError(f"Case {selector!r} is not present in predictions CSV")
        return key
    matches = [key for key in available if key[1] == normalized]
    if len(matches) != 1:
        raise ValueError(
            f"Case {selector!r} matched {len(matches)} images; use CLASS/stem."
        )
    return matches[0]


def image_path_from_row(row: dict, dataset_dir: Path) -> Path:
    filename = Path(row["file_path"]).name
    path = dataset_dir / row["actual_class"] / filename
    if not path.is_file():
        raise FileNotFoundError(f"Image path from predictions CSV not found: {path}")
    return path


def read_image(path: Path, image_size: int) -> np.ndarray:
    image = tf.io.decode_image(
        tf.io.read_file(str(path)), channels=3, expand_animations=False
    )
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    return np.asarray(image.numpy(), dtype=np.float32)


def find_clean_image(
    clean_dir: Path, key: tuple[str, str], image_size: int
) -> np.ndarray:
    class_dir = clean_dir / key[0]
    matches = [p for p in class_dir.glob("*") if p.is_file() and p.stem == key[1]]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one clean original for {key} in {class_dir}, found {len(matches)}"
        )
    return read_image(matches[0], image_size)


def class_only_model(model: tf.keras.Model) -> tf.keras.Model:
    output = model.output
    if isinstance(output, dict):
        if "class_output" not in output:
            raise ValueError("Model has multiple outputs but no class_output")
        return tf.keras.Model(model.inputs, output["class_output"])
    if isinstance(output, (tuple, list)):
        if len(output) == 1:
            return tf.keras.Model(model.inputs, output[0])
        names = list(model.output_names)
        if "class_output" not in names:
            raise ValueError(f"Cannot identify class head among outputs {names}")
        return tf.keras.Model(model.inputs, output[names.index("class_output")])
    return model


class AblationModel:
    """Split a saved ResNet pipeline at one convolutional feature-map tensor."""

    def __init__(self, full_model: tf.keras.Model, layer_name: str, num_classes: int):
        self.full_model = full_model
        self.class_model = class_only_model(full_model)
        pre_layers, backbone, head_layers = _locate_backbone_and_stages(
            self.class_model, layer_name
        )
        if not head_layers or not isinstance(head_layers[-1], tf.keras.layers.Dense):
            raise ValueError(
                "Expected a Dense class head after the ResNet backbone; "
                f"found {[layer.name for layer in head_layers]}"
            )
        self.class_head = head_layers[-1]
        if self.class_head.units != num_classes:
            raise ValueError(
                f"Model has {self.class_head.units} classes; expected {num_classes}"
            )
        if getattr(self.class_head.activation, "__name__", "") != "softmax":
            raise ValueError("Expected a softmax Dense class head to recover logits")
        self.pre_layers = pre_layers
        self.head_prefix = head_layers[:-1]
        feature_tensor = backbone.get_layer(layer_name).output
        self.feature_model = tf.keras.Model(backbone.input, feature_tensor)
        self.tail_model = tf.keras.Model(feature_tensor, backbone.output)

    def features(self, batch: tf.Tensor) -> tf.Tensor:
        x = batch
        for layer in self.pre_layers:
            x = layer(x, training=False)
        return self.feature_model(x, training=False)

    def logits_from_features(self, features: tf.Tensor) -> tf.Tensor:
        x = self.tail_model(features, training=False)
        for layer in self.head_prefix:
            x = layer(x, training=False)
        if x.shape.rank != 2:
            raise ValueError(f"Expected 2D features before class head, got {x.shape}")
        logits = tf.linalg.matmul(x, self.class_head.kernel)
        if self.class_head.use_bias:
            logits = tf.nn.bias_add(logits, self.class_head.bias)
        return logits

    def explain(
        self,
        raw_rgb_255: np.ndarray,
        expected_probabilities: np.ndarray,
        target_index: int,
        ablation_batch_size: int,
        prediction_tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
        caffe = preprocess_input(np.array(raw_rgb_255[None], dtype=np.float32, copy=True))
        batch = tf.convert_to_tensor(caffe, dtype=tf.float32)
        probabilities = np.asarray(self.class_model(batch, training=False).numpy())[0]
        if probabilities.shape != expected_probabilities.shape:
            raise ValueError(
                f"Probability shape {probabilities.shape} differs from CSV "
                f"{expected_probabilities.shape}"
            )
        difference = float(np.max(np.abs(probabilities - expected_probabilities)))
        if difference > prediction_tolerance:
            raise RuntimeError(
                f"Checkpoint prediction differs from saved CSV by {difference:.6g} "
                f"(tolerance {prediction_tolerance}). Check model and preprocessing."
            )

        feature_maps = self.features(batch)
        if feature_maps.shape.rank != 4 or feature_maps.shape[0] != 1:
            raise ValueError(f"Expected one NHWC feature map, got {feature_maps.shape}")
        channels = int(feature_maps.shape[-1])
        if channels < 1:
            raise ValueError("Target layer has no feature channels")
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(feature_maps)
            baseline_logits = self.logits_from_features(feature_maps)
            target_score = baseline_logits[:, target_index]
        gradients = tape.gradient(target_score, feature_maps)
        if gradients is None:
            raise RuntimeError("Target class score is disconnected from the feature maps")
        reconstructed = tf.nn.softmax(baseline_logits, axis=-1).numpy()[0]
        if float(np.max(np.abs(reconstructed - probabilities))) > prediction_tolerance:
            raise RuntimeError(
                "Ablation split does not reproduce the original class prediction. "
                "Check the target layer and nested model structure."
            )

        # Grad-CAM: average target-logit gradients over space, then combine
        # the same feature maps used below for Ablation-CAM.
        gradient_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        grad_cam = tf.nn.relu(
            tf.reduce_sum(feature_maps * gradient_weights, axis=-1)
        )[0].numpy()

        baseline_score = float(target_score.numpy()[0])
        drops = np.empty(channels, dtype=np.float32)
        for start in range(0, channels, ablation_batch_size):
            stop = min(start + ablation_batch_size, channels)
            indices = tf.range(start, stop)
            mask = 1.0 - tf.one_hot(indices, channels, dtype=feature_maps.dtype)
            mask = mask[:, None, None, :]
            ablated_features = feature_maps * mask
            ablated_scores = self.logits_from_features(ablated_features)
            drops[start:stop] = (
                baseline_score - ablated_scores[:, target_index].numpy()
            )

        # Ablation-CAM weights channels by the drop in the pre-softmax class
        # score. A positive constant divisor does not affect a normalized CAM;
        # using absolute drops also behaves sensibly when the logit is <= 0.
        weights = np.maximum(drops, 0.0)
        maps = feature_maps.numpy()[0]
        ablation_cam = np.maximum(np.sum(maps * weights[None, None, :], axis=-1), 0.0)

        def resize_cam(cam: np.ndarray, method: str) -> np.ndarray:
            if not np.isfinite(cam).all():
                raise RuntimeError(f"{method} contains non-finite values")
            peak = float(np.max(cam))
            if peak > 0:
                cam = cam / peak
            return tf.image.resize(
                cam[..., None], raw_rgb_255.shape[:2], method="bilinear"
            ).numpy()[..., 0]

        return (
            resize_cam(grad_cam, "Grad-CAM"),
            resize_cam(ablation_cam, "Ablation-CAM"),
            probabilities,
            baseline_score,
            int(np.sum(weights > 0)),
            channels,
        )


def embedded_convae(model: tf.keras.Model) -> tf.keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and (
            "autoencoder" in layer.name.lower() or "convae" in layer.name.lower()
        ):
            return layer
    return None


def make_figure(path: Path, sample: dict, model_names: list[str], alpha: float) -> None:
    panels = [
        (sample["clean"] / 255.0, "Clean original", None),
        (sample["raw"] / 255.0, "Evaluated image", None),
    ]
    if sample.get("reconstruction") is not None:
        panels.append((sample["reconstruction"], "ConvAE reconstruction", None))
    for name in model_names:
        result = sample["models"][name]
        panels.append((sample["raw"] / 255.0, name, result))

    fig, axes = plt.subplots(
        2, len(panels), figsize=(3.8 * len(panels), 8.6), squeeze=False
    )
    for row_index, (method, cam_key) in enumerate(
        (("Grad-CAM", "grad_cam"), ("Ablation-CAM", "ablation_cam"))
    ):
        for ax, (image, title, result) in zip(axes[row_index], panels):
            if result is None:
                ax.imshow(np.clip(image, 0, 1))
                ax.set_title(title)
            else:
                heat = plt.get_cmap("jet")(result[cam_key])[..., :3]
                overlay = (1.0 - alpha) * image + alpha * heat
                ax.imshow(np.clip(overlay, 0, 1))
                ax.set_title(
                    f"{title}\npred={result['predicted']} ({result['confidence']:.1%})"
                )
            ax.axis("off")
        fig.text(0.01, 0.74 - 0.47 * row_index, method, rotation=90, fontsize=12)
    fig.suptitle(
        f"{sample['dataset']} | {sample['key'][0]}/{sample['key'][1]} | "
        "True-class explanations (each map normalized separately)",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_single_figure(
    path: Path,
    raw_rgb_255: np.ndarray,
    cam: np.ndarray,
    actual: str,
    predicted: str,
    confidence: float,
    target: str,
    alpha: float,
) -> None:
    """Save the attacked input, Ablation-CAM heatmap and aligned overlay."""
    image = np.clip(raw_rgb_255 / 255.0, 0.0, 1.0)
    heat = plt.get_cmap("jet")(cam)[..., :3]
    overlay = np.clip((1.0 - alpha) * image + alpha * heat, 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, content, title in zip(
        axes,
        (image, cam, overlay),
        ("Input before defences", "Ablation-CAM heatmap", "Overlay"),
    ):
        if title == "Ablation-CAM heatmap":
            ax.imshow(content, cmap="jet", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(content)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"Predicted: {predicted} ({confidence:.1%})   |   Actual: {actual}"
        f"   |   CAM target: {target}",
        color="green" if predicted == actual else "red",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single_image(args: argparse.Namespace) -> None:
    """Explain the same one-image pipeline named by its test_all.py CSV."""
    modelling_dir = Path(__file__).resolve().parent
    config = load_config(args.config)
    class_names = config["class_names"]
    image_size = args.image_size or int(config["image_size"])
    if image_size < 1:
        raise ValueError("--image-size must be positive")
    tf.keras.utils.set_random_seed(int(config.get("seed", 4266)))

    model_path = resolve_project_path(args.model, modelling_dir)
    predictions_path = resolve_project_path(args.predictions, modelling_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    rows = read_predictions(predictions_path, class_names)
    wanted = Path(args.image).name
    matches = [
        row for row in rows.values()
        if Path(row["file_path"]).name == wanted
        or Path(row["file_path"]).stem == wanted
    ]
    if len(matches) != 1:
        raise ValueError(
            f"--image {args.image!r} matched {len(matches)} CSV rows; "
            "provide its exact filename"
        )
    row = matches[0]
    image_path = resolve_project_path(row["file_path"], modelling_dir)
    if not image_path.is_file():
        raise FileNotFoundError(f"Evaluated image not found: {image_path}")

    full_model = load_classifier(
        model_path, image_size=image_size, num_classes=len(class_names)
    )
    if tuple(full_model.input_shape[1:3]) != (image_size, image_size):
        raise ValueError(
            f"Model expects {full_model.input_shape[1:3]}, "
            f"but image_size={image_size}"
        )
    if args.preprocessing == "convae_jpeg":
        from modules.convae import is_wrapped_with_convae

        if is_wrapped_with_convae(full_model):
            raise ValueError("Model already embeds ConvAE; do not add a second ConvAE")
    full_model.trainable = False
    prepare_input, process_scores = load_defence_pipeline(
        args, modelling_dir, image_size
    )
    raw_rgb = read_image(image_path, image_size)
    processed_rgb = prepare_input(raw_rgb)
    caffe = preprocess_input(
        np.array(processed_rgb[None], dtype=np.float32, copy=True)
    )

    ablator = AblationModel(full_model, args.layer, len(class_names))
    base_probabilities = np.asarray(
        ablator.class_model(caffe, training=False).numpy(), dtype=np.float32
    )[0]
    target = row["actual_class"] if args.target == "true" else row["predicted_class"]
    _, ablation_cam, probabilities, _, positive, channels = ablator.explain(
        processed_rgb,
        base_probabilities,
        class_names.index(target),
        args.ablation_batch_size,
        args.prediction_tolerance,
    )
    final_probabilities = np.asarray(process_scores(probabilities), dtype=np.float64)
    expected = np.array(
        [float(row[f"probability_{name}"]) for name in class_names],
        dtype=np.float64,
    )
    difference = float(np.max(np.abs(final_probabilities - expected)))
    if not np.isfinite(final_probabilities).all() or difference > args.prediction_tolerance:
        raise RuntimeError(
            f"Pipeline probabilities differ from predictions.csv by {difference:.6g} "
            f"(tolerance {args.prediction_tolerance}). Check checkpoint and defences."
        )
    predicted_index = int(np.argmax(final_probabilities))
    predicted = class_names[predicted_index]
    if predicted != row["predicted_class"]:
        raise RuntimeError(
            f"Pipeline predicts {predicted}, but CSV records {row['predicted_class']}"
        )

    outcome = "correct" if row["correct"].strip().lower() == "true" else "incorrect"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
    figure_path = args.output_dir / outcome / f"{safe_stem}_ablation_cam.png"
    make_single_figure(
        figure_path,
        raw_rgb,
        ablation_cam,
        row["actual_class"],
        predicted,
        float(final_probabilities[predicted_index]),
        target,
        args.alpha,
    )
    print(
        f"Saved {figure_path} | predicted={predicted} | actual={row['actual_class']} "
        f"| positive ablated channels={positive}/{channels}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.ablation_batch_size < 1:
        raise ValueError("--ablation-batch-size must be positive")
    if args.prediction_tolerance <= 0:
        raise ValueError("--prediction-tolerance must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if any(value is not None for value in (args.model, args.predictions, args.image)):
        if not all(value is not None for value in (args.model, args.predictions, args.image)):
            raise ValueError(
                "Single-image mode requires --model, --predictions and --image together"
            )
        run_single_image(args)
        return
    if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Model and dataset names must not be repeated")

    modelling_dir = Path(__file__).resolve().parent
    config = load_config(args.config)
    class_names = config["class_names"]
    image_size = int(config["image_size"])
    if image_size < 1:
        raise ValueError("image_size must be positive")
    tf.keras.utils.set_random_seed(int(config.get("seed", 4266)))
    for name in args.models:
        if name not in config["models"]:
            raise ValueError(f"Unknown model {name!r}; configured: {list(config['models'])}")
    for name in args.datasets:
        if name not in config["datasets"]:
            raise ValueError(f"Unknown dataset {name!r}; configured: {list(config['datasets'])}")

    results_root = resolve_project_path(
        args.results_dir or config["output_dir"], modelling_dir
    )
    clean_dir = resolve_project_path(config["datasets"]["clean_test"], modelling_dir)
    if not results_root.is_dir():
        raise FileNotFoundError(f"test_all.py results directory not found: {results_root}")
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Clean test directory not found: {clean_dir}")

    # Select matched cases from the first model's CSV. Each model must use the
    # exact same dataset, class label and underlying image filename.
    samples: dict[tuple[str, tuple[str, str]], dict] = {}
    prediction_rows: dict[tuple[str, str], dict] = {}
    for dataset in args.datasets:
        dataset_dir = resolve_project_path(config["datasets"][dataset], modelling_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
        for model_name in args.models:
            csv_path = (
                results_root
                / "tests"
                / f"pre_none__model_{model_name}__post_none"
                / dataset
                / "predictions.csv"
            )
            prediction_rows[(dataset, model_name)] = read_predictions(
                csv_path, class_names
            )
        reference = prediction_rows[(dataset, args.models[0])]
        for selector in args.cases:
            key = select_case(selector, reference)
            reference_row = reference[key]
            for model_name in args.models[1:]:
                other_rows = prediction_rows[(dataset, model_name)]
                if key not in other_rows:
                    raise ValueError(f"{key} missing for model {model_name} on {dataset}")
                if Path(other_rows[key]["file_path"]).name != Path(
                    reference_row["file_path"]
                ).name:
                    raise ValueError(f"Image filename differs across models for {key}")
            image_path = image_path_from_row(reference_row, dataset_dir)
            samples[(dataset, key)] = {
                "dataset": dataset,
                "key": key,
                "path": image_path,
                "raw": read_image(image_path, image_size),
                "clean": find_clean_image(clean_dir, key, image_size),
                "models": {},
                "reconstruction": None,
            }

    model_paths = {
        name: resolve_project_path(config["models"][name], modelling_dir)
        for name in args.models
    }
    for name, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for {name} not found: {path}. "
                "Run this script where the saved .keras files are available."
            )

    summary_rows: list[dict] = []
    for model_name in args.models:
        path = model_paths[model_name]
        print(f"Loading {model_name}: {path}", flush=True)
        full_model = load_classifier(path, image_size=image_size, num_classes=len(class_names))
        full_model.trainable = False
        if tuple(full_model.input_shape[1:3]) != (image_size, image_size):
            raise ValueError(
                f"{model_name} expects {full_model.input_shape[1:3]}, "
                f"but config specifies {(image_size, image_size)}"
            )
        ablator = AblationModel(full_model, args.layer, len(class_names))
        autoencoder = embedded_convae(full_model) if model_name == "convae_rn50" else None

        for (dataset, key), sample in samples.items():
            row = prediction_rows[(dataset, model_name)][key]
            expected = np.array(
                [float(row[f"probability_{name}"]) for name in class_names],
                dtype=np.float32,
            )
            target_index = class_names.index(key[0])
            grad_cam, ablation_cam, probabilities, target_logit, positive, channels = ablator.explain(
                sample["raw"],
                expected,
                target_index,
                args.ablation_batch_size,
                args.prediction_tolerance,
            )
            predicted = class_names[int(np.argmax(probabilities))]
            if predicted != row["predicted_class"]:
                raise RuntimeError(
                    f"{model_name} predicts {predicted} but CSV says "
                    f"{row['predicted_class']} for {dataset}/{key}"
                )
            confidence = float(np.max(probabilities))
            sample["models"][model_name] = {
                "grad_cam": grad_cam,
                "ablation_cam": ablation_cam,
                "predicted": predicted,
                "confidence": confidence,
            }
            if autoencoder is not None:
                rgb01 = tf.convert_to_tensor(sample["raw"][None] / 255.0)
                reconstruction = autoencoder(rgb01, training=False)
                if isinstance(reconstruction, (list, tuple)):
                    reconstruction = reconstruction[0]
                reconstructed = np.asarray(reconstruction.numpy()[0])
                if reconstructed.shape != sample["raw"].shape:
                    raise ValueError(f"Unexpected ConvAE output shape: {reconstructed.shape}")
                sample["reconstruction"] = np.clip(reconstructed, 0.0, 1.0)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "original_class": key[0],
                    "original_stem": key[1],
                    "image_path": str(sample["path"]),
                    "model": model_name,
                    "checkpoint": str(path),
                    "predicted_class": predicted,
                    "confidence": confidence,
                    "target_logit": target_logit,
                    "positive_channels": positive,
                    "channels": channels,
                    "target_layer": args.layer,
                }
            )
            print(
                f"{dataset} {key[0]}/{key[1]} {model_name}: "
                f"pred={predicted}, positive_channels={positive}/{channels}",
                flush=True,
            )
        del ablator, autoencoder, full_model
        tf.keras.backend.clear_session()
        gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, key), sample in samples.items():
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{key[0]}_{key[1]}") + ".png"
        figure_path = args.output_dir / dataset / filename
        make_figure(figure_path, sample, args.models, args.alpha)
        print(f"Saved {figure_path}", flush=True)

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
