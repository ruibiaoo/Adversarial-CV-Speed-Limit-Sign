"""Grad-CAM for the speed-limit ResNet classifiers.

Reads the predictions CSV written by test.py, runs Grad-CAM on each image and
saves a figure (original | heatmap overlay) titled with predicted and actual
class. Output goes to <results>/gradcam/{correct,incorrect}/.

Example (inside the container):
    python -u gradcam.py --max-images 50 --only incorrect
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

LAST_CONV_LAYER = "conv5_block3_out"  # last conv block of ResNet50


def parse_args():
    phase_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Grad-CAM for ResNet50")
    parser.add_argument(
        "--model", default=str(phase_directory / "runs" / "best.keras")
    )
    parser.add_argument(
        "--predictions",
        default=str(phase_directory / "results" / "test" / "predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(phase_directory / "results" / "gradcam"),
    )
    parser.add_argument("--image-size", type=int, default=None,
                        help="Defaults to the model input size")
    parser.add_argument("--only", choices=["all", "correct", "incorrect"],
                        default="all")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images per outcome (0 = all)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Heatmap overlay strength")
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument(
        "--image",
        help="Explain only the CSV row whose image filename matches this name or stem.",
    )
    parser.add_argument(
        "--target", choices=("predicted", "true"), default="predicted",
        help="Class to explain. Use 'true' for matched cross-model comparisons.",
    )
    parser.add_argument(
        "--preprocessing", choices=("none", "convae_jpeg"), default="none",
        help="Apply the ConvAE then JPEG chain from test_all.yaml when requested.",
    )
    parser.add_argument(
        "--postprocessing", choices=("none", "highconf"), default="none",
        help="Apply High-Confidence Filtering to verify saved predictions; CAM remains pre-postprocessing.",
    )
    parser.add_argument(
        "--config", default=str(phase_directory / "configs" / "test_all.yaml"),
        help="Evaluation configuration used for defence parameters and paths.",
    )
    parser.add_argument(
        "--prediction-tolerance", type=float, default=1e-3,
        help="Maximum difference from the selected predictions.csv confidence.",
    )
    return parser.parse_args()


def resolve_image_path(raw_path, predictions_path):
    """Find an image path from the CSV, which may hold container paths."""
    path = Path(raw_path)
    if path.exists():
        return path
    # Try to re-anchor "/app/data/..." onto a local data folder.
    parts = path.parts
    if "data" in parts:
        tail = Path(*parts[parts.index("data") + 1:])
        for base in (
            Path("/app/data"),
            predictions_path.parents[3] / "data" / "data_processed",
        ):
            if (base / tail).exists():
                return base / tail
    raise FileNotFoundError(f"Image not found: {raw_path}")


def _has_direct_layer(model_or_layer, layer_name):
    if not isinstance(model_or_layer, tf.keras.Model):
        return False
    try:
        model_or_layer.get_layer(layer_name)
        return True
    except ValueError:
        return False


def _locate_backbone_and_stages(model, target_layer_name=LAST_CONV_LAYER):
    """Find the ResNet backbone containing ``target_layer_name`` and surrounding layers.

    Supports both standard classifiers (Input -> ResNet -> Dropout -> Dense) and
    ConvAE-wrapped pipelines (Input -> caffe_to_rgb01 -> ConvAE -> rgb01_to_caffe -> Classifier).
    """
    non_input_layers = [
        l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)
    ]

    # Case 1: ResNet backbone is a direct child of `model`.
    for idx, layer in enumerate(non_input_layers):
        if _has_direct_layer(layer, target_layer_name):
            pre_layers = non_input_layers[:idx]
            head_layers = non_input_layers[idx + 1 :]
            return pre_layers, layer, head_layers

    # Case 2: ResNet backbone is nested inside a child classifier model (e.g. wrap_with_convae).
    for idx, layer in enumerate(non_input_layers):
        if isinstance(layer, tf.keras.Model):
            inner_non_input = [
                l
                for l in layer.layers
                if not isinstance(l, tf.keras.layers.InputLayer)
            ]
            for inner_idx, inner_layer in enumerate(inner_non_input):
                if _has_direct_layer(inner_layer, target_layer_name):
                    pre_layers = (
                        non_input_layers[:idx] + inner_non_input[:inner_idx]
                    )
                    head_layers = (
                        inner_non_input[inner_idx + 1 :]
                        + non_input_layers[idx + 1 :]
                    )
                    return pre_layers, inner_layer, head_layers

    raise ValueError(
        f"Expected a nested ResNet backbone containing layer {target_layer_name!r}."
    )


def build_grad_model(model):
    """Return a function image_batch -> (conv_maps, class_logits, probabilities).

    The training script wraps ResNet50 as a nested model (optionally preceded by
    a ConvAE preprocessor), so the conv layer lives inside the ResNet submodel
    and any preceding/head layers are applied around it.
    """
    pre_layers, backbone, head_layers = _locate_backbone_and_stages(
        model, LAST_CONV_LAYER
    )
    if not head_layers or not isinstance(head_layers[-1], tf.keras.layers.Dense):
        raise ValueError("Expected a Dense classification head after the ResNet backbone")
    class_head = head_layers[-1]
    if getattr(class_head.activation, "__name__", "") != "softmax":
        raise ValueError("Expected a softmax Dense classification head")

    backbone_output = backbone.output
    if isinstance(backbone_output, (list, tuple)) and len(backbone_output) == 1:
        backbone_output = backbone_output[0]

    backbone_model = tf.keras.Model(
        backbone.input,
        [backbone.get_layer(LAST_CONV_LAYER).output, backbone_output],
    )

    def forward(batch):
        x = batch
        for layer in pre_layers:
            x = layer(x, training=False)
        conv_maps, x = backbone_model(x, training=False)
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        for layer in head_layers[:-1]:
            x = layer(x, training=False)
        logits = tf.linalg.matmul(x, class_head.kernel)
        if class_head.use_bias:
            logits = tf.nn.bias_add(logits, class_head.bias)
        probabilities = class_head(x, training=False)
        return conv_maps, logits, probabilities

    return forward


def make_heatmap(forward, batch, class_index=None):
    """Compute Grad-CAM from the pre-softmax class score, avoiding softmax saturation."""
    with tf.GradientTape() as tape:
        conv_maps, logits, predictions = forward(batch)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        score = logits[:, class_index]

    grads = tape.gradient(score, conv_maps)
    if grads is None:
        raise RuntimeError("Class score is disconnected from the target feature maps")
    weights = tf.reduce_mean(grads, axis=(1, 2))  # global-average pool
    cam = tf.reduce_sum(conv_maps * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    if not np.isfinite(cam).all():
        raise RuntimeError("Grad-CAM contains non-finite values")
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, np.asarray(predictions.numpy()[0], dtype=np.float64)


def load_defence_pipeline(args, phase_directory, image_size):
    """Reproduce the selected deterministic test_all.py input/output defences."""
    if args.preprocessing == "none" and args.postprocessing == "none":
        return lambda raw: raw, lambda probabilities: probabilities

    import yaml
    from test_all import resolve_path
    from modules import test_time_defences as ttd

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    defences_dir = resolve_path(config.get("defences_dir", "../defenses/modules"), phase_directory)

    if args.preprocessing == "convae_jpeg":
        from modules.convae import load_convae, resolve_convae_path

        pre_params = config["preprocessing"]["defences"]
        ae_path = resolve_convae_path(resolve_path(pre_params["convae"]["model_path"], phase_directory))
        if not ae_path.is_file():
            raise FileNotFoundError(f"ConvAE checkpoint not found: {ae_path}")
        autoencoder = load_convae(ae_path, img_size=image_size)
        autoencoder.trainable = False
        quality = int(pre_params["jpeg"]["quality"])
        pre_module = ttd._load_file(
            defences_dir / "preprocessing" / "defences.py", "gradcam_pre_defences"
        )

        def prepare_input(raw):
            rgb01 = np.clip(np.asarray(raw, dtype=np.float32)[None] / 255.0, 0.0, 1.0)
            reconstructed = np.asarray(autoencoder.predict_on_batch(rgb01))
            jpeg_input = pre_module._to_uint8(np.clip(reconstructed, 0.0, 1.0) * 255.0)[0]
            return pre_module.jpeg_compress(jpeg_input, quality)
    else:
        prepare_input = lambda raw: raw

    if args.postprocessing == "highconf":
        post_params = config["postprocessing"]["defences"]
        post_module = ttd._load_file(
            defences_dir / "post-processing" / "defences.py", "gradcam_post_defences"
        )

        def process_scores(probabilities):
            return post_module.apply_chain(
                np.asarray(probabilities, dtype=np.float64)[None],
                [ttd.POST_NAMES["highconf"]],
                {ttd.POST_NAMES["highconf"]: post_params["highconf"]},
            )[0]
    else:
        process_scores = lambda probabilities: probabilities

    return prepare_input, process_scores


def upscale(cam, size):
    cam = tf.image.resize(cam[..., None], (size, size)).numpy()[..., 0]
    return np.clip(cam, 0.0, 1.0)


def save_figure(path, image, cam, predicted, actual, confidence, alpha, target):
    colormap = matplotlib.colormaps["jet"]
    heat = colormap(cam)[..., :3]
    overlay = np.clip((1 - alpha) * image + alpha * heat, 0, 1)

    correct = predicted == actual
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(image)
    axes[0].set_title("Input before defences")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Predicted: {predicted} ({confidence:.1%})   |   Actual: {actual}"
        f"   |   CAM target: {target}",
        color="green" if correct else "red",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_classifier(model_path, image_size=256, num_classes=7):
    """Load model, falling back to architecture reconstruction for Keras 2/3 cross-compatibility."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        pass

    import tempfile
    import zipfile
    from modules.model import build_model

    model = build_model(
        {"model_name": "resnet50", "image_size": image_size, "pretrained": False},
        num_classes=num_classes,
    )
    with zipfile.ZipFile(model_path, "r") as z:
        with tempfile.TemporaryDirectory() as tmp_dir:
            extracted_path = z.extract("model.weights.h5", tmp_dir)
            model.load_weights(extracted_path)
    return model


def main():
    args = parse_args()
    if args.prediction_tolerance <= 0:
        raise ValueError("--prediction-tolerance must be positive")
    tf.keras.utils.set_random_seed(args.seed)

    phase_directory = Path(__file__).resolve().parent
    model_path = Path(args.model)
    predictions_path = Path(args.predictions)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    class_names = [
        c.replace("probability_", "")
        for c in pd.read_csv(predictions_path, nrows=0).columns
        if c.startswith("probability_")
    ]
    num_classes = len(class_names) or 7

    size = args.image_size
    if size is None:
        test_yaml_path = phase_directory / "configs" / "test.yaml"
        if test_yaml_path.exists():
            import yaml
            with open(test_yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                size = int(cfg.get("image_size", 256))
        else:
            size = 256

    model = load_classifier(model_path, image_size=size, num_classes=num_classes)
    size = args.image_size or int(model.input_shape[1])
    if args.preprocessing == "convae_jpeg":
        from modules.convae import is_wrapped_with_convae

        if is_wrapped_with_convae(model):
            raise ValueError("Model already embeds ConvAE; do not add a second ConvAE")
    forward = build_grad_model(model)
    prepare_input, process_scores = load_defence_pipeline(args, phase_directory, size)

    df = pd.read_csv(predictions_path)
    df["correct"] = df["correct"].astype(str).str.lower() == "true"
    if args.image:
        wanted = Path(args.image).name
        matches = df["file_path"].map(
            lambda raw_path: Path(raw_path).name == wanted or Path(raw_path).stem == wanted
        )
        df = df[matches]
        if len(df) != 1:
            raise ValueError(
                f"--image {args.image!r} matched {len(df)} CSV rows; provide its exact filename"
            )
        if args.only != "all" and bool(df.iloc[0]["correct"]) != (args.only == "correct"):
            raise ValueError("The selected image does not match --only")

    output_dir = Path(args.output_dir)
    groups = {"correct": df[df["correct"]], "incorrect": df[~df["correct"]]}

    total = 0
    for outcome, subset in groups.items():
        if args.only not in ("all", outcome):
            continue
        if args.max_images > 0 and len(subset) > args.max_images:
            subset = subset.sample(args.max_images, random_state=args.seed)

        folder = output_dir / outcome
        folder.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(subset.itertuples(index=False)):
            image_path = resolve_image_path(row.file_path, predictions_path)
            raw = tf.io.decode_image(
                tf.io.read_file(str(image_path)), channels=3,
                expand_animations=False,
            )
            raw = tf.image.resize(raw, (size, size))
            raw_rgb = np.asarray(raw.numpy(), dtype=np.float32)
            filtered_rgb = prepare_input(raw_rgb)
            batch = preprocess_input(
                np.array(filtered_rgb[None], dtype=np.float32, copy=True)
            )

            target_class = row.actual_class if args.target == "true" else row.predicted_class
            target_index = class_names.index(target_class)
            heatmap, base_probabilities = make_heatmap(forward, batch, target_index)
            final_probabilities = process_scores(base_probabilities)
            final_index = int(np.argmax(final_probabilities))
            if class_names[final_index] != row.predicted_class:
                raise RuntimeError(
                    f"Pipeline predicts {class_names[final_index]}, but CSV records "
                    f"{row.predicted_class} for {row.file_path}. Check model and defences."
                )
            if abs(float(final_probabilities[final_index]) - float(row.confidence)) > args.prediction_tolerance:
                raise RuntimeError(
                    f"Pipeline confidence differs from CSV for {row.file_path}. "
                    "Check model checkpoint, preprocessing and postprocessing."
                )
            cam = upscale(heatmap, size)
            image = np.clip(raw_rgb / 255.0, 0, 1)

            out_name = (
                f"{i:04d}_pred-{row.predicted_class}_actual-"
                f"{row.actual_class}_{image_path.stem[:40]}.png"
            )
            save_figure(
                folder / out_name, image, cam, row.predicted_class,
                row.actual_class, float(row.confidence), args.alpha, target_class,
            )
            total += 1

        print(f"{outcome}: saved to {folder}")

    print(f"Done. {total} Grad-CAM images in {output_dir}")


if __name__ == "__main__":
    main()
