"""Grad-CAM for the Phase 1 ResNet classifier.

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


def build_grad_model(model):
    """Return a function image_batch -> (conv_maps, predictions).

    The training script wraps ResNet50 as a nested model, so the conv layer
    lives inside it and the dropout/dense head must be re-applied by hand.
    """
    backbone = next(
        (l for l in model.layers if isinstance(l, tf.keras.Model)), None
    )
    if backbone is None:
        raise ValueError("Expected a nested ResNet backbone in the model.")

    backbone_output = backbone.output
    if isinstance(backbone_output, (list, tuple)) and len(backbone_output) == 1:
        backbone_output = backbone_output[0]

    backbone_model = tf.keras.Model(
        backbone.input,
        [backbone.get_layer(LAST_CONV_LAYER).output, backbone_output],
    )
    head_layers = model.layers[model.layers.index(backbone) + 1:]

    def forward(batch):
        conv_maps, x = backbone_model(batch, training=False)
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        for layer in head_layers:
            x = layer(x, training=False)
        return conv_maps, x

    return forward


def make_heatmap(forward, batch, class_index=None):
    """Compute Grad-CAM heatmap (H x W in [0, 1]) for one preprocessed image."""
    with tf.GradientTape() as tape:
        conv_maps, predictions = forward(batch)
        if len(predictions.shape) == 3 and predictions.shape[1] == 1:
            predictions = tf.squeeze(predictions, axis=1)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        score = predictions[:, class_index]

    grads = tape.gradient(score, conv_maps)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # global-average pool
    cam = tf.reduce_sum(conv_maps * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam


def upscale(cam, size):
    cam = tf.image.resize(cam[..., None], (size, size)).numpy()[..., 0]
    return np.clip(cam, 0.0, 1.0)


def save_figure(path, image, cam, predicted, actual, confidence, alpha):
    colormap = matplotlib.colormaps["jet"]
    heat = colormap(cam)[..., :3]
    overlay = np.clip((1 - alpha) * image + alpha * heat, 0, 1)

    correct = predicted == actual
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Predicted: {predicted} ({confidence:.1%})   |   Actual: {actual}",
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
    forward = build_grad_model(model)

    df = pd.read_csv(predictions_path)
    df["correct"] = df["correct"].astype(str).str.lower() == "true"

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
            batch = preprocess_input(tf.expand_dims(tf.cast(raw, tf.float32), 0))

            # Explain the model's predicted class.
            predicted_index = class_names.index(row.predicted_class)
            cam = upscale(make_heatmap(forward, batch, predicted_index), size)
            image = np.clip(raw.numpy() / 255.0, 0, 1)

            out_name = (
                f"{i:04d}_pred-{row.predicted_class}_actual-"
                f"{row.actual_class}_{image_path.stem[:40]}.png"
            )
            save_figure(
                folder / out_name, image, cam, row.predicted_class,
                row.actual_class, float(row.confidence), args.alpha,
            )
            total += 1

        print(f"{outcome}: saved to {folder}")

    print(f"Done. {total} Grad-CAM images in {output_dir}")


if __name__ == "__main__":
    main()
