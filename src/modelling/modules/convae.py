from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Default hyperparameters
IMG_SIZE = 256
CHANNELS = 3
HIDDEN_DIM = 1024
LATENT_DIM = 256
NOISE_STD = 0.2
BATCH_SIZE = 32
EPOCHS = 20

# Caffe ResNet preprocess_input channel means in BGR order.
# Raw RGB pixels in [0, 255] map to BGR in [0, 255] minus these means.
CAFFE_MEAN_BGR = (103.939, 116.779, 123.68)


def _is_true(value, default: bool = False) -> bool:
    """Robust bool check that handles strings from YAML or W&B configs."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


# def build_autoencoder(
#     img_size: int = IMG_SIZE,
#     channels: int = CHANNELS,
#     hidden_dim: int = HIDDEN_DIM,
#     latent_dim: int = LATENT_DIM,
#     noise_std: float = NOISE_STD,
#     name: str = "Scaled_Conv_Autoencoder",
# ) -> tf.keras.Model:
#     """Build the 5-stage strided convolutional autoencoder."""
#     img_size = int(img_size)
#     channels = int(channels)
#     hidden_dim = int(hidden_dim)
#     latent_dim = int(latent_dim)
#     noise_std = float(noise_std)

#     if img_size % 32 != 0:
#         raise ValueError(
#             f"img_size must be divisible by 32 for 5 stride-2 stages, got {img_size}"
#         )
#     bottleneck_grid = img_size // 32
#     bottleneck_units = bottleneck_grid * bottleneck_grid * 128

#     # --- ENCODER ---
#     encoder_input = layers.Input(
#         shape=(img_size, img_size, channels), name="encoder_input"
#     )

#     # 5 Strided Conv layers: 256 -> 128 -> 64 -> 32 -> 16 -> 8 (for img_size=256)
#     x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(
#         encoder_input
#     )
#     x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)

#     # Flatten (8 * 8 * 128 = 8192) and compress through Dense layers
#     x = layers.Flatten()(x)
#     x = layers.Dense(hidden_dim, activation="relu")(x)
#     latent = layers.Dense(latent_dim, name="latent_space")(x)

#     # Add Random Gaussian Noise at the bottleneck (active during training)
#     noisy_latent = layers.GaussianNoise(noise_std)(latent)

#     # --- DECODER ---
#     # Mirror the Dense layers: 256 -> 1024 -> 8192
#     x = layers.Dense(hidden_dim, activation="relu")(noisy_latent)
#     x = layers.Dense(bottleneck_units, activation="relu")(x)
#     x = layers.Reshape((bottleneck_grid, bottleneck_grid, 128))(x)

#     # 5 Strided Conv2DTranspose layers: 8 -> 16 -> 32 -> 64 -> 128 -> 256
#     x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
#     x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

#     # Final reconstruction layer (Sigmoid keeps pixel values in [0, 1])
#     decoder_output = layers.Conv2DTranspose(
#         channels,
#         3,
#         strides=2,
#         padding="same",
#         activation="sigmoid",
#         name="reconstructed_output",
#     )(x)

#     return models.Model(encoder_input, decoder_output, name=name)

def build_autoencoder(
    img_size: int = IMG_SIZE,
    channels: int = CHANNELS,
    hidden_dim: int = HIDDEN_DIM,
    latent_dim: int = LATENT_DIM,
    noise_std: float = NOISE_STD,
    name: str = "Scaled_Conv_Autoencoder",
) -> tf.keras.Model:
    """Build the 5-stage strided convolutional autoencoder following Waghela et al."""
    img_size = int(img_size)
    channels = int(channels)
    hidden_dim = int(hidden_dim)
    latent_dim = int(latent_dim)
    noise_std = float(noise_std)

    if img_size % 32 != 0:
        raise ValueError(
            f"img_size must be divisible by 32 for 5 stride-2 stages, got {img_size}"
        )
    bottleneck_grid = img_size // 32
    bottleneck_units = bottleneck_grid * bottleneck_grid * 128

    # --- ENCODER ---
    encoder_input = layers.Input(
        shape=(img_size, img_size, channels), name="encoder_input"
    )

    # Paper uses GELU instead of ReLU to avoid gradient discontinuity at zero,
    # and applies BatchNormalization after intermediate conv layers.
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="gelu")(
        encoder_input
    )
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="gelu")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="gelu")(x)

    # # Flatten (8 * 8 * 128 = 8192) and compress through Dense layers
    # x = layers.Flatten()(x)
    # x = layers.Dense(hidden_dim, activation="gelu")(x)
    # latent = layers.Dense(latent_dim, name="latent_space")(x)
    # Compress through Conv2D bottleneck
    x = layers.Conv2D(hidden_dim // (bottleneck_grid ** 2), 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    # Normalize latent space to unit variance so GaussianNoise cannot be bypassed
    latent_raw = layers.Conv2D(latent_dim // (bottleneck_grid ** 2), 1, padding="same", use_bias=False)(x)
    latent = layers.BatchNormalization(scale=False, name="latent_space")(latent_raw)

    # Add Random Gaussian Noise at the bottleneck
    noisy_latent = layers.GaussianNoise(noise_std)(latent)

    # --- DECODER ---
    x = layers.Conv2D(hidden_dim // (bottleneck_grid ** 2), 3, padding="same", activation="gelu")(noisy_latent)
    x = layers.Conv2D(128, 3, padding="same", activation="gelu")(x)

    # Paper applies BatchNormalization + GELU on the initial Conv2DTranspose stages
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="gelu")(x)

    # Final reconstruction layer (Sigmoid keeps pixel values in [0, 1])
    decoder_output = layers.Conv2DTranspose(
        channels,
        3,
        strides=2,
        padding="same",
        activation="sigmoid",
        name="reconstructed_output",
    )(x)

    return models.Model(encoder_input, decoder_output, name=name)

def _fixed_affine_1x1(name: str, kernel: np.ndarray, bias: np.ndarray) -> layers.Conv2D:
    """Create a non-trainable 1x1 Conv2D layer with fixed channel transform weights."""
    layer = layers.Conv2D(
        3,
        kernel_size=1,
        strides=1,
        padding="valid",
        use_bias=True,
        trainable=False,
        name=name,
    )
    layer.build((None, None, None, 3))
    layer.set_weights(
        [
            kernel.astype(np.float32).reshape((1, 1, 3, 3)),
            bias.astype(np.float32).reshape((3,)),
        ]
    )
    return layer


def caffe_to_rgb01_layer(name: str = "caffe_to_rgb01") -> layers.Conv2D:
    """Convert Caffe BGR (mean-subtracted [0, 255]) to RGB in [0, 1].

    Given input x = [B - mB, G - mG, R - mR]:
        R_01 = (x[2] + mR) / 255
        G_01 = (x[1] + mG) / 255
        B_01 = (x[0] + mB) / 255
    """
    mean_b, mean_g, mean_r = CAFFE_MEAN_BGR
    kernel = np.array(
        [
            [0.0, 0.0, 1.0 / 255.0],
            [0.0, 1.0 / 255.0, 0.0],
            [1.0 / 255.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    bias = np.array(
        [mean_r / 255.0, mean_g / 255.0, mean_b / 255.0],
        dtype=np.float32,
    )
    return _fixed_affine_1x1(name=name, kernel=kernel, bias=bias)


def rgb01_to_caffe_layer(name: str = "rgb01_to_caffe") -> layers.Conv2D:
    """Convert RGB in [0, 1] to Caffe BGR (mean-subtracted [0, 255]).

    Given input y = [R_01, G_01, B_01]:
        B_caffe = 255 * y[2] - mB
        G_caffe = 255 * y[1] - mG
        R_caffe = 255 * y[0] - mR
    """
    mean_b, mean_g, mean_r = CAFFE_MEAN_BGR
    kernel = np.array(
        [
            [0.0, 0.0, 255.0],
            [0.0, 255.0, 0.0],
            [255.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    bias = np.array([-mean_b, -mean_g, -mean_r], dtype=np.float32)
    return _fixed_affine_1x1(name=name, kernel=kernel, bias=bias)


def resolve_convae_path(raw_path: str | Path) -> Path:
    """Locate a ConvAE checkpoint locally or inside the Docker container."""
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate

    modelling_dir = Path(__file__).resolve().parents[1]
    repo_dir = modelling_dir.parents[1] if len(modelling_dir.parents) > 1 else modelling_dir

    search_candidates = [
        modelling_dir / candidate,
        repo_dir / candidate,
    ]
    normalized = str(raw_path).replace("\\", "/")
    if normalized.startswith("/app/"):
        rel_from_app = normalized[len("/app/") :]
        search_candidates.append(modelling_dir / rel_from_app)

    for path in search_candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"ConvAE checkpoint not found: {raw_path!r} "
        f"(checked {[str(p) for p in [candidate, *search_candidates]]})"
    )


def _as_single_io(model: tf.keras.Model) -> tf.keras.Model:
    """Re-wrap a functional model whose inputs/outputs were deserialized as 1-element lists."""
    ins, outs = model.inputs, model.outputs
    if (
        len(ins) == 1
        and len(outs) == 1
        and (
            isinstance(model.input, (list, tuple))
            or isinstance(model.output, (list, tuple))
        )
    ):
        return models.Model(ins[0], outs[0], name=model.name)
    return model


def load_convae(
    model_path: str | Path,
    img_size: int = IMG_SIZE,
    hidden_dim: int = HIDDEN_DIM,
    latent_dim: int = LATENT_DIM,
    noise_std: float = NOISE_STD,
) -> tf.keras.Model:
    """Load a trained ConvAE checkpoint, with fallback reconstruction."""
    resolved = resolve_convae_path(model_path)
    try:
        return _as_single_io(tf.keras.models.load_model(str(resolved), compile=False))
    except Exception:
        pass

    model = build_autoencoder(
        img_size=img_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        noise_std=noise_std,
    )
    if zipfile.is_zipfile(resolved):
        with zipfile.ZipFile(resolved, "r") as archive:
            with tempfile.TemporaryDirectory() as tmp_dir:
                extracted = archive.extract("model.weights.h5", tmp_dir)
                model.load_weights(extracted)
    else:
        model.load_weights(str(resolved))
    return model


def is_wrapped_with_convae(model: tf.keras.Model) -> bool:
    """Return True if ``model`` already has the ConvAE preprocessor attached."""
    layer_names = {layer.name for layer in model.layers}
    if "caffe_to_rgb01" in layer_names and "rgb01_to_caffe" in layer_names:
        return True
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "autoencoder" in layer.name.lower():
            return True
        if isinstance(layer, tf.keras.Model) and "convae" in layer.name.lower():
            return True
    return False


def parse_convae_config(config: dict) -> dict:
    """Extract normalized ``convae`` settings from a train/test config dict."""
    raw = config.get("convae")
    if isinstance(raw, bool):
        enabled = raw
        raw_dict = {}
    elif isinstance(raw, dict):
        enabled = _is_true(raw.get("enabled", False), default=False)
        raw_dict = raw
    else:
        enabled = _is_true(config.get("use_convae", False), default=False)
        raw_dict = {}

    model_path = raw_dict.get("model_path") or config.get("convae_model_path")
    trainable = _is_true(
        raw_dict.get("trainable", config.get("convae_trainable", False)),
        default=False,
    )
    return {
        "enabled": enabled,
        "model_path": model_path,
        "trainable": trainable,
    }


def wrap_with_convae(
    classifier: tf.keras.Model,
    convae: tf.keras.Model,
    trainable: bool = False,
    name: str = "convae_resnet_pipeline",
) -> tf.keras.Model:
    """Prepend a ConvAE preprocessor in front of a Caffe-space ResNet classifier.

    Pipeline:
        Input (Caffe BGR) -> caffe_to_rgb01 -> ConvAE (RGB [0,1]) -> rgb01_to_caffe -> Classifier
    """
    clf_shape = classifier.input_shape[1:]
    ae_shape = convae.input_shape[1:]
    if clf_shape != ae_shape:
        raise ValueError(
            f"ConvAE input shape {ae_shape} does not match classifier input shape {clf_shape}."
        )

    convae.trainable = bool(trainable)

    inputs = layers.Input(shape=clf_shape, name="input_caffe")
    x = caffe_to_rgb01_layer()(inputs)
    # When frozen, run the AE in inference mode so GaussianNoise is disabled.
    x = convae(x, training=None if trainable else False)
    if isinstance(x, (list, tuple)):
        x = x[0]
    x = rgb01_to_caffe_layer()(x)
    outputs = classifier(x)

    combined = models.Model(inputs, outputs, name=name)
    combined.classifier = classifier
    combined.convae = convae
    return combined


if __name__ == "__main__":
    autoencoder = build_autoencoder()
    autoencoder.summary()
