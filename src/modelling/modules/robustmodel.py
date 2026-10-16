"""Adversarially robust ResNet classifier implementing Equation (3) from:

    Engstrom et al., "Adversarial Robustness as a Prior for Learned Representations"
    (arXiv:1906.00945v2, page 4):

        theta* = argmin_theta E_{(x,y)~D} [ max_{delta in Delta} L_theta(x + delta, y) ]

The inner maximization over Delta (an l2 or linf ball of radius eps in [0, 1]
image space) is solved via Projected Gradient Descent (PGD) during ``train_step``,
while validation and inference run on unperturbed inputs.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import types
import zipfile

import tensorflow as tf

from .resnet50 import DEFAULT_BLOCKS, ResNet50, _is_keras2_h5, load_pretrained_weights

RESNET101_BLOCKS = (3, 4, 23, 3)

# Caffe ResNet preprocess_input channel means in BGR order.
# Raw RGB pixels in [0, 1] map to BGR in [0, 255] minus these means.
CAFFE_MEAN_BGR = (103.939, 116.779, 123.68)


def _is_true(value, default: bool = False) -> bool:
    """Robust bool check that handles strings from YAML or W&B configs."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_pretrained_setting(pretrained_value) -> tuple[str, str | None]:
    """Classify the ``pretrained`` config value into ('imagenet' | 'none' | 'checkpoint', path)."""
    if pretrained_value is None:
        return "none", None
    if isinstance(pretrained_value, bool):
        return ("imagenet", None) if pretrained_value else ("none", None)

    text = str(pretrained_value).strip()
    lower = text.lower()
    if lower in ("", "0", "false", "no", "n", "off", "none", "null"):
        return "none", None
    if lower in ("1", "true", "yes", "y", "on", "imagenet"):
        return "imagenet", None
    return "checkpoint", text


def _resolve_checkpoint_path(raw_path: str) -> Path:
    """Locate a checkpoint path locally or inside the Docker container."""
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate

    modelling_dir = Path(__file__).resolve().parents[1]
    repo_dir = modelling_dir.parents[1]

    search_candidates = [
        modelling_dir / candidate,
        repo_dir / candidate,
    ]
    # Map container paths like "/app/runs/best.keras" when running on the host.
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("/app/"):
        rel_from_app = normalized[len("/app/") :]
        search_candidates.append(modelling_dir / rel_from_app)

    for path in search_candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Pretrained checkpoint not found: {raw_path!r} "
        f"(checked {[str(p) for p in [candidate, *search_candidates]]})"
    )


def _copy_matching_weights(target_model: tf.keras.Model, source_model: tf.keras.Model) -> tuple[int, int]:
    """Copy weights layer-by-layer (including nested Functional backbones) where shapes match."""
    loaded = 0
    skipped = 0
    source_layers = {layer.name: layer for layer in source_model.layers}

    for target_layer in target_model.layers:
        source_layer = source_layers.get(target_layer.name)
        if source_layer is None and isinstance(target_layer, tf.keras.Model):
            # Fall back to any nested Functional backbone in source_model.
            source_layer = next(
                (l for l in source_model.layers if isinstance(l, tf.keras.Model)),
                None,
            )

        if isinstance(target_layer, tf.keras.Model) and isinstance(source_layer, tf.keras.Model):
            sub_loaded, sub_skipped = _copy_matching_weights(target_layer, source_layer)
            loaded += sub_loaded
            skipped += sub_skipped
            continue

        if not target_layer.weights:
            continue

        if source_layer is None or len(target_layer.weights) != len(source_layer.weights):
            skipped += 1
            continue

        target_shapes = [tuple(w.shape) for w in target_layer.weights]
        source_shapes = [tuple(w.shape) for w in source_layer.weights]
        if target_shapes != source_shapes:
            skipped += 1
            continue

        for target_var, source_var in zip(target_layer.weights, source_layer.weights):
            target_var.assign(source_var)
        loaded += 1

    return loaded, skipped


def load_checkpoint_weights(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    weights_path: str | Path,
    freeze_backbone: bool = False,
) -> None:
    """Load pretrained weights from a ``.keras`` or ``.h5`` checkpoint into ``model``."""
    resolved = _resolve_checkpoint_path(str(weights_path))
    loaded_successfully = False

    # 1. Direct load_weights on the full classifier model.
    try:
        model.load_weights(str(resolved))
        loaded_successfully = True
        print(f"Loaded pretrained checkpoint weights from: {resolved}")
    except Exception:
        pass

    # 2. If it is a .keras zip archive, extract model.weights.h5 and load weights.
    if not loaded_successfully and zipfile.is_zipfile(resolved):
        try:
            with zipfile.ZipFile(resolved, "r") as archive:
                if "model.weights.h5" in archive.namelist():
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        extracted = archive.extract("model.weights.h5", tmp_dir)
                        model.load_weights(extracted)
                        loaded_successfully = True
                        print(f"Loaded pretrained weights from archive: {resolved}")
        except Exception:
            pass

    # 3. Try loading the source model via load_model and copying matching layers.
    if not loaded_successfully:
        try:
            source_model = tf.keras.models.load_model(str(resolved), compile=False)
            loaded_count, skipped_count = _copy_matching_weights(model, source_model)
            if loaded_count > 0:
                loaded_successfully = True
                print(
                    f"Copied pretrained weights for {loaded_count} layers from {resolved} "
                    f"({skipped_count} unmatched layers left at random init)."
                )
        except Exception:
            pass

    # 4. Try Keras-2 style or backbone-only H5 weight loading.
    if not loaded_successfully:
        if _is_keras2_h5(str(resolved)):
            loaded, _ = load_pretrained_weights(backbone, weights_file=str(resolved))
            if loaded:
                loaded_successfully = True
        else:
            try:
                backbone.load_weights(str(resolved))
                loaded_successfully = True
                print(f"Loaded pretrained backbone weights from: {resolved}")
            except Exception:
                pass

    if not loaded_successfully:
        raise RuntimeError(f"Could not load pretrained weights from checkpoint: {resolved}")

    backbone.trainable = not freeze_backbone


def parse_robust_params(config: dict) -> dict:
    """Extract and validate PGD hyperparameters for Equation (3).

    Defaults match Table 3 of Engstrom et al. (2019):
        norm = 'l2', eps = 3.0, step_size = 0.5, steps = 7, random_start = True
    where ``eps`` and ``step_size`` are specified in [0, 1] image space.
    """
    norm = str(
        config.get("robust_norm", config.get("norm", "l2"))
    ).strip().lower()
    if norm not in ("l2", "linf"):
        raise ValueError(f"robust_norm must be 'l2' or 'linf', got {norm!r}")

    default_eps = 3.0 if norm == "l2" else (8.0 / 255.0)
    default_step_size = 0.5 if norm == "l2" else (2.0 / 255.0)

    eps_raw = config.get("robust_eps", config.get("eps", default_eps))
    step_raw = config.get("robust_step_size", config.get("step_size", default_step_size))
    steps_raw = config.get("robust_steps", config.get("pgd_steps", 7))
    random_start_raw = config.get(
        "robust_random_start", config.get("random_start", True)
    )

    eps = float(default_eps if eps_raw is None else eps_raw)
    step_size = float(default_step_size if step_raw is None else step_raw)
    steps = int(7 if steps_raw is None else steps_raw)
    random_start = _is_true(random_start_raw, default=True)

    # For linf in [0, 1] pixel space, values > 1.0 (e.g. 8 meaning 8/255) are
    # normalized to [0, 1] automatically.
    if norm == "linf" and eps > 1.0:
        eps = eps / 255.0
    if norm == "linf" and step_size > 1.0:
        step_size = step_size / 255.0

    if eps < 0.0:
        raise ValueError(f"robust_eps must be non-negative, got {eps}")
    if step_size < 0.0:
        raise ValueError(f"robust_step_size must be non-negative, got {step_size}")
    if steps < 0:
        raise ValueError(f"robust_steps must be non-negative, got {steps}")

    return {
        "norm": norm,
        "eps": eps,
        "step_size": step_size,
        "steps": steps,
        "random_start": random_start,
    }


def generate_pgd_adversarial(
    model: tf.keras.Model,
    x: tf.Tensor,
    y: tf.Tensor,
    norm: str = "l2",
    eps: float = 3.0,
    step_size: float = 0.5,
    steps: int = 7,
    random_start: bool = True,
) -> tf.Tensor:
    """Solve the inner maximization of Equation (3) via PGD:

        x_adv = x + argmax_{delta in Delta} L_theta(x + delta, y)

    Note on preprocessing space:
        ``data.py`` feeds Caffe-preprocessed images into the model (RGB [0, 255]
        flipped to BGR with channel means ``[103.939, 116.779, 123.68]`` subtracted).
        Because channel reversal is an orthogonal isometry for both l2 and linf
        norms and mean subtraction is a constant translation, perturbation budgets
        ``eps`` and ``step_size`` in [0, 1] image space correspond exactly to
        ``255 * eps`` and ``255 * step_size`` in Caffe space with pixel bounds
        ``[-mean_bgr, 255 - mean_bgr]``.
    """
    x = tf.cast(x, tf.float32)
    if steps == 0 or eps == 0.0:
        return x

    eps_caffe = tf.cast(eps * 255.0, x.dtype)
    step_caffe = tf.cast(step_size * 255.0, x.dtype)

    mean_bgr = tf.constant(CAFFE_MEAN_BGR, dtype=x.dtype)
    caffe_min = tf.reshape(-mean_bgr, (1, 1, 1, 3))
    caffe_max = tf.reshape(255.0 - mean_bgr, (1, 1, 1, 3))

    if random_start:
        if norm == "l2":
            noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
            noise_norm = tf.sqrt(
                tf.reduce_sum(tf.square(noise), axis=[1, 2, 3], keepdims=True) + 1e-12
            )
            batch_size = tf.shape(x)[0]
            radii = tf.random.uniform(
                (batch_size, 1, 1, 1), minval=0.0, maxval=eps_caffe, dtype=x.dtype
            )
            delta = (noise / noise_norm) * radii
            x_adv = tf.clip_by_value(x + delta, caffe_min, caffe_max)
        else:
            delta = tf.random.uniform(
                tf.shape(x), minval=-eps_caffe, maxval=eps_caffe, dtype=x.dtype
            )
            x_adv = tf.clip_by_value(x + delta, caffe_min, caffe_max)
    else:
        x_adv = tf.identity(x)

    for _ in range(steps):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_adv)
            predictions = model(x_adv, training=False)
            per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, predictions
            )
            total_loss = tf.reduce_sum(per_sample_loss)

        grad = tape.gradient(total_loss, x_adv)
        grad = tf.stop_gradient(grad)

        if norm == "l2":
            grad_norm = tf.sqrt(
                tf.reduce_sum(tf.square(grad), axis=[1, 2, 3], keepdims=True) + 1e-12
            )
            x_adv = x_adv + step_caffe * (grad / grad_norm)
            delta = x_adv - x
            delta_norm = tf.sqrt(
                tf.reduce_sum(tf.square(delta), axis=[1, 2, 3], keepdims=True) + 1e-12
            )
            factor = tf.minimum(1.0, eps_caffe / delta_norm)
            delta = delta * factor
        else:
            x_adv = x_adv + step_caffe * tf.sign(grad)
            delta = tf.clip_by_value(x_adv - x, -eps_caffe, eps_caffe)

        x_adv = tf.clip_by_value(x + delta, caffe_min, caffe_max)

    return tf.stop_gradient(x_adv)


def make_robust_train_step(
    norm: str = "l2",
    eps: float = 3.0,
    step_size: float = 0.5,
    steps: int = 7,
    random_start: bool = True,
):
    """Create a ``train_step`` function implementing Equation (3)."""

    def train_step(self: tf.keras.Model, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        # Inner maximization: find worst-case perturbed input x_adv = x + delta*
        x_adv = generate_pgd_adversarial(
            self,
            x,
            y,
            norm=norm,
            eps=eps,
            step_size=step_size,
            steps=steps,
            random_start=random_start,
        )

        # Outer minimization: update model parameters theta on (x_adv, y)
        with tf.GradientTape() as tape:
            y_pred = self(x_adv, training=True)
            loss = self.compute_loss(
                x=x_adv, y=y, y_pred=y_pred, sample_weight=sample_weight
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self.compute_metrics(x_adv, y, y_pred, sample_weight=sample_weight)

    return train_step


def build_robust_model(config: dict, num_classes: int) -> tf.keras.Model:
    """Build a ResNet classifier from ``modules/resnet50.py`` with PGD robust training.

    Keeping the returned object as a standard Keras ``Functional`` model with a
    nested ``resnet50`` backbone ensures ``best.keras`` checkpoints remain 100%
    compatible with ``test.py``, ``gradcam.py``, and ``src/attack/pgd/``.
    """
    model_name = str(config.get("model_name", "resnet50")).strip().lower()
    image_size = int(config.get("image_size", 224))
    dropout = float(config.get("dropout", 0.2) or 0.0)
    freeze_backbone = _is_true(config.get("freeze_backbone", False), default=False)
    weight_decay = float(config.get("weight_decay", 0.0001) or 0.0)

    if model_name == "resnet50":
        blocks = DEFAULT_BLOCKS
        backbone_name = "resnet50"
    elif model_name == "resnet101":
        blocks = RESNET101_BLOCKS
        backbone_name = "resnet101"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pretrained_mode, checkpoint_path = _parse_pretrained_setting(
        config.get("pretrained", True)
    )
    backbone_weights = "imagenet" if pretrained_mode == "imagenet" else None
    input_shape = (image_size, image_size, 3)

    backbone = ResNet50(
        include_top=False,
        weights=backbone_weights,
        input_shape=input_shape,
        pooling="avg",
        blocks=blocks,
        name=backbone_name,
    )

    if freeze_backbone:
        backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = backbone(inputs, training=False if freeze_backbone else None)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=(
            tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
        ),
    )(x)

    model = tf.keras.Model(inputs, outputs)

    if pretrained_mode == "checkpoint" and checkpoint_path is not None:
        load_checkpoint_weights(
            model,
            backbone,
            checkpoint_path,
            freeze_backbone=freeze_backbone,
        )

    robust_params = parse_robust_params(config)
    print(
        "Robust training enabled (Equation 3 PGD): "
        f"norm={robust_params['norm']}, eps={robust_params['eps']:.5g}, "
        f"step_size={robust_params['step_size']:.5g}, steps={robust_params['steps']}, "
        f"random_start={robust_params['random_start']}"
    )

    model.robust_config = robust_params
    model.generate_adversarial_batch = lambda x_batch, y_batch: generate_pgd_adversarial(
        model,
        x_batch,
        y_batch,
        **robust_params,
    )
    model.train_step = types.MethodType(
        make_robust_train_step(**robust_params),
        model,
    )
    return model


class RobustModel:
    """Factory/wrapper class for building an adversarially robust ResNet model."""

    def __new__(cls, config: dict, num_classes: int) -> tf.keras.Model:
        return build_robust_model(config, num_classes)
