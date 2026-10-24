"""ResNet classifier with a gradient-reversal domain-classification head."""

from pathlib import Path

import tensorflow as tf

from .gradient_reversal import GradientReversal
from .robustmodel import (
    _parse_pretrained_setting,
    _resolve_checkpoint_path,
    load_checkpoint_weights,
)


def _is_true(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _nested_layers(model):
    for layer in model.layers:
        yield layer
        if isinstance(layer, tf.keras.Model):
            yield from _nested_layers(layer)


def _copy_classification_head(checkpoint_path, class_head):
    """Copy the old classifier's final Dense weights when its shape matches."""
    resolved = _resolve_checkpoint_path(str(checkpoint_path))
    try:
        source = tf.keras.models.load_model(str(resolved), compile=False)
    except Exception as exc:
        raise RuntimeError(
            "Could not inspect the pretrained classifier checkpoint to transfer "
            "its final classification head. Use a loadable .keras model checkpoint."
        ) from exc

    target_shapes = [tuple(weight.shape) for weight in class_head.weights]
    for layer in _nested_layers(source):
        if isinstance(layer, tf.keras.layers.Dense):
            source_shapes = [tuple(weight.shape) for weight in layer.weights]
            if source_shapes == target_shapes:
                class_head.set_weights(layer.get_weights())
                print(f"Transferred classifier head weights from: {resolved}")
                return

    raise ValueError(
        f"No Dense classifier head with shapes {target_shapes} was found in "
        f"checkpoint {resolved}. The checkpoint must have the same number of classes."
    )


def build_grl_model(config, num_classes, num_domains=4):
    """Build shared ResNet50 features and sign/domain output heads.

    The pretrained checkpoint initializes the ResNet backbone and, when its
    output shape matches, the existing sign-classification head. The domain
    head is always newly initialized.
    """
    model_name = str(config.get("model_name", "resnet50")).strip().lower()
    image_size = int(config.get("image_size", 224))
    dropout_rate = float(config.get("dropout", 0.2) or 0.0)
    freeze_backbone = _is_true(config.get("freeze_backbone", False))
    weight_decay = float(config.get("weight_decay", 0.0001) or 0.0)
    grl_config = config.get("grl", {}) or {}
    if num_domains != 4:
        raise ValueError(
            "The combined dataset currently defines exactly four domains "
            "(clean, PGD, Pixle, and PGD+Pixle). Set grl.num_domains to 4."
        )
    reversal_scale = float(grl_config.get("lambda", 1.0))
    domain_hidden_units = int(grl_config.get("domain_hidden_units", 256))
    domain_dropout = float(grl_config.get("domain_dropout", dropout_rate))

    pretrained_mode, checkpoint_path = _parse_pretrained_setting(
        config.get("pretrained", True)
    )
    weights = "imagenet" if pretrained_mode == "imagenet" else None
    input_shape = (image_size, image_size, 3)

    if model_name == "resnet50":
        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )
    elif model_name == "resnet101":
        backbone = tf.keras.applications.ResNet101(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    backbone.trainable = not freeze_backbone
    inputs = tf.keras.Input(shape=input_shape, name="image")
    features = backbone(inputs, training=False if freeze_backbone else None)
    if dropout_rate > 0:
        features = tf.keras.layers.Dropout(dropout_rate, name="shared_dropout")(
            features
        )

    regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    class_output = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizer,
        name="class_output",
    )(features)

    reversed_features = GradientReversal(
        scale=reversal_scale, name="gradient_reversal"
    )(features)
    domain_features = tf.keras.layers.Dense(
        domain_hidden_units,
        activation="relu",
        name="domain_hidden",
    )(reversed_features)
    if domain_dropout > 0:
        domain_features = tf.keras.layers.Dropout(
            domain_dropout, name="domain_dropout"
        )(domain_features)
    domain_output = tf.keras.layers.Dense(
        num_domains,
        activation="softmax",
        name="domain_output",
    )(domain_features)

    model = tf.keras.Model(
        inputs,
        {"class_output": class_output, "domain_output": domain_output},
        name="resnet50_grl",
    )

    if pretrained_mode == "checkpoint" and checkpoint_path is not None:
        load_checkpoint_weights(
            model,
            backbone,
            checkpoint_path,
            freeze_backbone=freeze_backbone,
        )
        _copy_classification_head(
            Path(checkpoint_path), model.get_layer("class_output")
        )

    return model


def classification_model(model):
    """Return a standalone sign-classification model from the two-head GRL model."""
    output = model.output
    if not isinstance(output, dict) or "class_output" not in output:
        raise ValueError("Expected a GRL model with a 'class_output' output.")
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=output["class_output"],
        name="grl_sign_classifier",
    )
