"""Build resnet model, change model_name in config.yaml to try others."""
import tensorflow as tf


def _is_true(value, default=False):
    """Robust bool check. bool('false') is True in Python, so handle strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def _build_base_classifier(config, num_classes, is_robust=False):
    """Build the ResNet classifier without the ConvAE preprocessor wrapper."""
    if is_robust:
        from .robustmodel import build_robust_model

        # Attach PGD train_step later so it wraps the outer model if ConvAE is enabled.
        return build_robust_model(config, num_classes, attach_step=False)

    from .robustmodel import _parse_pretrained_setting, load_checkpoint_weights

    model_name = str(config.get("model_name", "resnet50")).lower()
    image_size = int(config.get("image_size", 224))
    dropout = float(config.get("dropout", 0.2) or 0.0)
    pretrained_mode, checkpoint_path = _parse_pretrained_setting(
        config.get("pretrained", True)
    )
    freeze_backbone = _is_true(config.get("freeze_backbone", False), default=False)
    weight_decay = float(config.get("weight_decay", 0.0001) or 0.0)

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
    return model


def build_model(config, num_classes):
    """Make a ResNet classifier, optionally preceded by a ConvAE preprocessor."""
    from .convae import (
        is_wrapped_with_convae,
        load_convae,
        parse_convae_config,
        resolve_convae_path,
        wrap_with_convae,
    )
    from .robustmodel import attach_robust_training

    is_robust = _is_true(config.get("robust", False), default=False)
    classifier = _build_base_classifier(config, num_classes, is_robust=is_robust)

    convae_cfg = parse_convae_config(config)
    if convae_cfg["enabled"]:
        convae_path = convae_cfg["model_path"]
        if not convae_path:
            raise ValueError(
                "convae.enabled is true, but convae.model_path was not specified in config."
            )
        if is_wrapped_with_convae(classifier):
            print(
                "Warning: classifier already contains a ConvAE preprocessor; skipping duplicate wrap."
            )
            final_model = classifier
        else:
            resolved_ae = resolve_convae_path(convae_path)
            image_size = int(config.get("image_size", 256))
            convae = load_convae(resolved_ae, img_size=image_size)
            trainable_ae = convae_cfg["trainable"]
            print(
                f"ConvAE preprocessing enabled: loaded {resolved_ae} "
                f"(trainable={trainable_ae})"
            )
            final_model = wrap_with_convae(
                classifier,
                convae,
                trainable=trainable_ae,
            )
    else:
        final_model = classifier

    if is_robust:
        attach_robust_training(final_model, config)

    return final_model


def get_optimizer(config):
    """Make sgd, adam or adamw from config."""
    name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate", 0.001))
    momentum = float(config.get("momentum", 0.9) or 0.0)
    weight_decay = float(config.get("weight_decay", 0.0001) or 0.0)

    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def load_classifier(model_path, image_size=256, num_classes=7):
    """Load model, falling back to architecture reconstruction for Keras 2/3 cross-compatibility."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        pass

    import tempfile
    import zipfile

    model = build_model(
        {"model_name": "resnet50", "image_size": image_size, "pretrained": False},
        num_classes=num_classes,
    )
    if zipfile.is_zipfile(model_path):
        with zipfile.ZipFile(model_path, "r") as z:
            with tempfile.TemporaryDirectory() as tmp_dir:
                extracted_path = z.extract("model.weights.h5", tmp_dir)
                model.load_weights(extracted_path)
    else:
        model.load_weights(model_path)
    return model
