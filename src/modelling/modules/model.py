"""Build resnet model, change model_name in config.yaml to try others."""
import tensorflow as tf


def _is_true(value, default=False):
    """Robust bool check. bool('false') is True in Python, so handle strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def build_model(config, num_classes):
    """Make a resnet classifier with imagenet weights and a new head."""
    if _is_true(config.get("robust", False), default=False):
        from .robustmodel import build_robust_model

        return build_robust_model(config, num_classes)

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
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None,
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

