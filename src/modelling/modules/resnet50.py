"""ResNet-50, written out one layer at a time so it stays easy to tweak.

Reference
---------
Ported from the Keras 2 reference implementation:
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
    paper: "Deep Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385

The port targets the Keras 3 API shipped with TensorFlow 2.16+ (``tf.keras``),
which is what this repo trains with, so this file drops in wherever
``tf.keras.applications.ResNet50`` is used today.

Layer names follow ``keras.applications.ResNet50`` (``conv1_conv``,
``conv2_block1_1_conv``, ``conv5_block3_out``, ...), for three practical
reasons:

* ``gradcam.py`` hooks the layer named ``conv5_block3_out``
* the checkpoints already in ``runs/`` were trained under those names
* the official model can be built next to this one and compared layer by layer

The published ImageNet weight files hold the same layer names, so
:func:`load_pretrained_weights` copies weights layer by layer, by name. It
reports every layer it cannot match instead of raising, which is what keeps
``weights="imagenet"`` working after you change the architecture: the backbone
gets its pretrained weights and your new layers keep their random init.

The network, top to bottom (default 224x224x3 input)
----------------------------------------------------
    input
    conv1_pad        zero padding, 3 on every side
    conv1_conv       7x7 conv, 64 filters, stride 2                 112x112x64
    conv1_bn         batch norm
    conv1_relu
    pool1_pad        zero padding, 1 on every side
    pool1_pool       3x3 max pool, stride 2                          56x56x64
    conv2            3 blocks, stride 1,  64 middle filters,  256 output filters
    conv3            4 blocks, stride 2, 128 middle filters,  512 output filters
    conv4            6 blocks, stride 2, 256 middle filters, 1024 output filters
    conv5            3 blocks, stride 2, 512 middle filters, 2048 output filters
    avg_pool         global average pooling                               2048
    predictions      dense(classes), softmax

One block is: 1x1 conv (squeeze) -> 3x3 conv (the work) -> 1x1 conv (expand),
then the shortcut is added and the sum goes through relu. The first block of a
stage also puts a 1x1 conv with stride 2 on the shortcut, because there the
input and the main path differ in shape.

Where to make small mods
------------------------
* another depth:   ``ResNet50(blocks=(3, 4, 23, 3))`` is ResNet-101
* other widths:    ``ResNet50(filters=(64, 128, 256, 512))``
* another block:   write your own and pass ``block_fn=my_block``
* a new layer inside every block: edit :func:`bottleneck_block`, one place and
  all 16 blocks follow
* freeze or unfreeze after loading: set ``layer.trainable``, then recompile

Usage
-----
    import tensorflow as tf
    from modules.resnet50 import ResNet50, preprocess_input

    # ImageNet backbone, then a new head for our 7 speed limit classes. Calling
    # the backbone as a layer keeps it nested inside the model, which is the
    # shape gradcam.py expects when it looks for the last conv block.
    inputs = tf.keras.Input(shape=(256, 256, 3))
    backbone = ResNet50(include_top=False, weights="imagenet",
                        input_shape=(256, 256, 3), pooling="avg")
    x = backbone(inputs)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(7, activation="softmax", name="head")(x)
    model = tf.keras.Model(inputs, outputs, name="resnet50_speed_limit")

    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=30)

    # Images go in the way ResNet was trained: BGR, channel means subtracted.
    batch = preprocess_input(tf.cast(images, tf.float32))
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# The ImageNet weight files published for keras.applications.ResNet50.
# Same URL and same hash, so Keras caches them once in ~/.keras/models and the
# official model and this one share the download.
# ---------------------------------------------------------------------------
BASE_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
)
WEIGHTS_FILES = {
    # include_top: (file name, hash)
    True: (
        "resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        "2cb95161c43110f7111970584f804107",
    ),
    False: (
        "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "4d473c1dd8becc155b73f8504c6f6626",
    ),
}

DEFAULT_BLOCKS = (3, 4, 6, 3)  # blocks in conv2, conv3, conv4, conv5
DEFAULT_FILTERS = (64, 128, 256, 512)  # middle (3x3) filters per stage
BN_EPSILON = 1.001e-5  # what keras.applications.ResNet50 uses
DEFAULT_KERNEL_INITIALIZER = "he_normal"

# The published file calls the classifier "probs" (very old copies say
# "fc1000"), this file and keras.applications call it "predictions".
FILE_NAME_ALIASES = {"predictions": ("probs", "fc1000")}


def preprocess_input(x, data_format=None):
    """RGB -> BGR, subtract the ImageNet channel means, no rescaling.

    This is the preprocessing ResNet was trained with, the same "caffe" mode
    ``keras.applications.resnet.preprocess_input`` uses. Images come in as
    0-255 RGB, so do not divide by 255 yourself.
    """
    return tf.keras.applications.imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="caffe"
    )


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------
def bottleneck_block(
    x,
    filters,
    stride=1,
    use_projection=False,
    base_name=None,
    bn_axis=3,
    bn_epsilon=BN_EPSILON,
    kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
):
    """One bottleneck block, the block ResNet-50 is built from.

    This is the reference's ``identity_block`` and ``conv_block`` in one
    function: ``use_projection=False`` is the identity block, ``True`` is the
    block that also transforms the shortcut.

    Args:
        x: input tensor.
        filters: filters in the middle 3x3 conv. The two outer 1x1 convs use
            ``4 * filters``, which is what makes the block a bottleneck.
        stride: stride of the first 1x1 conv. Anything but 1 downsamples, and
            then the shortcut has to be a strided 1x1 conv too.
        use_projection: put a 1x1 conv (plus batch norm) on the shortcut.
            The first block of every stage does this.
        base_name: name prefix of the layers, for example ``conv3_block2``.
        bn_axis: channel axis, 3 for channels_last, 1 for channels_first.
        bn_epsilon: batch norm epsilon.
        kernel_initializer: initializer of the conv kernels.

    Returns:
        Output tensor of the block.
    """
    output_filters = filters * 4
    base_name = base_name or f"block_{filters}"

    # Shortcut branch: identity when the shapes line up, else a strided 1x1.
    if use_projection:
        shortcut = layers.Conv2D(
            output_filters,
            1,
            strides=stride,
            kernel_initializer=kernel_initializer,
            name=f"{base_name}_0_conv",
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=bn_epsilon, name=f"{base_name}_0_bn"
        )(shortcut)
    else:
        shortcut = x

    # Main branch: squeeze with 1x1, work with 3x3, expand back with 1x1.
    x = layers.Conv2D(
        filters,
        1,
        strides=stride,
        kernel_initializer=kernel_initializer,
        name=f"{base_name}_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=bn_epsilon, name=f"{base_name}_1_bn"
    )(x)
    x = layers.Activation("relu", name=f"{base_name}_1_relu")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer=kernel_initializer,
        name=f"{base_name}_2_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=bn_epsilon, name=f"{base_name}_2_bn"
    )(x)
    x = layers.Activation("relu", name=f"{base_name}_2_relu")(x)

    x = layers.Conv2D(
        output_filters,
        1,
        kernel_initializer=kernel_initializer,
        name=f"{base_name}_3_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=bn_epsilon, name=f"{base_name}_3_bn"
    )(x)

    # The skip connection: add, then relu. This sum is the whole point.
    x = layers.Add(name=f"{base_name}_add")([shortcut, x])
    x = layers.Activation("relu", name=f"{base_name}_out")(x)
    return x


def residual_stage(
    x,
    filters,
    blocks,
    stride,
    stage,
    block_fn=bottleneck_block,
    **block_kwargs,
):
    """A stack of blocks that share one filter count.

    The first block downsamples when ``stride`` is 2 and therefore needs the
    projection shortcut. The blocks after it are identity blocks that keep the
    shapes unchanged.

    Args:
        x: input tensor.
        filters: middle filters of every block in this stage.
        blocks: how many blocks, so 3 for conv2 of ResNet-50.
        stride: stride of the first block, 1 for conv2, 2 for the later stages.
        stage: stage number, 2 to 5, used to name the layers.
        block_fn: block builder, swap in your own if you want something else.
        **block_kwargs: passed straight to ``block_fn``.
    """
    x = block_fn(
        x,
        filters,
        stride=stride,
        use_projection=True,
        base_name=f"conv{stage}_block1",
        **block_kwargs,
    )
    for index in range(2, blocks + 1):
        x = block_fn(
            x,
            filters,
            stride=1,
            use_projection=False,
            base_name=f"conv{stage}_block{index}",
            **block_kwargs,
        )
    return x


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def ResNet50(
    include_top: bool = True,
    weights: str | None = "imagenet",
    input_tensor=None,
    input_shape: tuple | None = None,
    pooling: str | None = None,
    classes: int = 1000,
    classifier_activation: str | None = "softmax",
    dropout: float = 0.0,
    blocks: tuple = DEFAULT_BLOCKS,
    filters: tuple = DEFAULT_FILTERS,
    bn_epsilon: float = BN_EPSILON,
    kernel_initializer: str = DEFAULT_KERNEL_INITIALIZER,
    block_fn=bottleneck_block,
    name: str = "resnet50",
):
    """Build a ResNet-50.

    Args:
        include_top: keep the ImageNet classifier on top. Set False for
            transfer learning and put your own head on the output.
        weights: ``"imagenet"``, ``None`` for random init, or a path to a
            weights file. A path to an ``.h5`` file in Keras 2 layout is read
            by :func:`load_pretrained_weights`, anything else goes through
            ``model.load_weights``.
        input_tensor: optional Keras tensor to use as image input.
        input_shape: ``(height, width, 3)``. Any size works, the head is a
            global average pool. Sizes other than 224 with
            ``weights="imagenet"`` and ``include_top=True`` still load, the
            accuracy is just no longer what ImageNet measured.
        pooling: with ``include_top=False``: ``None`` returns the 4D feature
            map, ``"avg"`` or ``"max"`` pool it down to a vector.
        classes: classes of the top layer, ignored when ``include_top=False``.
            With ``weights="imagenet"`` and a class count other than 1000 the
            pretrained head is skipped and left random, the backbone still
            loads.
        classifier_activation: activation of the top layer, ``None`` for logits.
        dropout: dropout rate in front of the top layer, 0 disables it. Only
            used when ``include_top=True``.
        blocks: blocks per stage, ``(3, 4, 6, 3)`` for ResNet-50,
            ``(3, 4, 23, 3)`` for ResNet-101.
        filters: middle filters per stage.
        bn_epsilon: batch norm epsilon. 1.001e-5 matches the pretrained
            weights, changing it changes the output slightly.
        kernel_initializer: conv kernel initializer. Only matters for layers
            that do not get pretrained weights.
        block_fn: block builder, see :func:`bottleneck_block`.
        name: model name.

    Returns:
        A Keras ``Model``.
    """
    if weights not in (None, "imagenet") and not tf.io.gfile.exists(weights):
        raise ValueError(
            'weights must be None, "imagenet" or a path to a weights file, '
            f"got weights={weights!r}"
        )
    if len(blocks) != 4 or len(filters) != 4:
        raise ValueError(
            "blocks and filters need four entries, one per stage, got "
            f"blocks={blocks}, filters={filters}"
        )

    input_shape = _obtain_input_shape(input_shape)
    bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        img_input = input_tensor

    # Stem: 7x7 conv over the padded image, then a stride 2 max pool.
    x = layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(img_input)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding="valid",
        kernel_initializer=kernel_initializer,
        name="conv1_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=bn_epsilon, name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.ZeroPadding2D(padding=(1, 1), name="pool1_pad")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name="pool1_pool")(x)

    # The four stages. Later stages widen the channels and halve the grid.
    stage_layout = (
        # stage, blocks, middle filters, stride of the first block
        (2, blocks[0], filters[0], 1),
        (3, blocks[1], filters[1], 2),
        (4, blocks[2], filters[2], 2),
        (5, blocks[3], filters[3], 2),
    )
    block_kwargs = {
        "bn_axis": bn_axis,
        "bn_epsilon": bn_epsilon,
        "kernel_initializer": kernel_initializer,
    }
    for stage, block_count, stage_filters, stride in stage_layout:
        x = residual_stage(
            x,
            stage_filters,
            block_count,
            stride,
            stage,
            block_fn=block_fn,
            **block_kwargs,
        )

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout:
            x = layers.Dropout(dropout, name="dropout")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    elif pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    inputs = img_input
    if input_tensor is not None:
        # Model inputs have to be the source Input layers, not the tensor the
        # caller handed us. keras.utils.get_source_inputs does that lookup.
        get_sources = getattr(tf.keras.utils, "get_source_inputs", None)
        inputs = get_sources(input_tensor) if get_sources is not None else input_tensor

    model = tf.keras.Model(inputs, x, name=name)

    if weights == "imagenet":
        load_pretrained_weights(model, include_top=include_top)
    elif weights is not None:
        if _is_keras2_h5(weights):
            load_pretrained_weights(model, weights_file=weights)
        else:
            model.load_weights(weights)

    return model


def _obtain_input_shape(
    input_shape: tuple | None = None, default_size: int = 224, min_size: int = 32
) -> tuple:
    """Default the shape to 224x224x3 and check the channels and the size."""
    data_format = tf.keras.backend.image_data_format()
    channels_last = data_format == "channels_last"
    if input_shape is None:
        if channels_last:
            return (default_size, default_size, 3)
        return (3, default_size, default_size)

    if len(input_shape) != 3:
        raise ValueError(
            f"input_shape must be (height, width, channels), got {input_shape}"
        )
    if channels_last:
        height, width, channels = input_shape
    else:
        channels, height, width = input_shape

    if channels != 3:
        raise ValueError(f"input_shape needs 3 channels, got {input_shape}")
    if min(height, width) < min_size:
        raise ValueError(
            f"input_shape is too small, each side needs at least {min_size} "
            f"pixels, got {input_shape}"
        )
    return tuple(input_shape)


# ---------------------------------------------------------------------------
# Pretrained weights
# ---------------------------------------------------------------------------
def download_imagenet_weights(include_top=True, cache_dir=None):
    """Download the published ResNet-50 ImageNet weights, cached by Keras.

    Returns the path of the file, usually ``~/.keras/models/<file>.h5``.
    """
    file_name, file_hash = WEIGHTS_FILES[bool(include_top)]
    return tf.keras.utils.get_file(
        fname=file_name,
        origin=f"{BASE_WEIGHTS_PATH}{file_name}",
        file_hash=file_hash,
        cache_subdir="models",
        cache_dir=cache_dir,
    )


def load_pretrained_weights(
    model,
    include_top: bool = True,
    weights_file: str | None = None,
    cache_dir: str | None = None,
    verbose: bool = True,
):
    """Copy ImageNet weights into ``model``, one layer at a time.

    Every layer that holds weights is looked up in the file by name, and its
    variables are matched by name inside that layer. A layer with no match, or
    with a shape that does not fit, is skipped and reported instead of raising.
    That is what makes ``weights="imagenet"`` survive architecture changes: the
    backbone still gets pretrained weights and a new head keeps its init.

    Args:
        model: the model to fill, built by this file.
        include_top: which published file to read, the one with or without the
            ImageNet classifier.
        weights_file: read this file instead of the published one.
        cache_dir: where Keras should cache downloads.
        verbose: print how many layers were loaded and which were skipped.

    Returns:
        ``(loaded, skipped)``, two lists of layer names.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "Reading the published ResNet-50 weights needs h5py: pip install h5py"
        ) from error

    if weights_file is None:
        weights_file = download_imagenet_weights(
            include_top=include_top, cache_dir=cache_dir
        )

    loaded, skipped = [], []
    with h5py.File(weights_file, "r") as handle:
        root = handle["model_weights"] if "model_weights" in handle else handle
        groups = {name: root[name] for name in _read_h5_attribute(root, "layer_names")}

        for layer in model.layers:
            variables = {variable.name: variable for variable in layer.weights}
            if not variables:
                continue

            group = _find_group(groups, layer.name)
            if group is None:
                skipped.append(layer.name)
                continue

            saved = {
                _variable_name(name): group[name][()]
                for name in _read_h5_attribute(group, "weight_names")
            }
            if not _assign_by_name(variables, saved) and not _assign_in_order(
                layer.weights, saved
            ):
                skipped.append(layer.name)
                continue
            loaded.append(layer.name)

    if verbose:
        print(
            f"ResNet-50 pretrained weights: loaded {len(loaded)} of "
            f"{len(loaded) + len(skipped)} layers from {weights_file}"
        )
        if skipped:
            print("  no match in the file, left at random init: " + ", ".join(skipped))
    return loaded, skipped


def _find_group(groups, layer_name):
    """Find the file group for a model layer, honouring the alias list."""
    for candidate in (layer_name, *FILE_NAME_ALIASES.get(layer_name, ())):
        if candidate in groups:
            return groups[candidate]
    return None


def _assign_by_name(variables, saved):
    """Assign weights matched by variable name: kernel, bias, gamma, and so on."""
    if set(variables) != set(saved):
        return False
    if any(tuple(variables[key].shape) != value.shape for key, value in saved.items()):
        return False
    for key, value in saved.items():
        variables[key].assign(value)
    return True


def _assign_in_order(variables, saved):
    """Fall back to positional assignment when the variable names do not line up."""
    values = list(saved.values())
    if len(values) != len(variables):
        return False
    if any(tuple(variable.shape) != value.shape for variable, value in zip(variables, values)):
        return False
    for variable, value in zip(variables, values):
        variable.assign(value)
    return True


def _variable_name(saved_name):
    """Last piece of a saved variable path: 'conv1_conv/kernel:0' -> 'kernel'."""
    return str(saved_name).split("/")[-1].split(":")[0]


def _read_h5_attribute(group, name):
    """Read a text attribute, including the chunked 'name0', 'name1' form."""
    keys = sorted(key for key in group.attrs if key == name or key[: len(name)] == name)
    values = []
    for key in keys:
        attribute = group.attrs[key]
        items = attribute if hasattr(attribute, "__len__") else [attribute]
        for item in items:
            values.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return values


def _is_keras2_h5(path):
    """True when the file is an h5 that stores layers under their names."""
    if not str(path).endswith((".h5", ".hdf5")):
        return False
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            root = handle["model_weights"] if "model_weights" in handle else handle
            return any(str(key).startswith("layer_names") for key in root.attrs)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Small self check: build it, and optionally load the ImageNet weights
# ---------------------------------------------------------------------------
def main():
    """Build the model and check it against keras.applications.ResNet50.

        python -m modules.resnet50                 # build only
        python -m modules.resnet50 --check         # build, load weights, compare
        python -m modules.resnet50 --check --image path_or_url
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the ResNet-50 in this file and check it"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="load the published ImageNet weights and compare with the official model",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="image path or URL to classify, only used with --check",
    )
    args = parser.parse_args()

    image_size = 224  # the size the ImageNet weights were trained at
    input_shape = (image_size, image_size, 3)

    backbone = ResNet50(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )
    print(f"backbone      {backbone.output_shape}  {backbone.count_params():,} params")
    print(f"  first       {[layer.name for layer in backbone.layers[:6]]}")
    print(f"  last        {[layer.name for layer in backbone.layers[-3:]]}")

    if not args.check:
        return

    model = ResNet50(include_top=True, weights="imagenet", input_shape=input_shape)
    print(f"classifier    {model.output_shape}  {model.count_params():,} params")

    image = args.image or (
        "https://storage.googleapis.com/download.tensorflow.org/example_images/"
        "grace_hopper.jpg"
    )
    if str(image).startswith(("http://", "https://")):
        image = tf.keras.utils.get_file("resnet50_sample_image.jpg", image)
    raw = tf.io.decode_image(
        tf.io.read_file(image), channels=3, expand_animations=False
    )
    raw = tf.image.resize(raw, (image_size, image_size))
    batch = preprocess_input(tf.expand_dims(tf.cast(raw, tf.float32), 0))

    probabilities = model.predict(batch, verbose=0)
    decode = tf.keras.applications.imagenet_utils.decode_predictions
    print(f"this model on {image}:")
    for rank, (code, label, score) in enumerate(decode(probabilities, top=5)[0], 1):
        print(f"  {rank}. {label} ({code}) {score:.4f}")

    official = tf.keras.applications.ResNet50(weights="imagenet", input_shape=input_shape)
    difference = float(np.abs(official.predict(batch, verbose=0) - probabilities).max())
    print(f"max difference from keras.applications.ResNet50: {difference:.3e}")
    if difference > 1e-4:
        raise SystemExit("weights did not load identically to the official model")
    print("same weights, same preprocessing, same predictions.")


if __name__ == "__main__":
    main()
