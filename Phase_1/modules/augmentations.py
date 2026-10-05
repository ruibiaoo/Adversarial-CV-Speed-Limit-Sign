"""Build the on-the-fly augmentations, same values as old YOLO config."""
import albumentations as A


def _is_true(value, default=True):
    """Robust bool check. bool('false') is True in Python, so handle strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def build_augmenter(config, image_size=224):
    """Make an albumentations pipeline from the yaml values.

    Only used on train split. Val just uses resize + preprocess.
    This is best-effort, not bit-identical to Ultralytics.
    Order is: decode -> resize to image_size -> albumentations -> resnet preprocess.
    """
    if not _is_true(config.get("augment", True), default=True):
        return None

    hsv_h = float(config.get("hsv_h", 0) or 0)
    hsv_s = float(config.get("hsv_s", 0) or 0)
    hsv_v = float(config.get("hsv_v", 0) or 0)
    degrees = float(config.get("degrees", 0) or 0)
    translate = float(config.get("translate", 0) or 0)
    scale = float(config.get("scale", 0) or 0)
    shear = float(config.get("shear", 0) or 0)
    perspective = float(config.get("perspective", 0) or 0)
    bgr = float(config.get("bgr", 0) or 0)
    grayscale = float(config.get("grayscale", 0) or 0)
    erasing = float(config.get("erasing", 0) or 0)

    transforms = []

    # hsv aug, yolo uses 0-1 fractions
    if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=int(180 * hsv_h),
                sat_shift_limit=int(255 * hsv_s),
                val_shift_limit=int(255 * hsv_v),
                p=1.0,
            )
        )

    # rotation + shift + scale + shear in one affine
    if degrees > 0 or translate > 0 or scale > 0 or shear > 0:
        scale_limit = max(0.0, scale)
        shear_limit = (-shear, shear) if shear > 0 else 0
        transforms.append(
            A.Affine(
                rotate=(-degrees, degrees) if degrees > 0 else 0,
                translate_percent=max(0.0, translate),
                scale=(1 - scale_limit, 1 + scale_limit),
                shear=shear_limit,
                p=1.0,
            )
        )

    # yolo perspective is tiny (0.0005), so keep this almost a no-op for parity
    if perspective > 0:
        transforms.append(A.Perspective(scale=(0.0, 0.01), keep_size=True, p=0.5))

    # bgr shuffle with same prob
    if bgr > 0:
        transforms.append(A.ChannelShuffle(p=bgr))

    # grayscale with a low probability so colour still matters
    if grayscale > 0:
        transforms.append(A.ToGray(p=grayscale))

    # random erasing with same prob
    if erasing > 0:
        crop = max(8, int(image_size * 0.2))
        transforms.append(
            A.CoarseDropout(
                max_holes=1,
                max_height=crop,
                max_width=crop,
                p=erasing,
            )
        )

    return A.Compose(transforms)
