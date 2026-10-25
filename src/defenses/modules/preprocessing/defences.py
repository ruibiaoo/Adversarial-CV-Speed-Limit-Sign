"""Preprocessing defences: one function per defence, plus one predictor interface.

Every defence runs at test time in front of the same frozen Phase 1 model:

  D1 jpeg_compression    re-encode the input as JPEG at quality q
  D2 gaussian_smoothing  randomized smoothing: n noisy copies (N(0, sigma^2)),
                         majority vote of the model's predictions (Cohen et al., 2019)
  D3 median_smoothing    k x k median filter

Images are uint8 RGB HxWx3 in 0-255, the same format as the dataset folders.
make_predictor() turns any defence into fn(batch_uint8) -> class scores, so the
evaluation treats the four options (including no defence) the same way.

No attack data or attack knowledge is used here: the grids come from general
image-processing ranges. They are not tuned at the moment: test_defences.py
tests one fixed setting per defence.
"""
import cv2
import numpy as np

NO_DEFENCE = "no_defence"
JPEG = "jpeg_compression"
GAUSSIAN = "gaussian_smoothing"
MEDIAN = "median_smoothing"

# Grids, mildest setting first. The strength order is what the tuning rule uses:
# lower quality, larger sigma and larger kernel all remove more image detail.
GRIDS = {
    JPEG: [{"quality": q} for q in (90, 75, 60, 50, 40, 30)],
    GAUSSIAN: [{"sigma": s} for s in (4, 8, 12, 16, 24)],
    MEDIAN: [{"kernel": k} for k in (3, 5, 7)],
}
SMOOTHING_SAMPLES = 32  # n noisy copies per image, fixed a priori (not tuned)
SMOOTHING_SEED = 4266   # same as the project seed, so predictions are reproducible


def jpeg_compress(image, quality):
    """Encode and decode one uint8 RGB image as JPEG at the given quality (1-100)."""
    bgr = cv2.cvtColor(np.ascontiguousarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def median_smooth(image, kernel):
    """Median filter one uint8 RGB image per channel with a k x k window (k odd)."""
    return cv2.medianBlur(np.ascontiguousarray(image, dtype=np.uint8), int(kernel))


def gaussian_noise(images, sigma, rng):
    """images + N(0, sigma^2) noise in pixel units, clipped to 0-255. Returns float32."""
    noisy = images.astype(np.float32) + rng.normal(0.0, float(sigma), images.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 255.0)


def randomized_smoothing_probs(classify, images, sigma, n=SMOOTHING_SAMPLES, seed=SMOOTHING_SEED,
                               chunk=64):
    """Vote share of each class over n noisy copies of each image.

    classify: fn(float32 batch 0-255) -> class scores. images: uint8 BxHxWx3.
    Returns BxC vote frequencies; argmax is the smoothed prediction (no abstain,
    because the project reports accuracy). The seed is fixed per call, so the
    same images always get the same noise.
    """
    rng = np.random.default_rng(seed)
    per_step = max(1, chunk // n)  # images per model call, bounds memory to ~chunk copies
    counts = []
    for start in range(0, len(images), per_step):
        batch = images[start:start + per_step]
        copies = np.repeat(batch, n, axis=0)  # (b*n)xHxWx3, copies of one image are adjacent
        scores = np.asarray(classify(gaussian_noise(copies, sigma, rng)))
        pred = np.argmax(scores, axis=1).reshape(len(batch), n)
        counts += [np.bincount(p, minlength=scores.shape[1]) for p in pred]
    return np.stack(counts).astype(np.float32) / float(n)


def transform_for(name, params):
    """uint8 -> uint8 transform of a deterministic defence (None for no defence)."""
    if name == NO_DEFENCE:
        return None
    if name == JPEG:
        return lambda im: jpeg_compress(im, params["quality"])
    if name == MEDIAN:
        return lambda im: median_smooth(im, params["kernel"])
    raise ValueError(f"{name} has no deterministic transform")


def make_predictor(name, model, params=None):
    """fn(BxHxWx3 images in 0-255, uint8 or float) -> BxC class scores for one defence in front of model."""
    from tensorflow.keras.applications.resnet import preprocess_input

    params = params or {}

    def classify(x):  # float 0-255 -> probabilities
        return model.predict_on_batch(preprocess_input(np.array(x, dtype=np.float32)))

    if name == GAUSSIAN:
        sigma = float(params["sigma"])
        n = int(params.get("samples", SMOOTHING_SAMPLES))
        seed = int(params.get("seed", SMOOTHING_SEED))
        return lambda images: randomized_smoothing_probs(classify, images, sigma, n, seed)

    transform = transform_for(name, params)

    def predict(images):
        if transform is not None:  # the filters work on uint8, so round the resized image first
            images = np.clip(np.rint(images), 0, 255).astype(np.uint8)
            images = np.stack([transform(im) for im in images])
        return np.asarray(classify(images))

    return predict


def _to_uint8(images):
    return np.clip(np.rint(images), 0, 255).astype(np.uint8)


def make_chain_predictor(steps, model, params):
    """fn(BxHxWx3 images in 0-255) -> BxC class scores for several defences applied in order.

    steps: defence names in the order they are applied, e.g. [JPEG, GAUSSIAN, MEDIAN].
    params: {name: params dict}. JPEG and MEDIAN filter the image at their position.
    GAUSSIAN turns the image into n noisy copies at its position; the later steps
    then run on every copy and the model's predictions are majority-voted, as in
    randomized smoothing. Without GAUSSIAN the output is the model's softmax.
    """
    from tensorflow.keras.applications.resnet import preprocess_input

    def classify(x):
        return model.predict_on_batch(preprocess_input(np.array(x, dtype=np.float32)))

    def filtered(images, names):
        for name in names:
            transform = transform_for(name, params[name])
            images = np.stack([transform(im) for im in _to_uint8(images)])
        return images

    if GAUSSIAN not in steps:
        return lambda images: np.asarray(classify(filtered(images, steps)))

    at = steps.index(GAUSSIAN)
    before, after = steps[:at], steps[at + 1:]
    sigma = float(params[GAUSSIAN]["sigma"])
    n = int(params[GAUSSIAN].get("samples", SMOOTHING_SAMPLES))
    seed = int(params[GAUSSIAN].get("seed", SMOOTHING_SEED))

    def predict(images):
        rng = np.random.default_rng(seed)  # fixed per call, as in randomized_smoothing_probs
        counts = []
        for start in range(0, len(images), 2):  # 2 images x n copies per model call
            batch = filtered(images[start:start + 2], before)
            noisy = gaussian_noise(np.repeat(batch, n, axis=0), sigma, rng)
            scores = np.asarray(classify(filtered(noisy, after)))
            pred = np.argmax(scores, axis=1).reshape(len(batch), n)
            counts += [np.bincount(p, minlength=scores.shape[1]) for p in pred]
        return np.stack(counts).astype(np.float32) / float(n)

    return predict
