"""Augmentation-based defence (noise / corruption): random erasing (Zhong et al., 2020).

With probability P, one random rectangle in each training image is replaced
with uniform random pixel values. This runs after the Phase 1 albumentations
and before preprocess_input.

Why random erasing (attack-agnostic, PRD_defences.md):
- It teaches the model not to rely on any single region of the sign, so a
  locally corrupted or occluded input is still classified from what remains.
- Real signs get stickers, dirt, graffiti and partial occlusion, so this is a
  generic, physically motivated corruption, not a model of any attack.
- Gaussian noise augmentation only covers dense, small-amplitude noise.
  Random erasing adds the sparse, large-amplitude case.
- The maximum erased area is capped at 0.3 of the image, because a larger
  rectangle can cover the whole digit and make the label wrong.

Grid (Zhong et al. ranges, chosen on clean val macro F1): maximum erased area
in {0.1, 0.2, 0.3}, P = 0.5, area at least 0.02, aspect ratio 0.3 to 3.3.
Only train/ and clean val/ are read; no attacked image or attack setting is used.

  python modules/augmentation-based/random_erasing.py
  python modules/augmentation-based/random_erasing.py --max-area 0.2 --epochs 2   # quick test
"""
import argparse

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

import _trainer as T

PROB = 0.5
MIN_AREA = 0.02
MAX_AREAS = (0.1, 0.2, 0.3)
ASPECT = (0.3, 3.3)


def random_erasing(x, prob=PROB, min_area=MIN_AREA, max_area=0.2, aspect=ASPECT):
    """Erase one random rectangle per image (with probability prob). x: float 0-255, BxHxWxC."""
    shape = tf.shape(x)
    b, h, w = shape[0], shape[1], shape[2]
    hf, wf = tf.cast(h, tf.float32), tf.cast(w, tf.float32)

    area = tf.random.uniform([b], min_area, max_area) * hf * wf
    log_r = tf.random.uniform([b], tf.math.log(aspect[0]), tf.math.log(aspect[1]))
    ratio = tf.exp(log_r)
    eh = tf.minimum(tf.sqrt(area * ratio), hf)
    ew = tf.minimum(tf.sqrt(area / ratio), wf)
    y0 = tf.random.uniform([b]) * (hf - eh)
    x0 = tf.random.uniform([b]) * (wf - ew)

    rows = tf.cast(tf.range(h), tf.float32)[None, :, None]
    cols = tf.cast(tf.range(w), tf.float32)[None, None, :]
    inside = ((rows >= y0[:, None, None]) & (rows < (y0 + eh)[:, None, None])
              & (cols >= x0[:, None, None]) & (cols < (x0 + ew)[:, None, None]))
    apply = tf.random.uniform([b]) < prob
    mask = tf.cast(inside & apply[:, None, None], x.dtype)[..., None]  # BxHxWx1
    noise = tf.random.uniform(shape, 0.0, 255.0)
    return x * (1.0 - mask) + noise * mask


def make_step_for(params):
    max_area = float(params["max_area"])

    def make_step(model, optimizer, class_weight, num_classes):
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        @tf.function
        def train_step(x, y):
            x = random_erasing(x, max_area=max_area)
            w = tf.gather(class_weight, y)
            with tf.GradientTape() as tape:
                probs = model(preprocess_input(x), training=True)
                loss = loss_fn(tf.one_hot(y, num_classes), probs, sample_weight=w)
                loss += tf.add_n(model.losses) if model.losses else 0.0
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        return train_step

    return make_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-area", type=float, nargs="+", default=list(MAX_AREAS))
    parser.add_argument("--epochs", type=int, help="override Phase 1 epochs (quick test)")
    args = parser.parse_args()

    config = T.load_config()
    if args.epochs:
        config["epochs"] = args.epochs
    T.setup_gpu()
    T.train_grid("random_erasing", [{"max_area": a} for a in args.max_area], make_step_for, config)


if __name__ == "__main__":
    main()
