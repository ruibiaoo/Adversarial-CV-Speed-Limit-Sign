"""Subcategory 3 (label / mixing regularisation): Mixup (Zhang et al., 2018).

Each training image is blended with another image from the same batch:
x = lam * x_i + (1 - lam) * x_j, and the one-hot targets and class weights are
blended the same way, with lam ~ Beta(alpha, alpha) drawn per image. Applied
after the Phase 1 albumentations, before preprocess_input.

Why Mixup for this project (over label smoothing and CutMix):
- Mixup makes the model behave close to linearly between training images, so
  the decision boundary sits further from each image. Zhang et al. report
  better robustness to FGSM / I-FGSM, which are PGD-type attacks. Label smoothing
  only rescales the targets. Its apparent PGD robustness is mostly gradient
  masking (PRD section 10), so it does not hold against a stronger or adaptive
  attacker.
- CutMix pastes a rectangle from another sign and weights the label by area.
  The class here depends only on the digits, so if the patch covers the digits
  the area-weighted label is wrong. Mixup blends the whole image, so both
  signs' digits stay visible in proportion to their label weights.
- It adds no training cost, unlike adversarial training.

Grid (chosen on clean val macro F1): alpha in {0.1, 0.2, 0.4}.

  python modules/augmentation-based/mixup.py
  python modules/augmentation-based/mixup.py --alpha 0.2 --epochs 2   # quick test
"""
import argparse

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

import _trainer as T

ALPHAS = (0.1, 0.2, 0.4)


def sample_beta(n, alpha):
    """Beta(alpha, alpha) via two Gamma draws (tf.random has no Beta)."""
    g1 = tf.random.gamma([n], alpha)
    g2 = tf.random.gamma([n], alpha)
    return g1 / (g1 + g2 + 1e-12)


def mixup(x, y_onehot, weights, alpha):
    """Mix each image with a random partner in the batch. x: float 0-255, BxHxWxC."""
    n = tf.shape(x)[0]
    lam = sample_beta(n, alpha)
    perm = tf.random.shuffle(tf.range(n))
    lx = lam[:, None, None, None]
    ly = lam[:, None]
    x = lx * x + (1.0 - lx) * tf.gather(x, perm)
    y_onehot = ly * y_onehot + (1.0 - ly) * tf.gather(y_onehot, perm)
    weights = lam * weights + (1.0 - lam) * tf.gather(weights, perm)
    return x, y_onehot, weights


def make_step_for(params):
    alpha = float(params["alpha"])

    def make_step(model, optimizer, class_weight, num_classes):
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        @tf.function
        def train_step(x, y):
            x, y_soft, w = mixup(x, tf.one_hot(y, num_classes), tf.gather(class_weight, y), alpha)
            with tf.GradientTape() as tape:
                probs = model(preprocess_input(x), training=True)
                loss = loss_fn(y_soft, probs, sample_weight=w)
                loss += tf.add_n(model.losses) if model.losses else 0.0
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        return train_step

    return make_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--epochs", type=int, help="override Phase 1 epochs (quick test)")
    args = parser.parse_args()

    config = T.load_config()
    if args.epochs:
        config["epochs"] = args.epochs
    T.setup_gpu()
    T.train_grid("mixup", [{"alpha": a} for a in args.alpha], make_step_for, config)


if __name__ == "__main__":
    main()
