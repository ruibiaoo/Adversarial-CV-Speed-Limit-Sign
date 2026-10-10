"""Subcategory 4 (adversarial augmentation): TRADES (Zhang et al., 2019).

Loss = CE(f(x), y) + beta * KL(f(x) || f(x_adv)). Here x_adv is an L-inf PGD
point that maximises the KL term, made fresh inside the training loop from the
current train batch. It stays in memory and is never saved (PRD section 3).

Why TRADES for this project (over PGD-AT and Fast-AT):
- Selection uses the worst case min(clean, WB, BB) (PRD section 6), so clean
  accuracy counts as much as robust accuracy. TRADES keeps the clean CE term
  and adds a separate robustness term, so beta sets the clean/robust trade-off
  directly. Plain PGD adversarial training only has a coarse clean/adv
  batch ratio.
- The KL term pulls f(x_adv) towards f(x) rather than towards the hard label.
  This tends to give smoother decision boundaries, which also helps against
  attacks that were not trained on (Pixle).
- Fast-AT (one FGSM step) only saves compute. With 865 train images that is not
  needed, and it risks catastrophic overfitting.

Budget fixed a priori from Phase 2's imperceptibility cap, not tuned:
eps = 8/255 L-inf (the Phase 2 PGD budget), step 2/255, 5 steps
(adv_eps / adv_alpha / adv_steps in config.yaml). Grid (chosen on clean val
macro F1): beta in {1, 6}. 6 is the paper default, 1 favours clean accuracy.

  python modules/augmentation-based/trades.py
  python modules/augmentation-based/trades.py --beta 6 --epochs 2   # quick test
"""
import argparse

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

import _trainer as T

BETAS = (1.0, 6.0)


def kl(p, q):
    """Per-sample KL(p || q) for probability vectors."""
    p = tf.clip_by_value(p, 1e-7, 1.0)
    q = tf.clip_by_value(q, 1e-7, 1.0)
    return tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)), axis=-1)


def trades_attack(model, x, eps, alpha, steps):
    """PGD on KL(f(x) || f(x_adv)), L-inf ball of radius eps, pixel space 0-255.

    Starts from x + small Gaussian noise (0.001 in [0, 1] scale), as in the paper.
    """
    p_nat = tf.stop_gradient(model(preprocess_input(x), training=False))
    x_adv = tf.clip_by_value(x + 0.255 * tf.random.normal(tf.shape(x)), 0.0, 255.0)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = tf.reduce_sum(kl(p_nat, model(preprocess_input(x_adv), training=False)))
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)
    return tf.stop_gradient(x_adv)


def make_step_for(params, config):
    beta = float(params["beta"])
    eps = float(config["adv_eps"])
    alpha = float(config["adv_alpha"])
    steps = int(config["adv_steps"])

    def make_step(model, optimizer, class_weight, num_classes):
        @tf.function
        def train_step(x, y):
            x_adv = trades_attack(model, x, eps, alpha, steps)
            w = tf.gather(class_weight, y)
            with tf.GradientTape() as tape:
                p_nat = model(preprocess_input(x), training=True)
                p_adv = model(preprocess_input(x_adv), training=True)
                ce = tf.keras.losses.sparse_categorical_crossentropy(y, p_nat)
                loss = tf.reduce_mean(w * ce) + beta * tf.reduce_mean(w * kl(p_nat, p_adv))
                loss += tf.add_n(model.losses) if model.losses else 0.0
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        return train_step

    return make_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, nargs="+", default=list(BETAS))
    parser.add_argument("--epochs", type=int, help="override Phase 1 epochs (quick test)")
    args = parser.parse_args()

    config = T.load_config()
    if args.epochs:
        config["epochs"] = args.epochs
    T.setup_gpu()
    T.train_grid("trades", [{"beta": b} for b in args.beta],
                 lambda p: make_step_for(p, config), config)


if __name__ == "__main__":
    main()
