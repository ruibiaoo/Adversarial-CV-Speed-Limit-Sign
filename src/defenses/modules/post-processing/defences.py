"""Post-processing defences: ART postprocessors in front of the frozen base classifier.

Unlike the preprocessing defences, these do not touch the image. They transform the
model's output probability vector at test time:

  P1 gaussian_noise    add N(0, scale^2) noise to the probability vector (ART GaussianNoise)
  P2 high_confidence   zero every probability below a cutoff (ART HighConfidence)
  P3 reverse_sigmoid   add a reverse-sigmoid perturbation that corrupts the vector while
                       keeping the top class (ART ReverseSigmoid; Lee et al., 2019)

ART's fourth postprocessor, Rounded, is left out: it only truncates probabilities to a
number of decimals and never changes the argmax, so it cannot help or hurt accuracy and
is the weakest of the four (see README.md).

make_predictor()/make_chain_predictor() turn a defence (or an ordered chain of them) into
fn(BxHxWx3 images in 0-255) -> BxC class scores, so the evaluation treats them like the
preprocessing defences. The base model's softmax is computed once; the postprocessors
then compose on the (N, C) array, so order matters (e.g. high_confidence before vs after
gaussian_noise gives different vectors) and chains are enumerated as permutations.

No attack data or attack knowledge is used here: the grids are general ranges on the
output vector, not taken from any attack setting.
"""
import numpy as np
from art.defences.postprocessor import GaussianNoise, HighConfidence, ReverseSigmoid

GAUSSIAN_NOISE = "gaussian_noise"
HIGH_CONFIDENCE = "high_confidence"
REVERSE_SIGMOID = "reverse_sigmoid"

# Short names for the CLI (--defence) and for the chain folder labels (noise_then_highconf, ...).
ALIASES = {"noise": GAUSSIAN_NOISE, "highconf": HIGH_CONFIDENCE, "revsig": REVERSE_SIGMOID}
SHORT = {full: short for short, full in ALIASES.items()}

# Grids, mildest setting first (the tuning rule uses this order for ties). "Strength" means how
# much the output vector is distorted: larger scale, higher cutoff and larger gamma all distort more.
GRIDS = {
    GAUSSIAN_NOISE: [{"scale": s} for s in (0.01, 0.05, 0.1, 0.2, 0.3)],
    HIGH_CONFIDENCE: [{"cutoff": c} for c in (0.1, 0.25, 0.5, 0.75, 0.9)],
    REVERSE_SIGMOID: [{"beta": 1.0, "gamma": g} for g in (0.1, 0.2, 0.3, 0.5)],
}
NOISE_SEED = 4266  # GaussianNoise draws from numpy's global RNG; fixed so runs are reproducible


def build_postprocessor(name, params):
    """One ART postprocessor instance (apply_predict=True so __call__ transforms predictions)."""
    if name == GAUSSIAN_NOISE:
        return GaussianNoise(scale=float(params["scale"]), apply_fit=False, apply_predict=True)
    if name == HIGH_CONFIDENCE:
        return HighConfidence(cutoff=float(params["cutoff"]), apply_fit=False, apply_predict=True)
    if name == REVERSE_SIGMOID:
        return ReverseSigmoid(beta=float(params.get("beta", 1.0)), gamma=float(params["gamma"]),
                              apply_fit=False, apply_predict=True)
    raise ValueError(f"unknown post-processing defence: {name}")


def normalize(scores):
    """Clip to non-negative and renormalize each row to a valid probability distribution.

    ART postprocessors can leave rows that do not sum to 1 (noise) or that are all zero
    (high_confidence when no class clears the cutoff). A proper distribution is needed for
    the probability-based metrics; all-zero rows fall back to uniform.
    """
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    totals = scores.sum(axis=1, keepdims=True)
    uniform = np.full_like(scores, 1.0 / scores.shape[1])
    return np.where(totals > 0, np.divide(scores, totals, where=totals > 0), uniform)


def base_classify(model):
    """fn(BxHxWx3 images in 0-255, uint8 or float) -> BxC softmax of the base model."""
    from tensorflow.keras.applications.resnet import preprocess_input

    def classify(images):
        return np.asarray(model.predict_on_batch(preprocess_input(np.array(images, dtype=np.float32))))

    return classify


def apply_chain(base_probs, steps, params):
    """Apply the named postprocessors to base_probs in order, then normalize once."""
    np.random.seed(NOISE_SEED)  # deterministic GaussianNoise draw per call
    scores = np.asarray(base_probs, dtype=np.float64)
    for name in steps:
        scores = np.asarray(build_postprocessor(name, params[name])(scores))
    return normalize(scores)


def make_predictor(name, model, params=None):
    """fn(images 0-255) -> BxC scores for one post-processing defence in front of model."""
    classify = base_classify(model)
    params = {name: params or {}}
    return lambda images: apply_chain(classify(images), [name], params)


def make_chain_predictor(steps, model, params):
    """fn(images 0-255) -> BxC scores for several post-processing defences applied in order.

    steps: defence names in application order. params: {name: params dict}. The base softmax is
    computed once; the postprocessors then compose on the (N, C) array (order matters).
    """
    classify = base_classify(model)
    return lambda images: apply_chain(classify(images), steps, params)


def entropy(probs):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def obfuscation_metrics(base_probs, defended_probs):
    """How much the output vector is distorted vs the undefended base, as a secondary tuning signal.

    confidence_drop: mean fall in the top-1 probability. entropy_increase: mean rise in output
    entropy (a flatter vector leaks less to a score-based attacker). Both use normalized rows.
    """
    base, defended = normalize(base_probs), normalize(defended_probs)
    return {
        "confidence_drop": float(np.mean(base.max(axis=1) - defended.max(axis=1))),
        "entropy_increase": float(np.mean(entropy(defended) - entropy(base))),
    }
