"""Optional test-time defences for test.py, switched on in configs/test.yaml.

The defence code itself stays in src/defenses (Phase 3) and is loaded from there:
  preprocessing  -> <defences_dir>/preprocessing/defences.py    (filters the image)
  postprocessing -> <defences_dir>/post-processing/defences.py  (changes the output probabilities)

Order at test time:
  raw image 0-255 -> preprocessing chain -> preprocess_input -> model (incl. ConvAE if wrapped)
  -> postprocessing chain -> probabilities

test.yaml format (each chain runs top to bottom, each defence at most once):
  preprocessing:
    enabled: true
    permutations: false   # true = test no defence, each defence alone, then every order of 2..n
    chain:
      - jpeg: {quality: 90}
      - median: {kernel: 7}
  postprocessing:
    enabled: false
    chain:
      - revsig: {beta: 1.0, gamma: 0.5}
"""
import csv
import importlib.util
from itertools import permutations
from pathlib import Path

import numpy as np

# short name -> full name used inside the defences.py files
PRE_NAMES = {
    "jpeg": "jpeg_compression",
    "median": "median_smoothing",
    "gaussian": "gaussian_smoothing",
}
POST_NAMES = {
    "noise": "gaussian_noise",
    "highconf": "high_confidence",
    "revsig": "reverse_sigmoid",
}
# params that have no default in defences.py
REQUIRED_PARAMS = {
    "jpeg_compression": ["quality"],
    "median_smoothing": ["kernel"],
    "gaussian_smoothing": ["sigma"],
    "gaussian_noise": ["scale"],
    "high_confidence": ["cutoff"],
    "reverse_sigmoid": ["gamma"],
}


def _parse_chain(config, key, names):
    """Read one block of test.yaml. Returns (steps, params); empty when off or missing."""
    block = config.get(key) or {}
    if not block.get("enabled", False):
        return [], {}

    steps, params = [], {}
    full_names = {full: full for full in names.values()}
    for item in block.get("chain") or []:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(
                f"{key}.chain items must look like '- jpeg: {{quality: 90}}', got: {item!r}"
            )
        raw_name, item_params = next(iter(item.items()))
        name = names.get(raw_name) or full_names.get(raw_name)
        if name is None:
            raise ValueError(
                f"Unknown {key} defence '{raw_name}'. Use one of: {', '.join(names)}"
            )
        if name in params:
            raise ValueError(f"{key}.chain lists '{raw_name}' twice; use each defence once.")
        item_params = dict(item_params or {})
        missing = [p for p in REQUIRED_PARAMS[name] if p not in item_params]
        if missing:
            raise ValueError(f"{key} defence '{raw_name}' is missing: {', '.join(missing)}")
        steps.append(name)
        params[name] = item_params

    if not steps:
        raise ValueError(f"{key}.enabled is true in test.yaml, but {key}.chain is empty.")
    return steps, params


def parse_defences(config):
    """(pre_steps, pre_params, post_steps, post_params) from test.yaml. All empty = no defence."""
    pre_steps, pre_params = _parse_chain(config, "preprocessing", PRE_NAMES)
    post_steps, post_params = _parse_chain(config, "postprocessing", POST_NAMES)
    return pre_steps, pre_params, post_steps, post_params


def _load_file(path, module_name):
    """Import a .py file by path (the post-processing folder name has a hyphen)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Defence code not found: {path}. Check defences_dir in test.yaml "
            "(and the ../defenses/modules volume in docker-compose.yml)."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ClassOutput:
    """Model whose predict_on_batch returns only the sign-class probabilities.

    A full GRL model has two heads (class_output, domain_output); the defences need the class
    head only. Same rule as test.py: dict -> 'class_output', list/tuple -> first output.
    """

    def __init__(self, model):
        self.model = model

    def predict_on_batch(self, x):
        out = self.model.predict_on_batch(x)
        if isinstance(out, dict):
            return out["class_output"]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def build_predictor(model, defences_dir, pre_steps, pre_params, post_steps, post_params):
    """fn(batch of images 0-255) -> class probabilities, with the defences around the model."""
    model = _ClassOutput(model)
    defences_dir = Path(defences_dir)
    pre = _load_file(defences_dir / "preprocessing" / "defences.py", "test_time_pre_defences")
    # an empty pre chain is just preprocess_input + model, same as test.py
    pre_predict = pre.make_chain_predictor(pre_steps, model, pre_params)
    if not post_steps:
        return pre_predict

    # imports ART, so only loaded when postprocessing is on
    post = _load_file(defences_dir / "post-processing" / "defences.py", "test_time_post_defences")

    def predict(images):
        return post.apply_chain(pre_predict(images), post_steps, post_params)

    return predict


def describe(steps, params):
    """e.g. 'jpeg_compression(quality=90) -> median_smoothing(kernel=7)'."""
    parts = []
    for name in steps:
        settings = ", ".join(f"{k}={v}" for k, v in params[name].items())
        parts.append(f"{name}({settings})")
    return " -> ".join(parts)


SHORT_NAMES = {full: short for short, full in {**PRE_NAMES, **POST_NAMES}.items()}
NO_DEFENCE = "no_defence"


def permutation_flags(config):
    """(pre, post): True for a block that is enabled and has permutations: true."""

    def flag(key):
        block = config.get(key) or {}
        return bool(block.get("enabled", False) and block.get("permutations", False))

    return flag("preprocessing"), flag("postprocessing")


def chain_options(steps, permute):
    """The chains to test for one block.

    permute=False: just the chain as written. permute=True: no defence, each defence alone,
    then every ordered chain of 2..n (same as the src/defenses test scripts).
    """
    if not permute:
        return [list(steps)]
    options = [[]]
    for size in range(1, len(steps) + 1):
        options += [list(chain) for chain in permutations(steps, size)]
    return options


def chain_label(steps):
    """Folder-name label, e.g. 'jpeg_then_median'; 'none' for an empty chain."""
    return "_then_".join(SHORT_NAMES[s] for s in steps) if steps else "none"


def run_permutations(model, defences_dir, pre, post, dataset, output_directory, save_run, title=""):
    """Test every pre chain x every post chain on one split.

    pre/post: (steps, params, permute). save_run(folder, true, probs, extra_metrics) writes the
    five report files and returns the metrics. Results: output_directory/<run>/ per run, plus
    summary.csv and summary.md (ranked by accuracy, then macro F1).
    Each pre chain runs the model once; the post chains are applied to its saved output.
    """
    model = _ClassOutput(model)
    pre_steps, pre_params, pre_permute = pre
    post_steps, post_params, post_permute = post
    pre_options = chain_options(pre_steps, pre_permute)
    post_options = chain_options(post_steps, post_permute)
    total = len(pre_options) * len(post_options)
    print(
        f"Permutations: {len(pre_options)} preprocessing x {len(post_options)} "
        f"postprocessing = {total} runs",
        flush=True,
    )

    defences_dir = Path(defences_dir)
    pre_module = _load_file(defences_dir / "preprocessing" / "defences.py", "test_time_pre_defences")
    post_module = None
    if post_steps:
        post_module = _load_file(
            defences_dir / "post-processing" / "defences.py", "test_time_post_defences"
        )

    rows = []
    for pre_chain in pre_options:
        # model output per batch for this pre chain (post defences reuse it)
        pre_predict = pre_module.make_chain_predictor(pre_chain, model, pre_params)
        batch_probs, batch_labels = [], []
        for images, labels in dataset:
            batch_probs.append(np.asarray(pre_predict(images.numpy())))
            batch_labels.append(labels.numpy())
        true_labels = np.concatenate(batch_labels)

        for post_chain in post_options:
            if post_chain:  # per batch, exactly as in a single run
                probabilities = np.concatenate(
                    [post_module.apply_chain(p, post_chain, post_params) for p in batch_probs]
                )
            else:
                probabilities = np.concatenate(batch_probs)

            parts = []
            if pre_steps:
                parts.append(chain_label(pre_chain))
            if post_steps:
                parts.append(chain_label(post_chain))
            name = "__".join(parts) if (pre_chain or post_chain) else NO_DEFENCE

            pre_used = describe(pre_chain, pre_params) if pre_chain else "none"
            post_used = describe(post_chain, post_params) if post_chain else "none"
            metrics = save_run(
                Path(output_directory) / name,
                true_labels,
                probabilities,
                {"preprocessing_defences": pre_used, "postprocessing_defences": post_used},
            )
            rows.append({
                "run": name,
                "preprocessing": pre_used,
                "postprocessing": post_used,
                "n": metrics["number_of_test_images"],
                "accuracy": round(metrics["accuracy"], 6),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
                "f1_macro": round(metrics["f1_macro"], 6),
                "cross_entropy_loss": round(metrics["cross_entropy_loss"], 6),
                "mean_confidence": round(float(np.mean(probabilities.max(axis=1))), 6),
            })
            print(
                f"[{len(rows)}/{total}] {name}: accuracy {metrics['accuracy']:.4f}, "
                f"macro F1 {metrics['f1_macro']:.4f}",
                flush=True,
            )
            _write_summary(Path(output_directory), rows, title)  # progress kept if stopped

    print(f"Summary: {Path(output_directory) / 'summary.md'}")


def _write_summary(output_directory, rows, title):
    with (output_directory / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(rows, key=lambda r: (-r["accuracy"], -r["f1_macro"]))
    lines = [
        "# Test-time defence permutations",
        "",
        title,
        "Ranked by accuracy, then macro F1. Chains run left to right; "
        f"`{NO_DEFENCE}` is the model with no defence (the reference row).",
        "",
        "| Rank | Run | Preprocessing | Postprocessing | Accuracy | Balanced accuracy "
        "| Macro F1 | Cross-entropy | Mean confidence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(ranked, start=1):
        run = f"**{r['run']}**" if r["run"] == NO_DEFENCE else r["run"]
        lines.append(
            f"| {rank} | {run} | {r['preprocessing']} | {r['postprocessing']} "
            f"| {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['f1_macro']:.4f} "
            f"| {r['cross_entropy_loss']:.4f} | {r['mean_confidence']:.4f} |"
        )
    (output_directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def predict_dataset(predict, dataset):
    """Run predict over a batched (image, label) dataset of raw 0-255 images."""
    true_labels, probabilities = [], []
    for index, (images, labels) in enumerate(dataset, start=1):
        probabilities.append(np.asarray(predict(images.numpy())))
        true_labels.append(labels.numpy())
        if index % 10 == 0:
            print(f"  defended prediction: {index} batches done", flush=True)
    return np.concatenate(true_labels), np.concatenate(probabilities)
