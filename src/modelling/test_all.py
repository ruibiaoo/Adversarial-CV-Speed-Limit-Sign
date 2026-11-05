"""Evaluate all configured combinations of preprocessing, models, and postprocessing across multiple datasets."""
from __future__ import annotations

import argparse
import gc
from itertools import combinations, permutations
from pathlib import Path
import random
import re
import sys
import threading
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import yaml

if __package__:
    from . import test as single_test
    from .modules import test_time_defences as ttd
    from .modules.convae import (
        is_wrapped_with_convae,
        load_convae,
        resolve_convae_path,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test as single_test
    from modules import test_time_defences as ttd
    from modules.convae import (
        is_wrapped_with_convae,
        load_convae,
        resolve_convae_path,
    )


CONVAE_STEP = "convae"

PRE_NAMES = {
    **ttd.PRE_NAMES,
    "convae": CONVAE_STEP,
}
POST_NAMES = dict(ttd.POST_NAMES)

SHORT_NAMES = {
    **ttd.SHORT_NAMES,
    CONVAE_STEP: "convae",
}

REQUIRED_PARAMS = {
    **ttd.REQUIRED_PARAMS,
    CONVAE_STEP: ["model_path"],
}

# Ordered list of all classification metrics that get their own <metric>.txt summary table
METRIC_FILES = [
    "accuracy",
    "balanced_accuracy",
    "top_3_accuracy",
    "top_5_accuracy",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "cross_entropy_loss",
    "matthews_correlation_coefficient",
    "cohen_kappa",
    "roc_auc_macro_ovr",
    "roc_auc_weighted_ovr",
    "average_precision_macro",
    "average_precision_weighted",
]


def _is_true(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_path(value: str | Path, base_directory: Path) -> Path:
    """Resolve paths both inside the Docker container (/app/...) and locally on the host."""
    candidate = Path(value)
    if candidate.exists():
        return candidate

    normalized = str(value).replace("\\", "/")
    repo_dir = base_directory.parents[1] if len(base_directory.parents) > 1 else None

    if repo_dir is not None:
        # Map /app/eval_data[/...] -> <repo>/data/data_updated[/...]
        if normalized == "/app/eval_data" or normalized.startswith("/app/eval_data/"):
            rel = normalized[len("/app/eval_data") :].lstrip("/")
            mapped = repo_dir / "data" / "data_updated" / rel
            if mapped.exists():
                return mapped

        # Map /app/data[/...] -> <repo>/data/data_combined[/...] or <repo>/data/data_updated[/...]
        if normalized == "/app/data" or normalized.startswith("/app/data/"):
            rel = normalized[len("/app/data") :].lstrip("/")
            for folder in ("data_combined", "data_updated"):
                mapped = repo_dir / "data" / folder / rel
                if mapped.exists():
                    return mapped

        # Map /defenses/... -> <repo>/src/defenses/...
        if normalized.startswith("/defenses/"):
            rel = normalized.lstrip("/")
            mapped = repo_dir / "src" / rel
            if mapped.exists():
                return mapped

    if normalized.startswith("/app/"):
        rel_from_app = normalized[len("/app/") :]
        if repo_dir is not None and Path(rel_from_app).parts:
            data_candidate = repo_dir / "data" / rel_from_app
            if data_candidate.exists():
                return data_candidate
        return base_directory / rel_from_app

    if candidate.is_absolute():
        return candidate

    rel_candidate = (base_directory / candidate).resolve()
    if rel_candidate.exists():
        return rel_candidate

    # Inside Docker, ../defenses/modules from /app resolves to /defenses/modules
    docker_defences = Path("/defenses/modules")
    if "defenses" in normalized and docker_defences.exists():
        return docker_defences

    if repo_dir is not None and (repo_dir / candidate).exists():
        return (repo_dir / candidate).resolve()

    return base_directory / candidate


def sanitize_folder_token(text: str) -> str:
    """Make a string safe for directory names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return cleaned.strip("_") or "default"


def reset_all_seeds(seed: int) -> None:
    """Reset Python, NumPy, and TensorFlow RNG seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def clear_memory() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def _normalize_defence_entries(raw_entries, key: str, allowed_names: dict) -> tuple[list[str], dict[str, dict]]:
    """Parse `defences` (dict or list of single-key dicts) into (ordered_step_names, params_by_full_name)."""
    if not raw_entries:
        return [], {}

    items: list[tuple[str, dict]] = []
    if isinstance(raw_entries, dict):
        for raw_name, raw_params in raw_entries.items():
            items.append((str(raw_name), dict(raw_params or {})))
    elif isinstance(raw_entries, (list, tuple)):
        for entry in raw_entries:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    f"{key} list entries must be single-key dicts like '- jpeg: {{quality: 90}}', got: {entry!r}"
                )
            raw_name, raw_params = next(iter(entry.items()))
            items.append((str(raw_name), dict(raw_params or {})))
    else:
        raise ValueError(f"Unsupported format for {key}: {raw_entries!r}")

    full_lookup = {full: full for full in allowed_names.values()}
    steps: list[str] = []
    params: dict[str, dict] = {}

    for raw_name, item_params in items:
        full_name = allowed_names.get(raw_name) or full_lookup.get(raw_name)
        if full_name is None:
            valid = ", ".join(sorted(allowed_names.keys()))
            raise ValueError(f"Unknown {key} defence {raw_name!r}. Valid options: {valid}")
        if full_name in params:
            raise ValueError(f"Duplicate defence {raw_name!r} in {key}; list each defence at most once.")
        missing = [p for p in REQUIRED_PARAMS.get(full_name, []) if p not in item_params]
        if missing:
            raise ValueError(f"Defence {raw_name!r} in {key} is missing required parameter(s): {', '.join(missing)}")
        steps.append(full_name)
        params[full_name] = item_params

    return steps, params


def _parse_explicit_chain(raw_chain, key: str, allowed_names: dict, configured_params: dict) -> list[str]:
    """Convert an entry from `chains:` into a list of canonical full defence names."""
    if raw_chain is None or raw_chain == [] or str(raw_chain).strip().lower() in ("none", "no_defence", ""):
        return []

    if isinstance(raw_chain, str):
        tokens = [
            tok.strip()
            for tok in re.split(r"->|_then_|,", raw_chain)
            if tok.strip() and tok.strip().lower() not in ("none", "no_defence")
        ]
    elif isinstance(raw_chain, (list, tuple)):
        tokens = [
            str(tok).strip()
            for tok in raw_chain
            if str(tok).strip() and str(tok).strip().lower() not in ("none", "no_defence")
        ]
    else:
        raise ValueError(f"Invalid chain entry in {key}.chains: {raw_chain!r}")

    full_lookup = {full: full for full in allowed_names.values()}
    resolved: list[str] = []
    for tok in tokens:
        full_name = allowed_names.get(tok) or full_lookup.get(tok)
        if full_name is None:
            raise ValueError(f"Unknown defence {tok!r} in {key}.chains.")
        if full_name not in configured_params:
            raise ValueError(
                f"Defence {tok!r} in {key}.chains is not configured under {key}.defences."
            )
        if full_name in resolved:
            raise ValueError(f"Defence {tok!r} appears multiple times in chain {raw_chain!r}.")
        resolved.append(full_name)
    return resolved


def parse_defence_block(config: dict, key: str, allowed_names: dict, default_seed: int) -> tuple[list[list[str]], dict[str, dict]]:
    """Return (list_of_chains, params_by_full_name) for either `preprocessing` or `postprocessing`."""
    block = config.get(key) or {}
    enabled = _is_true(block.get("enabled", False), default=False)
    if not enabled:
        return [[]], {}

    raw_defences = block.get("defences")
    if raw_defences is None:
        raw_defences = block.get("chain")

    steps, params = _normalize_defence_entries(raw_defences, key, allowed_names)
    if "gaussian_smoothing" in params:
        params["gaussian_smoothing"].setdefault("seed", default_seed)

    include_none = _is_true(block.get("include_none", True), default=True)

    # Determine generation mode
    raw_mode = block.get("mode")
    if raw_mode is None:
        if "permutations" in block:
            mode = "permutations" if _is_true(block.get("permutations"), default=False) else "single_chain"
        else:
            mode = "permutations"
    else:
        mode = str(raw_mode).strip().lower()

    chains: list[list[str]] = []
    if include_none:
        chains.append([])

    if mode == "permutations":
        for size in range(1, len(steps) + 1):
            for perm in permutations(steps, size):
                chain = list(perm)
                if chain not in chains:
                    chains.append(chain)
    elif mode == "combinations":
        for size in range(1, len(steps) + 1):
            for comb in combinations(steps, size):
                chain = list(comb)
                if chain not in chains:
                    chains.append(chain)
    elif mode in ("single_chain", "chain", "fixed"):
        if steps and steps not in chains:
            chains.append(list(steps))
    elif mode == "explicit":
        raw_chains = block.get("chains")
        if raw_chains is None:
            raise ValueError(f"{key}.mode is 'explicit', but {key}.chains is not defined.")
        explicit_chains: list[list[str]] = []
        if include_none:
            explicit_chains.append([])
        for item in raw_chains:
            parsed = _parse_explicit_chain(item, key, allowed_names, params)
            if parsed not in explicit_chains:
                explicit_chains.append(parsed)
        chains = explicit_chains
    else:
        raise ValueError(
            f"Unsupported {key}.mode: {mode!r}. Use 'permutations', 'combinations', or 'explicit'."
        )

    if not chains:
        chains = [[]]
    return chains, params


def chain_display_name(steps: list[str]) -> str:
    """Human-readable short string name for tables and config.txt (e.g. 'none', 'jpeg', 'jpeg -> median')."""
    if not steps:
        return "none"
    return " -> ".join(SHORT_NAMES.get(s, s) for s in steps)


def chain_folder_name(steps: list[str]) -> str:
    """Filesystem-friendly chain label (e.g. 'none', 'jpeg', 'jpeg_then_median')."""
    if not steps:
        return "none"
    return "_then_".join(SHORT_NAMES.get(s, s) for s in steps)


def chain_detailed_description(steps: list[str], params: dict[str, dict]) -> str:
    """Detailed description including parameter values (e.g. 'jpeg_compression(quality=90) -> median_smoothing(kernel=7)')."""
    if not steps:
        return "none"
    parts = []
    for name in steps:
        step_params = params.get(name, {})
        settings = ", ".join(f"{k}={v}" for k, v in step_params.items())
        parts.append(f"{name}({settings})")
    return " -> ".join(parts)


def preload_datasets(
    datasets_cfg: dict[str, str],
    class_names: list[str],
    image_size: int,
    batch_size: int,
    base_directory: Path,
) -> dict[str, dict]:
    """Load all configured datasets into memory once (raw [0, 255] float32 batches) so I/O does not skew runtime."""
    if not isinstance(datasets_cfg, dict) or not datasets_cfg:
        raise ValueError("`datasets` in test_all.yaml must be a non-empty dictionary mapping dataset_name -> path.")

    loaded: dict[str, dict] = {}
    for ds_name, raw_path in datasets_cfg.items():
        resolved_dir = resolve_path(raw_path, base_directory)
        if not resolved_dir.exists():
            raise FileNotFoundError(
                f"Dataset {ds_name!r} directory not found: {resolved_dir} (configured as {raw_path!r})"
            )

        ds = tf.keras.utils.image_dataset_from_directory(
            resolved_dir,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=False,
        )
        file_paths = list(ds.file_paths)
        batches: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        for images, labels in ds:
            batches.append(images.numpy())
            labels_list.append(labels.numpy())

        true_labels = np.concatenate(labels_list, axis=0) if labels_list else np.array([], dtype=np.int32)
        loaded[str(ds_name)] = {
            "name": str(ds_name),
            "raw_path": str(raw_path),
            "resolved_path": resolved_dir,
            "file_paths": file_paths,
            "batches": batches,
            "true_labels": true_labels,
            "num_images": int(len(true_labels)),
        }
        print(
            f"Loaded dataset '{ds_name}' ({len(true_labels)} images in {len(batches)} batches) from {resolved_dir}",
            flush=True,
        )
    return loaded


def build_combined_predictor(
    model,
    pre_module,
    post_module,
    pre_steps: list[str],
    pre_params: dict[str, dict],
    post_steps: list[str],
    post_params: dict[str, dict],
    convae_model=None,
    seed: int = 4266,
):
    """Build fn(batch_images_0_255) -> probabilities for any (pre_steps, model, post_steps) combination."""
    wrapped_model = ttd._ClassOutput(model)
    model_lock = threading.Lock()

    def classify_caffe(caffe_input: np.ndarray) -> np.ndarray:
        with model_lock:
            return np.asarray(wrapped_model.predict_on_batch(caffe_input))

    def classify(images_0_255: np.ndarray) -> np.ndarray:
        caffe_input = preprocess_input(np.array(images_0_255, dtype=np.float32))
        return classify_caffe(caffe_input)

    def apply_deterministic_step(images: np.ndarray, step_name: str, workers: int = 1) -> np.ndarray:
        if step_name == CONVAE_STEP:
            if convae_model is None:
                raise RuntimeError("ConvAE step requested in preprocessing chain, but convae_model was not loaded.")
            x_01 = np.clip(np.asarray(images, dtype=np.float32) / 255.0, 0.0, 1.0)
            with model_lock:
                rec_01 = np.asarray(convae_model.predict_on_batch(x_01))
            return np.clip(rec_01, 0.0, 1.0) * 255.0
        transform = pre_module.transform_for(step_name, pre_params[step_name])
        uint8_imgs = pre_module._to_uint8(images)
        if workers > 1 and len(uint8_imgs) > 1:
            executor = pre_module._get_executor(workers)
            return np.stack(list(executor.map(transform, uint8_imgs)))
        return np.stack([transform(im) for im in uint8_imgs])

    def apply_deterministic_chain(images: np.ndarray, step_names: list[str], workers: int = 1) -> np.ndarray:
        out = images
        for s in step_names:
            out = apply_deterministic_step(out, s, workers=workers)
        return out

    gaussian_name = "gaussian_smoothing"
    if gaussian_name not in pre_steps:
        def pre_predict(images: np.ndarray) -> np.ndarray:
            filtered = apply_deterministic_chain(images, pre_steps)
            return classify(filtered)
    else:
        at = pre_steps.index(gaussian_name)
        before = pre_steps[:at]
        after = pre_steps[at + 1 :]
        sigma = float(pre_params[gaussian_name]["sigma"])
        n = int(pre_params[gaussian_name].get("samples", pre_module.SMOOTHING_SAMPLES))
        g_seed = int(pre_params[gaussian_name].get("seed", seed))
        workers = max(
            1,
            int(pre_params[gaussian_name].get("workers", getattr(pre_module, "SMOOTHING_WORKERS", 8))),
        )

        def pre_predict(images: np.ndarray) -> np.ndarray:
            rng = np.random.default_rng(g_seed)
            starts = list(range(0, len(images), 2))
            if not starts:
                return np.zeros((0, 0), dtype=np.float32)

            chunk_rngs = pre_module._spawn_rngs(rng, len(starts))

            if len(starts) == 1 or workers <= 1:
                counts = []
                for start, chunk_rng in zip(starts, chunk_rngs):
                    batch = apply_deterministic_chain(images[start : start + 2], before, workers=workers)
                    noisy = pre_module.gaussian_noise(
                        np.repeat(batch, n, axis=0), sigma, chunk_rng, workers=workers
                    )
                    ready = apply_deterministic_chain(noisy, after, workers=workers)
                    scores = classify(ready)
                    pred = np.argmax(scores, axis=1).reshape(len(batch), n)
                    counts += [np.bincount(p, minlength=scores.shape[1]) for p in pred]
                return np.stack(counts).astype(np.float32) / float(n)

            executor = pre_module._get_executor(workers)

            def prepare_chunk(start: int, chunk_rng) -> tuple[int, np.ndarray]:
                batch = apply_deterministic_chain(images[start : start + 2], before, workers=1)
                noisy = pre_module._gaussian_noise_slice(np.repeat(batch, n, axis=0), sigma, chunk_rng)
                ready = apply_deterministic_chain(noisy, after, workers=1)
                caffe_input = preprocess_input(np.array(ready, dtype=np.float32))
                return len(batch), caffe_input

            futures = [
                executor.submit(prepare_chunk, start, chunk_rng)
                for start, chunk_rng in zip(starts, chunk_rngs)
            ]
            counts = []
            for fut in futures:
                batch_len, caffe_input = fut.result()
                scores = classify_caffe(caffe_input)
                pred = np.argmax(scores, axis=1).reshape(batch_len, n)
                counts += [np.bincount(p, minlength=scores.shape[1]) for p in pred]
            return np.stack(counts).astype(np.float32) / float(n)

    if not post_steps:
        return pre_predict

    def full_predict(images: np.ndarray) -> np.ndarray:
        base_probs = pre_predict(images)
        post_module.NOISE_SEED = int(seed)
        return post_module.apply_chain(base_probs, post_steps, post_params)

    return full_predict


def write_config_txt(
    path: Path,
    pre_display: str,
    pre_details: str,
    model_name: str,
    model_raw_path: str,
    post_display: str,
    post_details: str,
    dataset_names: list[str],
    seed: int,
    num_runs: int,
    warmup_runs: int,
    runtime_stats: dict,
) -> None:
    """Write config.txt inside a combination's folder under <output_dir>/tests/<combo>/."""
    lines = [
        "TEST CONFIGURATION",
        "==================",
        f"Preprocessing: {pre_display}",
        f"Preprocessing Details: {pre_details}",
        f"Model: {model_name}",
        f"Model Path: {model_raw_path}",
        f"Postprocessing: {post_display}",
        f"Postprocessing Details: {post_details}",
        f"Datasets: {', '.join(dataset_names)}",
        f"Seed: {seed}",
        f"Warmup Runs: {warmup_runs}",
        f"Number of Runs: {num_runs}",
        "",
        "RUNTIME SUMMARY",
        "===============",
        f"Total Runtime (s): {runtime_stats['total_runtime']:.6f}",
        f"Average Runtime per Run (s): {runtime_stats['avg_runtime']:.6f}",
        f"Total Runs: {runtime_stats['total_runs']}",
        f"Total Inferences per Run: {runtime_stats['inferences_per_run']}",
        f"Average Time per Inference (s): {runtime_stats['avg_time_per_inference_s']:.6f}",
        f"Average Time per Inference (ms): {runtime_stats['avg_time_per_inference_ms']:.4f}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_table(headers: list[str], rows: list[list[str]], left_align_cols: int = 3) -> str:
    """Format a 2D list of strings as a padded plain-text table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    sep = "  "
    header_cells = [
        headers[i].ljust(col_widths[i]) if i < left_align_cols else headers[i].rjust(col_widths[i])
        for i in range(len(headers))
    ]
    divider_cells = ["-" * col_widths[i] for i in range(len(headers))]

    lines = [sep.join(header_cells), sep.join(divider_cells)]
    for row in rows:
        formatted_cells = [
            str(row[i]).ljust(col_widths[i]) if i < left_align_cols else str(row[i]).rjust(col_widths[i])
            for i in range(len(headers))
        ]
        lines.append(sep.join(formatted_cells))
    return "\n".join(lines) + "\n"


def write_summary_tables(
    output_dir: Path,
    dataset_names: list[str],
    combination_records: list[dict],
) -> None:
    """Write <output_dir>/<metric>.txt for every metric and <output_dir>/runtime.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write one .txt file per classification metric
    metric_headers = ["Preprocessing", "Model", "Postprocessing", *dataset_names]
    for metric_name in METRIC_FILES:
        table_rows: list[list[str]] = []
        for rec in combination_records:
            row = [
                rec["pre_display"],
                rec["model_name"],
                rec["post_display"],
            ]
            for ds_name in dataset_names:
                ds_metrics = rec["dataset_metrics"].get(ds_name, {})
                val = ds_metrics.get(metric_name)
                if isinstance(val, (int, float)):
                    row.append(f"{float(val):.6f}")
                else:
                    row.append("N/A")
            table_rows.append(row)

        content = f"Metric: {metric_name}\n\n" + _format_table(metric_headers, table_rows, left_align_cols=3)
        (output_dir / f"{metric_name}.txt").write_text(content, encoding="utf-8")

    # 2. Write runtime.txt
    runtime_headers = [
        "Preprocessing",
        "Model",
        "Postprocessing",
        "Total Runtime (s)",
        "Avg Runtime (s)",
        "Total Runs",
        "Inferences per Run",
        "Avg Time per Inference (s)",
        "Avg Time per Inference (ms)",
    ]
    runtime_rows: list[list[str]] = []
    for rec in combination_records:
        rt = rec["runtime"]
        runtime_rows.append([
            rec["pre_display"],
            rec["model_name"],
            rec["post_display"],
            f"{rt['total_runtime']:.6f}",
            f"{rt['avg_runtime']:.6f}",
            str(rt["total_runs"]),
            str(rt["inferences_per_run"]),
            f"{rt['avg_time_per_inference_s']:.6f}",
            f"{rt['avg_time_per_inference_ms']:.4f}",
        ])

    runtime_content = "Metric: runtime\n\n" + _format_table(runtime_headers, runtime_rows, left_align_cols=3)
    (output_dir / "runtime.txt").write_text(runtime_content, encoding="utf-8")


def parse_args():
    phase_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate all combinations of preprocessing, models, and postprocessing across multiple datasets."
    )
    parser.add_argument(
        "--config",
        default=str(phase_directory / "configs" / "test_all.yaml"),
        help="Path to the test_all YAML configuration file (defaults to configs/test_all.yaml).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    phase_directory = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = phase_directory / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(config_path)

    # Let GPU memory grow dynamically
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    output_dir = resolve_path(config.get("output_dir", "/app/results/test_all"), phase_directory)
    tests_root = output_dir / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)

    num_runs = int(config.get("num_runs", 1))
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")

    warmup_runs = int(config.get("warmup_runs", 1))
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")

    seed = int(config.get("seed", 4266))
    image_size = int(config.get("image_size", 256))
    batch_size = int(config.get("batch", 16))
    class_names = list(config.get("class_names") or [])
    if not class_names:
        raise ValueError("`class_names` must be a non-empty list in test_all.yaml.")
    num_classes = len(class_names)

    # Validate models dictionary
    models_cfg = config.get("models")
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise ValueError("`models` in test_all.yaml must be a non-empty dictionary mapping model_name -> model_path.")

    resolved_models: dict[str, tuple[str, Path]] = {}
    for model_name, raw_model_path in models_cfg.items():
        resolved_model_path = resolve_path(raw_model_path, phase_directory)
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"Model {model_name!r} checkpoint not found: {resolved_model_path} (configured as {raw_model_path!r})"
            )
        resolved_models[str(model_name)] = (str(raw_model_path), resolved_model_path)

    # Parse preprocessing and postprocessing chains
    pre_chains, pre_params = parse_defence_block(config, "preprocessing", PRE_NAMES, default_seed=seed)
    post_chains, post_params = parse_defence_block(config, "postprocessing", POST_NAMES, default_seed=seed)

    pre_block = config.get("preprocessing") or {}
    skip_duplicate_convae = _is_true(pre_block.get("skip_duplicate_convae", True), default=True)

    # Load defence helper modules from src/defenses/modules
    defences_dir = resolve_path(config.get("defences_dir", "../defenses/modules"), phase_directory)
    pre_module = ttd._load_file(defences_dir / "preprocessing" / "defences.py", "test_all_pre_defences")
    post_module = None
    if any(len(c) > 0 for c in post_chains):
        post_module = ttd._load_file(defences_dir / "post-processing" / "defences.py", "test_all_post_defences")

    # Validate ConvAE checkpoint path if `convae` is used as a preprocessing step
    convae_resolved_path = None
    if CONVAE_STEP in pre_params:
        raw_ae = pre_params[CONVAE_STEP]["model_path"]
        convae_resolved_path = resolve_convae_path(resolve_path(raw_ae, phase_directory))

    # Pre-load all datasets into memory once
    datasets_cfg = config.get("datasets")
    loaded_datasets = preload_datasets(
        datasets_cfg=datasets_cfg,
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        base_directory=phase_directory,
    )
    dataset_names = list(loaded_datasets.keys())
    inferences_per_run = sum(ds["num_images"] for ds in loaded_datasets.values())

    print(
        f"\nConfiguration loaded:\n"
        f"  Models ({len(resolved_models)}): {', '.join(resolved_models.keys())}\n"
        f"  Datasets ({len(dataset_names)}): {', '.join(dataset_names)} ({inferences_per_run} total images/run)\n"
        f"  Preprocessing chains ({len(pre_chains)}): {', '.join(chain_display_name(c) for c in pre_chains)}\n"
        f"  Postprocessing chains ({len(post_chains)}): {', '.join(chain_display_name(c) for c in post_chains)}\n"
        f"  Warmup runs per model: {warmup_runs}\n"
        f"  Runs per combination: {num_runs} (seed={seed})\n"
        f"  Main output directory: {output_dir}\n",
        flush=True,
    )

    combination_records: list[dict] = []
    total_upper_bound = len(resolved_models) * len(pre_chains) * len(post_chains)
    combo_counter = 0

    for model_name, (raw_model_path, model_path) in resolved_models.items():
        print(f"\n============================================================", flush=True)
        print(f"Loading model '{model_name}' from {model_path} ...", flush=True)
        reset_all_seeds(seed)
        model = single_test.load_classifier(model_path, image_size=image_size, num_classes=num_classes)

        input_height = model.input_shape[1]
        input_width = model.input_shape[2]
        if input_height != image_size or input_width != image_size:
            raise ValueError(
                f"Model '{model_name}' expects {input_height}x{input_width} images, "
                f"but test_all.yaml specifies image_size={image_size}."
            )

        model_has_embedded_convae = is_wrapped_with_convae(model)

        # Load standalone ConvAE preprocessor if needed for this model's preprocessing chains
        convae_model = None
        if convae_resolved_path is not None and not (skip_duplicate_convae and model_has_embedded_convae):
            convae_model = load_convae(convae_resolved_path, img_size=image_size)
            convae_model.trainable = False

        # Warm up the loaded model (and ConvAE) across all unique batch sizes so TF graph
        # compilation and cuDNN autotuning (including on the final remainder batch) do not
        # skew the first run's timing.
        if warmup_runs > 0:
            unique_raw_batches: dict[int, np.ndarray] = {}
            for ds_info in loaded_datasets.values():
                for batch in ds_info["batches"]:
                    b_size = int(len(batch))
                    if b_size > 0 and b_size not in unique_raw_batches:
                        unique_raw_batches[b_size] = np.asarray(batch, dtype=np.float32)

            if "gaussian_smoothing" in pre_params and unique_raw_batches:
                n_samples = int(pre_params["gaussian_smoothing"].get("samples", getattr(pre_module, "SMOOTHING_SAMPLES", 32)))
                sample_img = next(iter(unique_raw_batches.values()))[:1]
                for g_bsize in (1, 2, n_samples, 2 * n_samples):
                    if g_bsize > 0 and g_bsize not in unique_raw_batches:
                        unique_raw_batches[g_bsize] = np.repeat(sample_img, g_bsize, axis=0)

            if unique_raw_batches:
                shapes_str = ", ".join(str(b) for b in sorted(unique_raw_batches.keys(), reverse=True))
                print(
                    f"Running {warmup_runs} warmup pass(es) for model '{model_name}' "
                    f"(batch sizes: {shapes_str}) ...",
                    flush=True,
                )
                warmup_clf = ttd._ClassOutput(model)
                warmup_inputs = [
                    (
                        preprocess_input(raw_b.copy()),
                        np.clip(raw_b / 255.0, 0.0, 1.0) if convae_model is not None else None,
                    )
                    for raw_b in unique_raw_batches.values()
                ]
                for _ in range(warmup_runs):
                    for warmup_batch, warmup_01 in warmup_inputs:
                        _ = warmup_clf.predict_on_batch(warmup_batch)
                        if convae_model is not None and warmup_01 is not None:
                            _ = convae_model.predict_on_batch(warmup_01)

        for pre_chain in pre_chains:
            if CONVAE_STEP in pre_chain and model_has_embedded_convae and skip_duplicate_convae:
                print(
                    f"Skipping preprocessing chain '{chain_display_name(pre_chain)}' for model '{model_name}' "
                    f"(model already embeds a ConvAE preprocessor).",
                    flush=True,
                )
                continue

            for post_chain in post_chains:
                combo_counter += 1
                pre_display = chain_display_name(pre_chain)
                post_display = chain_display_name(post_chain)
                pre_details = chain_detailed_description(pre_chain, pre_params)
                post_details = chain_detailed_description(post_chain, post_params)

                combo_folder_name = (
                    f"pre_{sanitize_folder_token(chain_folder_name(pre_chain))}__"
                    f"model_{sanitize_folder_token(model_name)}__"
                    f"post_{sanitize_folder_token(chain_folder_name(post_chain))}"
                )
                combo_dir = tests_root / combo_folder_name

                print(
                    f"\n[{combo_counter}/{total_upper_bound}] "
                    f"Pre: {pre_display} | Model: {model_name} | Post: {post_display}",
                    flush=True,
                )

                predict_fn = build_combined_predictor(
                    model=model,
                    pre_module=pre_module,
                    post_module=post_module,
                    pre_steps=pre_chain,
                    pre_params=pre_params,
                    post_steps=post_chain,
                    post_params=post_params,
                    convae_model=convae_model,
                    seed=seed,
                )

                run_times: list[float] = []
                dataset_run_times: dict[str, list[float]] = {ds_name: [] for ds_name in dataset_names}
                saved_probabilities: dict[str, np.ndarray] = {}

                for run_idx in range(num_runs):
                    reset_all_seeds(seed)
                    run_elapsed = 0.0

                    for ds_name, ds_info in loaded_datasets.items():
                        ds_t0 = time.perf_counter()
                        batch_probs = [
                            np.asarray(predict_fn(batch_images))
                            for batch_images in ds_info["batches"]
                        ]
                        ds_elapsed = time.perf_counter() - ds_t0
                        run_elapsed += ds_elapsed
                        dataset_run_times[ds_name].append(ds_elapsed)

                        probs = (
                            np.concatenate(batch_probs, axis=0)
                            if batch_probs
                            else np.zeros((0, num_classes), dtype=np.float32)
                        )
                        if run_idx == 0:
                            saved_probabilities[ds_name] = probs

                    run_times.append(run_elapsed)
                    print(
                        f"  Run {run_idx + 1}/{num_runs}: {run_elapsed:.4f}s "
                        f"({inferences_per_run} images)",
                        flush=True,
                    )

                total_runtime = float(sum(run_times))
                avg_runtime = total_runtime / float(num_runs)
                avg_time_s = avg_runtime / float(inferences_per_run) if inferences_per_run > 0 else 0.0
                avg_time_ms = avg_time_s * 1000.0

                runtime_stats = {
                    "total_runtime": total_runtime,
                    "avg_runtime": avg_runtime,
                    "total_runs": num_runs,
                    "inferences_per_run": inferences_per_run,
                    "avg_time_per_inference_s": avg_time_s,
                    "avg_time_per_inference_ms": avg_time_ms,
                }

                # Write config.txt inside <output_dir>/tests/<combo_folder>/
                write_config_txt(
                    path=combo_dir / "config.txt",
                    pre_display=pre_display,
                    pre_details=pre_details,
                    model_name=model_name,
                    model_raw_path=raw_model_path,
                    post_display=post_display,
                    post_details=post_details,
                    dataset_names=dataset_names,
                    seed=seed,
                    num_runs=num_runs,
                    warmup_runs=warmup_runs,
                    runtime_stats=runtime_stats,
                )

                # Write per-dataset reports inside <output_dir>/tests/<combo_folder>/<dataset_name>/
                dataset_metrics: dict[str, dict] = {}
                for ds_name, ds_info in loaded_datasets.items():
                    ds_total_rt = float(sum(dataset_run_times[ds_name]))
                    ds_avg_rt = ds_total_rt / float(num_runs)
                    ds_num_imgs = ds_info["num_images"]
                    ds_avg_inf_s = ds_avg_rt / float(ds_num_imgs) if ds_num_imgs > 0 else 0.0

                    extra_metrics = {
                        "preprocessing": pre_display,
                        "preprocessing_details": pre_details,
                        "model": model_name,
                        "postprocessing": post_display,
                        "postprocessing_details": post_details,
                        "dataset": ds_name,
                        "seed": seed,
                        "warmup_runs": warmup_runs,
                        "num_runs": num_runs,
                        "dataset_total_runtime_s": ds_total_rt,
                        "dataset_avg_runtime_s": ds_avg_rt,
                        "dataset_avg_time_per_inference_s": ds_avg_inf_s,
                    }

                    metrics = single_test.write_reports(
                        output_directory=combo_dir / ds_name,
                        true_labels=ds_info["true_labels"],
                        probabilities=saved_probabilities[ds_name],
                        file_paths=ds_info["file_paths"],
                        class_names=class_names,
                        extra_metrics=extra_metrics,
                    )
                    dataset_metrics[ds_name] = metrics
                    print(
                        f"  -> [{ds_name}] acc={metrics['accuracy']:.4f} | "
                        f"macro_f1={metrics['f1_macro']:.4f} | avg_time={ds_avg_rt:.4f}s",
                        flush=True,
                    )

                combination_records.append({
                    "pre_display": pre_display,
                    "model_name": model_name,
                    "post_display": post_display,
                    "dataset_metrics": dataset_metrics,
                    "runtime": runtime_stats,
                })

                # Flush top-level summary tables after every combination so progress is always saved
                write_summary_tables(
                    output_dir=output_dir,
                    dataset_names=dataset_names,
                    combination_records=combination_records,
                )

        del model
        if convae_model is not None:
            del convae_model
        clear_memory()

    print(f"\nAll {len(combination_records)} combinations finished!", flush=True)
    print(f"Per-combination reports saved under: {tests_root}", flush=True)
    print(f"Top-level metric & runtime tables saved under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

