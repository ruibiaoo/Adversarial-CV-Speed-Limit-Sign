"""Shared test-set evaluation for eval_preprocessing.py and eval_augmentation.py.

Test sets (config.yaml test_sets): clean test/, white-box pgd_attacked/,
black-box pixle_attacked/. Only the evaluation scripts read them; no training
or tuning script does. Every defence setting comes from tuned_defences.yaml,
and evaluate() stops if a requested defence has not been tuned yet.

Selection (Phase 3 brief): worst case = min(clean, white-box, black-box)
accuracy per defence; the defence with the largest worst case wins. The no
defence / Phase 1 row is the reference baseline, not a candidate.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

PHASE3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE3 / "modules" / "augmentation-based"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import _trainer as T  # noqa: E402

BASELINE = "no_defence"
PIXLE_PREFIX = re.compile(r"^\d{4}_")  # pixle_attacked files are NNNN_<clean stem>.jpg


def load_model(path):
    from modules.model import load_classifier

    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    return load_classifier(str(path))


def predict_split(predict, ds):
    """Run predict(uint8 batch) over a batched (image, label) dataset."""
    true, probs = [], []
    for x, y in ds:
        probs.append(predict(x.numpy().astype(np.uint8)))
        true.extend(y.numpy())
    return true, np.concatenate(probs)


def parser_for(description, defences, aliases):
    """CLI shared by both evaluation scripts. --defence picks which defences to test."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--defence", nargs="+", default=["all"],
                        choices=["all", *defences, *aliases],
                        help="defences to evaluate (default: all). No defence is always included.")
    parser.add_argument("--model", help="Phase 1 model (default: phase1_model in config.yaml)")
    parser.add_argument("--data", help="data root holding the test folders (default: config.yaml)")
    return parser


def selected(requested, defences, aliases):
    """Resolve --defence values (names, aliases or 'all') to defence names in fixed order."""
    if "all" in requested:
        return list(defences)
    names = {aliases.get(r, r) for r in requested}
    return [d for d in defences if d in names]


def frozen_configs(names):
    configs = {}
    if T.TUNED_DEFENCES.exists():
        configs = yaml.safe_load(T.TUNED_DEFENCES.read_text(encoding="utf-8")) or {}
    missing = [n for n in names if n not in configs]
    if missing:
        raise SystemExit(f"Not tuned yet: {missing}. Tune or train them first; "
                         "test data is never used to choose settings.")
    return configs


def stem(path):
    return PIXLE_PREFIX.sub("", Path(path).stem)


def load_test_set(folder, config):
    ds = T.load_split(folder, config, shuffle=False)
    return ds.batch(int(config["batch"])), [stem(p) for p in ds.file_paths]


def wilson(correct, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar(base_ok, ok):
    """Exact two-sided McNemar p-value on paired correctness."""
    b = int(np.sum(base_ok & ~ok))
    c = int(np.sum(~base_ok & ok))
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_config(args):
    config = T.load_config()
    if args.data:
        config["data"] = str(Path(args.data).resolve())
    return config


def phase1_model_path(args, config):
    return Path(args.model).resolve() if args.model else Path(config["phase1_model"])


def evaluate(group, predictors, order, config):
    """Run [(name, params, predict)] (baseline first) on the 3 test sets.

    Writes test_results/<group>/results.csv and summary.csv. Rows of defences
    not evaluated in this call are kept from the previous results.csv, so
    `--defence median` updates only that defence (and the baseline).
    order: every defence name of the group, for a stable row order.
    """
    out = Path(config["output_dir"]) / group  # test_results/<group>/
    out.mkdir(parents=True, exist_ok=True)
    T.setup_gpu()
    test_sets = {cond: load_test_set(folder, config) for cond, folder in config["test_sets"].items()}
    attacked_stems = set().union(*(stems for cond, (_, stems) in test_sets.items() if cond != "clean"))

    results, correct = [], {}
    for name, params, predict in predictors:
        for cond, (ds, stems) in test_sets.items():
            true, probs = predict_split(predict, ds)
            ok = np.argmax(probs, axis=1) == np.asarray(true)
            correct[name, cond] = ok
            m = T.metrics(true, probs)
            lo, hi = wilson(int(ok.sum()), len(ok))
            row = {"defence": name, "params": params, "test_set": cond, "n": len(ok),
                   "accuracy": round(m["accuracy"], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                   "macro_f1": round(m["macro_f1"], 4),
                   "mcnemar_p_vs_no_defence": (round(mcnemar(correct[BASELINE, cond], ok), 4)
                                               if name != BASELINE else ""),
                   "accuracy_on_attacked_subset": ""}
            if cond == "clean":  # same images as the attacked sets (Phase 1 clean-correct only)
                subset = np.array([s in attacked_stems for s in stems])
                if subset.any():
                    row["accuracy_on_attacked_subset"] = round(float(ok[subset].mean()), 4)
            results.append(row)
            print(f"{name:20s} {cond:10s} acc={m['accuracy']:.4f} [{lo:.3f}, {hi:.3f}] "
                  f"macro_f1={m['macro_f1']:.4f} n={len(ok)}", flush=True)

    results = merge_previous(out / "results.csv", results, [BASELINE, *order])
    summary = summarise(results)
    T.write_csv(out / "results.csv", results)
    T.write_csv(out / "summary.csv", summary)
    best = next((s for s in summary if s["chosen"]), None)
    if best:
        print(f"\nChosen ({group}): {best['defence']}  worst case = {best['worst_case']:.4f}")
    print(f"Results written to {out}")
    return summary


def merge_previous(path, results, order):
    """New rows replace old rows of the same defence; other old rows are kept."""
    if path.exists():
        done = {r["defence"] for r in results}
        with path.open(encoding="utf-8") as f:
            results = [r for r in csv.DictReader(f) if r["defence"] not in done] + results
    rank = {name: i for i, name in enumerate(order)}
    return sorted(results, key=lambda r: rank.get(r["defence"], len(rank)))


def summarise(results):
    summary = []
    for name in dict.fromkeys(r["defence"] for r in results):
        rows = {r["test_set"]: r for r in results if r["defence"] == name}
        acc = {cond: float(r["accuracy"]) for cond, r in rows.items()}
        summary.append({"defence": name, **acc,
                        "clean_on_attacked_subset": rows.get("clean", {}).get("accuracy_on_attacked_subset", ""),
                        "worst_case": min(acc.values()), "chosen": ""})
    candidates = [s for s in summary if s["defence"] != BASELINE]
    if candidates:
        max(candidates, key=lambda s: s["worst_case"])["chosen"] = "yes"  # first max wins ties
    return summary
