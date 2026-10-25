"""Test the preprocessing defences on the clean and attacked test sets.

Each defence uses its tuned setting from modules/preprocessing/tuned_defences.yaml
if tune_preprocessing.py has been run; otherwise it falls back to one commonly
used default (FIXED). The settings source is printed at startup.
Every defence runs at test time in front of the same base classifier
(phase1_model in config.yaml). The script tests each defence alone and the
defences chained together in every order (order matters: jpeg then median is
not median then jpeg). The undefended classifier (no_defence) is tested too, as
the reference row: the same base model on the same test sets, with no filter.

Run from src/defenses/:

  python modules/preprocessing/test_defences.py                       # no_defence + 15 pipelines
  python modules/preprocessing/test_defences.py --defence jpeg median # only the pipelines made of these
  python modules/preprocessing/test_defences.py --model path/to/best.keras --data path/to/data_root

Test sets: config.yaml `test_sets` — the clean test split (shows what a defence costs on
normal images), a white-box PGD copy and a black-box Pixle copy of it.

The metrics and report files are the ones Phase 2 used to score the attacks:
the functions of src/modelling/test.py are imported and called unchanged.
Results: testing_results/preprocessing_based/<a>[_then_<b>[_then_<c>]]/<test set>/ with
classification_report.txt, test_metrics.txt, confusion_matrix.txt,
confusion_matrix_normalized.txt and predictions.csv. In the same folder, summary.csv
has one row per pipeline and test set, summary.md one ranked table. Both list
only the pipelines of the current run. A pairing whose predictions.csv already
exists is not predicted again if its run.json (setting, model, data folder) still
matches; --rerun forces it.
"""
import csv
import importlib.util
import json
import sys
from itertools import permutations
from pathlib import Path

PHASE3 = Path(__file__).resolve().parents[2]  # src/defenses, so `modules` is importable from here
if str(PHASE3) not in sys.path:
    sys.path.insert(0, str(PHASE3))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from modules import evaluation as E  # noqa: E402
from modules.preprocessing import defences as D  # noqa: E402

T = E.T  # shared helpers (config, data loading, write_csv)

# Fixed, untuned defaults. General image-processing values, not taken from any attack setting.
# tune_preprocessing.py may override these per defence (see resolve_settings).
FIXED = {
    D.JPEG: {"quality": 75},
    D.MEDIAN: {"kernel": 3},
    D.GAUSSIAN: {"sigma": 8, "samples": D.SMOOTHING_SAMPLES, "seed": D.SMOOTHING_SEED},
}
ALIASES = {"jpeg": D.JPEG, "median": D.MEDIAN, "gaussian": D.GAUSSIAN}
GROUP = "preprocessing_based"  # results folder inside testing_results/
ANALYSIS_HEADING = "## Analysis"  # summary.md: everything from this heading on is written by hand
SUMMARY_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro",
                   "matthews_correlation_coefficient", "cross_entropy_loss"]


def load_phase2_test():
    """src/modelling/test.py as a module, so its metric and report code is reused as is."""
    path = E.PHASE3.parent / "modelling" / "test.py"
    spec = importlib.util.spec_from_file_location("phase2_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_reports(P2, out, true, probs, file_paths, class_names):
    """Same five files, in the same format, as main() of src/modelling/test.py."""
    out.mkdir(parents=True, exist_ok=True)
    pred = np.argmax(probs, axis=1)
    metrics, per_class, matrix = P2.calculate_metrics(true, pred, probs, class_names)

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
                           where=row_totals != 0)
    P2.save_confusion_matrix(out / "confusion_matrix.txt", matrix, class_names)
    P2.save_confusion_matrix(out / "confusion_matrix_normalized.txt", normalized, class_names,
                             normalized=True)
    P2.save_metrics(out / "test_metrics.txt", metrics, per_class)
    report = classification_report(true, pred, labels=list(range(len(class_names))),
                                   target_names=class_names, digits=6, zero_division=0)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    P2.save_predictions(out / "predictions.csv", file_paths, true, pred, probs, class_names)
    return metrics, report


def predict_split(predict, ds):
    """Run predict over a batched (image, label) dataset. Images stay float 0-255, as in test.py."""
    true, probs = [], []
    for x, y in ds:
        probs.append(predict(x.numpy()))
        true.extend(y.numpy())
    return np.asarray(true), np.concatenate(probs)


RUN_FILE = "run.json"  # next to predictions.csv: the setting, model and data folder it was made with


def run_key(params, model_path, folder):
    """What a saved prediction depends on, JSON-normalised so it compares equal after a round trip."""
    key = {"setting": params, "model": str(model_path), "data": str(folder)}
    return json.loads(json.dumps(key, sort_keys=True, default=str))


def save_run_key(out, key):
    (out / RUN_FILE).write_text(json.dumps(key, sort_keys=True, indent=1), encoding="utf-8")


def saved_predictions(path, class_names, key=None):
    """(true, probs) of an earlier run, read back from its predictions.csv, or None.

    With key, the earlier run is reused only if its run.json matches (same setting, model and data
    folder), so re-tuning a defence or switching the model/data never reuses stale predictions.
    """
    if not path.exists():
        return None
    if key is not None:
        run = path.parent / RUN_FILE
        if not run.exists() or json.loads(run.read_text(encoding="utf-8")) != key:
            return None
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    true = np.array([class_names.index(r["actual_class"]) for r in rows])
    probs = np.array([[float(r[f"probability_{c}"]) for c in class_names] for r in rows])
    return true, probs


SHORT = {v: k for k, v in ALIASES.items()}


def size_of(name):
    """Number of defences in a chain label (0 for the no-defence baseline)."""
    return 0 if name == D.NO_DEFENCE else name.count("_then_") + 1


def chains(names):
    """The no-defence baseline, each single defence, then every ordered chain of 2 and 3 of them."""
    out = [(D.NO_DEFENCE, [])]  # same base model, no filter: the reference row for these test sets
    for size in range(1, len(names) + 1):
        out += [("_then_".join(SHORT[n] for n in c), list(c)) for c in permutations(names, size)]
    return out


def write_markdown(path, summary, splits, settings):
    """One row per chain: accuracy and macro F1 on each test set, best worst-case first.

    splits: the test-set folder names in column order (from config["test_sets"], so the clean
    split follows data.yaml). settings: the per-defence params actually used (tuned or FIXED).
    """
    rows = {}
    for r in summary:
        rows.setdefault(r["defence"], {})[r["test_set"]] = r
    table = sorted(rows.items(), key=lambda kv: -min(r["accuracy"] for r in kv[1].values()))
    acc_cols = " ".join(f"| {s} accuracy " for s in splits)
    f1_cols = " ".join(f"| {s} macro F1 " for s in splits)
    header = f"| Rank | Chain | Defences {acc_cols}{f1_cols}| Mean accuracy | Worst accuracy |"
    lines = ["# Preprocessing defences, alone and chained", "",
             f"Settings: {', '.join(f'{SHORT[n]} {settings[n]}' for n in FIXED)}.",
             "Sorted by the worst accuracy across the test sets. Chains are applied left to right. "
             f"`{D.NO_DEFENCE}` is the same base model with no filter (the reference row).",
             f"Test sets (from data.yaml / config.yaml): {', '.join(splits)}. Mean accuracy is the "
             "image-weighted accuracy over all of them.", "",
             header, "|---|---|---|" + "---|" * (2 * len(splits) + 2)]
    for rank, (name, r) in enumerate(table, 1):
        accs = [r[s]["accuracy"] for s in splits]
        f1s = [r[s]["f1_macro"] for s in splits]
        mean = sum(r[s]["accuracy"] * r[s]["n"] for s in splits) / sum(r[s]["n"] for s in splits)
        cells = "".join(f" {a:.4f} |" for a in accs) + "".join(f" {f:.4f} |" for f in f1s)
        label = f"**{name}**" if name == D.NO_DEFENCE else name.replace("_then_", " → ")
        lines.append(f"| {rank} | {label} | {size_of(name)} |{cells} {mean:.4f} | {min(accs):.4f} |")
    # the hand-written analysis below the table is kept when the table is regenerated
    analysis = ""
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if ANALYSIS_HEADING in previous:
            analysis = "\n" + previous[previous.index(ANALYSIS_HEADING):]
    path.write_text("\n".join(lines) + "\n" + analysis, encoding="utf-8")


def resolve_settings():
    """Per-defence params: the tuned value from tuned_defences.yaml if present, else the FIXED default.

    tune_preprocessing.py freezes {defence: {params: {...}, ...}} there. For Gaussian the tuned grid
    only stores sigma, so the FIXED samples/seed are kept as defaults under it.
    """
    settings = {name: dict(params) for name, params in FIXED.items()}
    tuned = {}
    if T.TUNED_PREPROCESSING.exists():
        tuned = yaml.safe_load(T.TUNED_PREPROCESSING.read_text(encoding="utf-8")) or {}
    for name in settings:
        if name in tuned and isinstance(tuned[name].get("params"), dict):
            settings[name] = {**settings[name], **tuned[name]["params"]}
    return settings, bool(tuned)


def main():
    parser = E.parser_for(__doc__.splitlines()[0], list(FIXED), ALIASES)
    parser.add_argument("--rerun", action="store_true",
                        help="predict again even if a pairing already has predictions.csv")
    args = parser.parse_args()
    names = E.selected(args.defence, list(FIXED), ALIASES)
    config = E.load_config(args)
    class_names = list(config["class_names"])
    settings, tuned = resolve_settings()
    print(f"Settings source: {'tuned_defences.yaml' if tuned else 'FIXED defaults'}", flush=True)
    out = Path(config["output_dir"]) / GROUP  # testing_results/preprocessing_based/
    out.mkdir(parents=True, exist_ok=True)

    T.setup_gpu()
    P2 = load_phase2_test()
    model_path = E.phase1_model_path(args, config)
    model = E.load_model(model_path)
    predictors = [(label, {SHORT[n]: settings[n] for n in steps},
                   D.make_chain_predictor(steps, model, settings)) for label, steps in chains(names)]

    summary = []
    for split in list(config["test_sets"].values()):
        ds = T.load_split(split, config, shuffle=False)
        file_paths = list(ds.file_paths)
        ds = ds.batch(int(config["batch"]))
        for name, params, predict in predictors:
            key = run_key(params, model_path, Path(config["data"]) / split)
            saved = None if args.rerun else saved_predictions(out / name / split / "predictions.csv",
                                                              class_names, key)
            true, probs = saved or predict_split(predict, ds)
            metrics, report = write_reports(P2, out / name / split, true, probs, file_paths, class_names)
            save_run_key(out / name / split, key)
            summary.append({"defence": name, "setting": params, "test_set": split,
                            "n": metrics["number_of_test_images"],
                            **{m: round(metrics[m], 6) for m in SUMMARY_METRICS}})
            print(f"\n===== {name} {params} on {split} =====\n{report}", flush=True)
            T.write_csv(out / "summary.csv", summary)  # rewritten after every run, so progress is kept

    write_markdown(out / "summary.md", summary, list(config["test_sets"].values()), settings)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
