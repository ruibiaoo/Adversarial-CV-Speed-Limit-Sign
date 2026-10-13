"""Quick test of the preprocessing defences on the clean and attacked test sets, with fixed settings.

Nothing is tuned here: each defence uses one commonly used setting (FIXED).
Every defence runs at test time in front of the same base classifier
(phase1_model in config.yaml). The script tests each defence alone and the
defences chained together in every order (order matters: jpeg then median is
not median then jpeg). The undefended classifier is not tested here: Phase 2
already did that with src/modelling/test.py.

Run from src/defenses/:

  python modules/preprocessing/test_defences.py                       # jpeg, median, gaussian: 15 pipelines
  python modules/preprocessing/test_defences.py --defence jpeg median # only the pipelines made of these
  python modules/preprocessing/test_defences.py --model path/to/best.keras --data path/to/data_updated

Test sets: test (clean, shows what a defence costs on normal images),
pgd_attacked (white-box) and pixle_attacked (black-box).

The metrics and report files are the ones Phase 2 used to score the attacks:
the functions of src/modelling/test.py are imported and called unchanged.
Results: testing_results/preprocessing_based/<a>[_then_<b>[_then_<c>]]/<test set>/ with
classification_report.txt, test_metrics.txt, confusion_matrix.txt,
confusion_matrix_normalized.txt and predictions.csv. In the same folder, summary.csv
has one row per pipeline and test set, summary.md one ranked table. Both list
only the pipelines of the current run. A pairing whose predictions.csv already
exists is not predicted again; --rerun forces it.
"""
import csv
import importlib.util
import sys
from itertools import permutations
from pathlib import Path

PHASE3 = Path(__file__).resolve().parents[2]  # src/defenses, so `modules` is importable from here
if str(PHASE3) not in sys.path:
    sys.path.insert(0, str(PHASE3))

import numpy as np  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from modules import evaluation as E  # noqa: E402
from modules.preprocessing import defences as D  # noqa: E402

T = E.T  # shared helpers (config, data loading, write_csv)

# Fixed, untuned settings. General image-processing defaults, not taken from any attack setting.
FIXED = {
    D.JPEG: {"quality": 75},
    D.MEDIAN: {"kernel": 3},
    D.GAUSSIAN: {"sigma": 8, "samples": D.SMOOTHING_SAMPLES, "seed": D.SMOOTHING_SEED},
}
ALIASES = {"jpeg": D.JPEG, "median": D.MEDIAN, "gaussian": D.GAUSSIAN}
# clean test images, then the same images attacked white-box (PGD) and black-box (Pixle)
TEST_SETS = ["test", "pgd_attacked", "pixle_attacked"]
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


def saved_predictions(path, class_names):
    """(true, probs) of an earlier run, read back from its predictions.csv, or None."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    true = np.array([class_names.index(r["actual_class"]) for r in rows])
    probs = np.array([[float(r[f"probability_{c}"]) for c in class_names] for r in rows])
    return true, probs


SHORT = {v: k for k, v in ALIASES.items()}


def chains(names):
    """Each single defence, then every ordered chain of 2 and 3 of them."""
    out = []
    for size in range(1, len(names) + 1):
        out += [("_then_".join(SHORT[n] for n in c), list(c)) for c in permutations(names, size)]
    return out


def write_markdown(path, summary):
    """One row per chain: accuracy and macro F1 on the three test sets, best worst-case first."""
    rows = {}
    for r in summary:
        rows.setdefault(r["defence"], {})[r["test_set"]] = r
    table = sorted(rows.items(), key=lambda kv: -min(r["accuracy"] for r in kv[1].values()))
    lines = ["# Preprocessing defences, alone and chained (fixed settings)", "",
             f"Settings: {', '.join(f'{SHORT[n]} {p}' for n, p in FIXED.items())}.",
             "Sorted by the worst of the three accuracies. Chains are applied left to right.",
             "Clean = test/ (unattacked), PGD = pgd_attacked/, Pixle = pixle_attacked/; 115 images each.", "",
             "| Rank | Chain | Defences | Clean accuracy | PGD accuracy | Pixle accuracy "
             "| Clean macro F1 | PGD macro F1 | Pixle macro F1 | Worst accuracy |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for rank, (name, r) in enumerate(table, 1):
        clean, pgd, pixle = r["test"], r["pgd_attacked"], r["pixle_attacked"]
        lines.append(f"| {rank} | {name.replace('_then_', ' → ')} | {name.count('_then_') + 1} "
                     f"| {clean['accuracy']:.4f} | {pgd['accuracy']:.4f} | {pixle['accuracy']:.4f} "
                     f"| {clean['f1_macro']:.4f} | {pgd['f1_macro']:.4f} | {pixle['f1_macro']:.4f} "
                     f"| {min(x['accuracy'] for x in (clean, pgd, pixle)):.4f} |")
    # the hand-written analysis below the table is kept when the table is regenerated
    analysis = ""
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if ANALYSIS_HEADING in previous:
            analysis = "\n" + previous[previous.index(ANALYSIS_HEADING):]
    path.write_text("\n".join(lines) + "\n" + analysis, encoding="utf-8")


def main():
    parser = E.parser_for(__doc__.splitlines()[0], list(FIXED), ALIASES)
    parser.add_argument("--rerun", action="store_true",
                        help="predict again even if a pairing already has predictions.csv")
    args = parser.parse_args()
    names = E.selected(args.defence, list(FIXED), ALIASES)
    config = E.load_config(args)
    class_names = list(config["class_names"])
    out = Path(config["output_dir"]) / GROUP  # testing_results/preprocessing_based/
    out.mkdir(parents=True, exist_ok=True)

    T.setup_gpu()
    P2 = load_phase2_test()
    model = E.load_model(E.phase1_model_path(args, config))
    predictors = [(label, {SHORT[n]: FIXED[n] for n in steps},
                   D.make_chain_predictor(steps, model, FIXED)) for label, steps in chains(names)]

    summary = []
    for split in TEST_SETS:
        ds = T.load_split(split, config, shuffle=False)
        file_paths = list(ds.file_paths)
        ds = ds.batch(int(config["batch"]))
        for name, params, predict in predictors:
            saved = None if args.rerun else saved_predictions(out / name / split / "predictions.csv",
                                                              class_names)
            true, probs = saved or predict_split(predict, ds)
            metrics, report = write_reports(P2, out / name / split, true, probs, file_paths, class_names)
            summary.append({"defence": name, "setting": params, "test_set": split,
                            "n": metrics["number_of_test_images"],
                            **{m: round(metrics[m], 6) for m in SUMMARY_METRICS}})
            print(f"\n===== {name} {params} on {split} =====\n{report}", flush=True)
            T.write_csv(out / "summary.csv", summary)  # rewritten after every run, so progress is kept

    write_markdown(out / "summary.md", summary)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
