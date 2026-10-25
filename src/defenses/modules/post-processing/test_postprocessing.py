"""Test the post-processing defences on the clean and attacked test sets, alone and chained.

Each defence uses its tuned setting from modules/post-processing/tuned_defences.yaml
(run tune_postprocessing.py first). Like the preprocessing tester, this runs each defence
alone and then every ordered chain of two and three of them (order matters because the
postprocessors compose on the output vector), in front of the same frozen base classifier.

Because these defences change the probability vector, not the image, the top-1 accuracy
on an already-attacked image barely moves; the real effect is on the probability-based
metrics a score-based attacker would read. So summary.md reports accuracy and macro F1 (as
for preprocessing) and also the cross-entropy, mean top-1 confidence and mean output entropy
on the clean split, which show how much the output is obfuscated.

The five per-pairing report files are the same as Phase 2 / the preprocessing tester
(classification_report.txt, test_metrics.txt, confusion_matrix.txt,
confusion_matrix_normalized.txt, predictions.csv). Results:
testing_results/post_processing_based/<a>[_then_<b>[_then_<c>]]/<test set>/, plus no_defence/
(the raw base softmax, as the reference row). A pairing whose predictions.csv already exists is
not predicted again if its run.json (setting, model, data folder) still matches; --rerun forces it.

Run from src/defenses/:

  python modules/post-processing/test_postprocessing.py
  python modules/post-processing/test_postprocessing.py --defence noise revsig
"""
import sys
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent              # modules/post-processing (holds defences.py)
PHASE3 = HERE.parents[1]                             # src/defenses, so `modules` is importable
for _p in (str(PHASE3), str(HERE)):                  # HERE first so the sibling import is unambiguous
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import defences as D  # noqa: E402  (sibling module in this folder)

from modules import evaluation as E  # noqa: E402
from modules.preprocessing import test_defences as TD  # noqa: E402  (reuse report helpers)

T = E.T
NO_DEFENCE = "no_defence"
GROUP = "post_processing_based"
TUNED = PHASE3 / "modules" / "post-processing" / "tuned_defences.yaml"
ANALYSIS_HEADING = "## Analysis"
# accuracy/macro-F1 plus the probability-side metrics that matter for output defences
SUMMARY_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "cross_entropy_loss",
                   "roc_auc_macro_ovr"]


def frozen_settings(names):
    """{defence: params} from tuned_defences.yaml; stops if a requested defence is not tuned yet."""
    tuned = yaml.safe_load(TUNED.read_text(encoding="utf-8")) or {} if TUNED.exists() else {}
    missing = [n for n in names if n not in tuned]
    if missing:
        raise SystemExit(f"Not tuned yet: {missing}. Run tune_postprocessing.py first; "
                         "test data is never used to choose settings.")
    return {n: tuned[n]["params"] for n in names}


def chains(names):
    """The no-defence baseline, each single defence, then every ordered chain of 2 and 3 of them
    (order matters)."""
    out = [(NO_DEFENCE, [])]  # raw base softmax: the reference row for these test sets
    for size in range(1, len(names) + 1):
        out += [("_then_".join(D.SHORT[n] for n in c), list(c)) for c in permutations(names, size)]
    return out


def size_of(name):
    """Number of defences in a chain label (0 for the no-defence baseline)."""
    return 0 if name == NO_DEFENCE else name.count("_then_") + 1


def label(name):
    return f"**{name}**" if name == NO_DEFENCE else name.replace("_then_", " → ")


def write_markdown(path, summary, splits):
    """Main table (accuracy + macro F1 per test set) plus a clean-split obfuscation table."""
    rows = {}
    for r in summary:
        rows.setdefault(r["defence"], {})[r["test_set"]] = r
    table = sorted(rows.items(), key=lambda kv: -min(r["accuracy"] for r in kv[1].values()))
    acc_cols = " ".join(f"| {s} accuracy " for s in splits)
    f1_cols = " ".join(f"| {s} macro F1 " for s in splits)
    lines = ["# Post-processing defences, alone and chained", "",
             "Sorted by the worst accuracy across the test sets. Chains are applied left to right on "
             "the output vector. Settings are tuned on val (tuned_defences.yaml). "
             f"`{NO_DEFENCE}` is the raw base softmax (the reference row).",
             f"Test sets (from data.yaml / config.yaml): {', '.join(splits)}. Mean accuracy is the "
             "image-weighted accuracy over all of them.", "",
             f"| Rank | Chain | Defences {acc_cols}{f1_cols}| Mean accuracy | Worst accuracy |",
             "|---|---|---|" + "---|" * (2 * len(splits) + 2)]
    for rank, (name, r) in enumerate(table, 1):
        accs = [r[s]["accuracy"] for s in splits]
        f1s = [r[s]["f1_macro"] for s in splits]
        mean = sum(r[s]["accuracy"] * r[s]["n"] for s in splits) / sum(r[s]["n"] for s in splits)
        cells = "".join(f" {a:.4f} |" for a in accs) + "".join(f" {f:.4f} |" for f in f1s)
        lines.append(f"| {rank} | {label(name)} | {size_of(name)} |{cells} {mean:.4f} | {min(accs):.4f} |")

    clean = splits[0]  # the (unattacked) clean split drives the obfuscation view
    lines += ["", f"## Output obfuscation on the clean split ({clean})",
              "Higher cross-entropy, lower mean confidence and higher mean entropy = more obfuscated "
              "output for a score-based attacker, at (ideally) unchanged accuracy.", "",
              "| Chain | Cross-entropy | Mean top-1 confidence | Mean output entropy |",
              "|---|---|---|---|"]
    for name, r in table:
        c = r[clean]
        lines.append(f"| {label(name)} | {c['cross_entropy_loss']:.4f} "
                     f"| {c['mean_confidence']:.4f} | {c['mean_entropy']:.4f} |")

    analysis = ""
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if ANALYSIS_HEADING in previous:
            analysis = "\n" + previous[previous.index(ANALYSIS_HEADING):]
    path.write_text("\n".join(lines) + "\n" + analysis, encoding="utf-8")


def main():
    parser = E.parser_for(__doc__.splitlines()[0], list(D.GRIDS), D.ALIASES)
    parser.add_argument("--rerun", action="store_true",
                        help="predict again even if a pairing already has predictions.csv")
    args = parser.parse_args()
    names = E.selected(args.defence, list(D.GRIDS), D.ALIASES)
    config = E.load_config(args)
    class_names = list(config["class_names"])
    settings = frozen_settings(names)
    out = Path(config["output_dir"]) / GROUP
    out.mkdir(parents=True, exist_ok=True)

    T.setup_gpu()
    P2 = TD.load_phase2_test()
    model_path = E.phase1_model_path(args, config)
    model = E.load_model(model_path)
    predictors = [(name, {D.SHORT[n]: settings[n] for n in steps},
                   D.make_chain_predictor(steps, model, settings)) for name, steps in chains(names)]

    summary = []
    splits = list(config["test_sets"].values())
    for split in splits:
        ds = T.load_split(split, config, shuffle=False)
        file_paths = list(ds.file_paths)
        ds = ds.batch(int(config["batch"]))
        for name, params, predict in predictors:
            key = TD.run_key(params, model_path, Path(config["data"]) / split)
            saved = None if args.rerun else TD.saved_predictions(
                out / name / split / "predictions.csv", class_names, key)
            true, probs = saved or TD.predict_split(predict, ds)
            metrics, report = TD.write_reports(P2, out / name / split, true, probs, file_paths,
                                               class_names)
            TD.save_run_key(out / name / split, key)
            summary.append({"defence": name, "setting": params, "test_set": split,
                            "n": metrics["number_of_test_images"],
                            **{m: round(metrics[m], 6) for m in SUMMARY_METRICS},
                            "mean_confidence": round(float(np.mean(probs.max(axis=1))), 6),
                            "mean_entropy": round(float(np.mean(D.entropy(probs))), 6)})
            print(f"\n===== {name} {params} on {split} =====\n{report}", flush=True)
            T.write_csv(out / "summary.csv", summary)

    write_markdown(out / "summary.md", summary, splits)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
