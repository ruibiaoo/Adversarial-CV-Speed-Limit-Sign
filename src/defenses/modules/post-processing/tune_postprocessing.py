"""Tune the post-processing defences on the validation split, then freeze the winners.

These defences perturb the output probability vector, not the image, so the top-1 class
(hence accuracy / macro F1) barely moves. "Best val macro F1" would therefore pick the
mildest setting every time and tell us nothing. Instead, for each defence we pick the
setting that distorts the output the most (obfuscation = mean top-1 confidence drop +
mean output-entropy increase vs the undefended base) among the settings whose val macro F1
stays within a tolerance tau of the undefended baseline. See README.md for why this is the
right objective for output-side defences.

Only the val split is read (point data.yaml's `val:` at an attacked-val folder to tune
against an attack; the test splits are never read here). The base softmax is computed once
and the postprocessors are applied to it, so the grid sweep needs a single model pass.

Chosen settings are written to modules/post-processing/tuned_defences.yaml; per-setting
scores are logged to tuning_logs/post-processing/<defence>/grid.csv.

Run from src/defenses/:

  python modules/post-processing/tune_postprocessing.py                     # all three defences
  python modules/post-processing/tune_postprocessing.py --defence noise     # one defence
  python modules/post-processing/tune_postprocessing.py --tolerance 0.02
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent              # modules/post-processing (holds defences.py)
PHASE3 = HERE.parents[1]                             # src/defenses, so `modules` is importable
for _p in (str(PHASE3), str(HERE)):                  # HERE first so the sibling import is unambiguous
    if _p not in sys.path:
        sys.path.insert(0, _p)

import defences as D  # noqa: E402  (sibling module in this folder)

from modules import evaluation as E  # noqa: E402

T = E.T  # shared helpers (_trainer: load_split, metrics, write_csv, save_choice, setup_gpu)
TUNED = PHASE3 / "modules" / "post-processing" / "tuned_defences.yaml"


def obfuscation_score(row):
    return row["confidence_drop"] + row["entropy_increase"]


def tune_one(defence, true, base_probs, baseline_f1, tolerance, val, file_paths):
    """Sweep defence's grid on the precomputed base val probs; freeze and return the chosen setting."""
    rows = []
    for params in D.GRIDS[defence]:  # mild -> strong
        defended = D.apply_chain(base_probs, [defence], {defence: params})
        m = T.metrics(true, defended)
        obf = D.obfuscation_metrics(base_probs, defended)
        rows.append({**params, "val_accuracy": round(m["accuracy"], 4),
                     "val_macro_f1": round(m["macro_f1"], 4),
                     "confidence_drop": round(obf["confidence_drop"], 4),
                     "entropy_increase": round(obf["entropy_increase"], 4),
                     **E.per_source_accuracy(file_paths, true, defended)})
        print(f"[{defence}] {params} on {val}: val_macro_f1={m['macro_f1']:.4f} "
              f"conf_drop={obf['confidence_drop']:.4f} entropy+={obf['entropy_increase']:.4f}", flush=True)

    T.write_csv(T.PHASE3 / "tuning_logs" / "post-processing" / defence / "grid.csv", rows)
    # strongest obfuscation among settings within tolerance of the baseline; >= keeps the stronger
    # (later in the mild->strong grid) setting on ties.
    candidates = [r for r in rows if r["val_macro_f1"] >= baseline_f1 - tolerance] or rows
    chosen = candidates[0]
    for r in candidates[1:]:
        if obfuscation_score(r) >= obfuscation_score(chosen):
            chosen = r

    grid_keys = list(D.GRIDS[defence][0])
    T.save_choice(defence, {
        "params": {k: chosen[k] for k in grid_keys},
        "val_accuracy": chosen["val_accuracy"],
        "val_macro_f1": chosen["val_macro_f1"],
        "confidence_drop": chosen["confidence_drop"],
        "entropy_increase": chosen["entropy_increase"],
        "baseline_val_macro_f1": round(baseline_f1, 4),
        "tolerance": tolerance,
        "selected_on": f"max output obfuscation with val macro F1 within {tolerance} of baseline on '{val}'",
    }, TUNED)
    print(f"Chosen {defence}: { {k: chosen[k] for k in grid_keys} } "
          f"(val_macro_f1={chosen['val_macro_f1']:.4f}, obfuscation={obfuscation_score(chosen):.4f})",
          flush=True)
    return chosen


def main():
    parser = E.parser_for(__doc__.splitlines()[0], list(D.GRIDS), D.ALIASES)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="max val macro-F1 drop below baseline a setting may cost (default 0.01)")
    args = parser.parse_args()
    names = E.selected(args.defence, list(D.GRIDS), D.ALIASES)
    config = E.load_config(args)
    val = config["splits"]["val"]

    T.setup_gpu()
    model = E.load_model(E.phase1_model_path(args, config))
    ds = T.load_split(val, config, shuffle=False)
    file_paths = list(ds.file_paths)
    true, base_probs = E.predict_split(D.base_classify(model), ds.batch(int(config["batch"])))
    baseline_f1 = T.metrics(true, base_probs)["macro_f1"]
    print(f"Baseline (no defence) val macro F1 on '{val}': {baseline_f1:.4f} "
          f"{E.per_source_accuracy(file_paths, true, base_probs)}", flush=True)

    for defence in names:
        tune_one(defence, true, base_probs, baseline_f1, args.tolerance, val, file_paths)
    print(f"Frozen settings written to {TUNED}", flush=True)


if __name__ == "__main__":
    main()
