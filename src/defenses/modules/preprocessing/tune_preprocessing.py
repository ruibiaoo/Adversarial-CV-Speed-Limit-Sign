"""Tune the preprocessing defences on the validation split, then freeze the winners.

For each defence (jpeg, median, gaussian) this sweeps the strength-ordered grid in
defences.py (GRIDS), runs the defence in front of the base classifier on the val split
named in data.yaml, and keeps the setting with the best val macro F1. Ties go to the
milder setting (the grid is ordered mild -> strong and the first maximum wins), so a
defence is never made stronger than the val evidence supports.

Only the val split is read. Point data.yaml's `val:` at an attacked-val folder to tune
against an attack; the test splits are never read here.

The chosen setting per defence is written to modules/preprocessing/tuned_defences.yaml,
which test_defences.py then uses. Per-setting val scores are logged to
tuning_logs/preprocessing/<defence>/grid.csv.

Run from src/defenses/:

  python modules/preprocessing/tune_preprocessing.py                 # all three defences
  python modules/preprocessing/tune_preprocessing.py --defence jpeg  # one defence
  python modules/preprocessing/tune_preprocessing.py --model path/to/best.keras --data path/to/data_updated
"""
import sys
from pathlib import Path

PHASE3 = Path(__file__).resolve().parents[2]  # src/defenses, so `modules` is importable from here
if str(PHASE3) not in sys.path:
    sys.path.insert(0, str(PHASE3))

from modules import evaluation as E  # noqa: E402
from modules.preprocessing import defences as D  # noqa: E402

T = E.T  # shared helpers (_trainer: config, data loading, metrics, write_csv, save_choice)

ALIASES = {"jpeg": D.JPEG, "median": D.MEDIAN, "gaussian": D.GAUSSIAN}


def tune_one(defence, model, config):
    """Sweep defence's grid on the val split; freeze and return the best-val-macro-F1 setting."""
    val = config["splits"]["val"]
    ds = T.load_split(val, config, shuffle=False).batch(int(config["batch"]))
    rows = []
    for params in D.GRIDS[defence]:  # mild -> strong
        true, probs = E.predict_split(D.make_predictor(defence, model, params), ds)
        m = T.metrics(true, probs)
        rows.append({**params, "val_accuracy": round(m["accuracy"], 4),
                     "val_macro_f1": round(m["macro_f1"], 4), "val_loss": round(m["loss"], 4)})
        print(f"[{defence}] {params} on {val}: val_acc={m['accuracy']:.4f} "
              f"val_macro_f1={m['macro_f1']:.4f}", flush=True)

    T.write_csv(T.PHASE3 / "tuning_logs" / "preprocessing" / defence / "grid.csv", rows)
    best = max(rows, key=lambda r: r["val_macro_f1"])  # first max wins ties -> milder setting
    grid_keys = list(D.GRIDS[defence][0])
    T.save_choice(defence, {
        "params": {k: best[k] for k in grid_keys},
        "val_accuracy": best["val_accuracy"],
        "val_macro_f1": best["val_macro_f1"],
        "selected_on": f"val macro F1 on '{val}' (ties -> milder)",
    }, T.TUNED_PREPROCESSING)
    print(f"Chosen {defence}: { {k: best[k] for k in grid_keys} } "
          f"(val_macro_f1={best['val_macro_f1']:.4f})", flush=True)
    return best


def main():
    parser = E.parser_for(__doc__.splitlines()[0], list(D.GRIDS), ALIASES)
    args = parser.parse_args()
    names = E.selected(args.defence, list(D.GRIDS), ALIASES)
    config = E.load_config(args)

    T.setup_gpu()
    model = E.load_model(E.phase1_model_path(args, config))
    for defence in names:
        tune_one(defence, model, config)
    print(f"Frozen settings written to {T.TUNED_PREPROCESSING}", flush=True)


if __name__ == "__main__":
    main()
