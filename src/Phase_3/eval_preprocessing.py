"""Test the preprocessing defences on the clean, white-box and black-box test sets.

Each defence runs at test time in front of the same frozen Phase 1 model, with
the setting frozen in tuned_defences.yaml by tune_preprocessing.py. No defence
is always evaluated as the baseline. With all three defences this is the
3 test sets x 4 options = 12 pairings of the Phase 3 brief.

  python eval_preprocessing.py                                  # all: jpeg, gaussian, median
  python eval_preprocessing.py --defence jpeg
  python eval_preprocessing.py --defence gaussian median
  python eval_preprocessing.py --defence all --model path/to/best.keras --data path/to/data_processed

--defence takes the short names (jpeg, gaussian, median), the full names in
tuned_defences.yaml (jpeg_compression, gaussian_smoothing, median_smoothing)
or all. Results: test_results/preprocessing/results.csv and summary.csv.
Rows of defences not tested in a run are kept from the previous run.
"""
from modules import evaluation as E
from modules.preprocessing import defences as D

GROUP = "preprocessing"
DEFENCES = [D.JPEG, D.GAUSSIAN, D.MEDIAN]
ALIASES = {"jpeg": D.JPEG, "gaussian": D.GAUSSIAN, "median": D.MEDIAN}


def main():
    args = E.parser_for(__doc__.splitlines()[0], DEFENCES, ALIASES).parse_args()
    names = E.selected(args.defence, DEFENCES, ALIASES)
    configs = E.frozen_configs(names)
    config = E.load_config(args)

    model = E.load_model(E.phase1_model_path(args, config))
    predictors = [(E.BASELINE, {}, D.make_predictor(D.NO_DEFENCE, model))]
    predictors += [(n, configs[n]["params"], D.make_predictor(n, model, configs[n]["params"]))
                   for n in names]
    E.evaluate(GROUP, predictors, DEFENCES, config)


if __name__ == "__main__":
    main()
