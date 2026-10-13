"""Test the augmentation-based defences on the clean, white-box and black-box test sets.

Each defence is a model retrained with that augmentation; its checkpoint is
the one frozen in tuned_defences.yaml by the training scripts. The Phase 1
model (no defence) is always evaluated as the baseline.

  python eval_augmentation.py                                   # all: random_erasing, mixup
  python eval_augmentation.py --defence mixup
  python eval_augmentation.py --defence erasing mixup
  python eval_augmentation.py --defence all --model path/to/phase1/best.keras --data path/to/data_processed

--defence takes the names in tuned_defences.yaml (random_erasing, mixup),
the short name erasing, or all. Results: test_results/augmentation/results.csv
and summary.csv. Rows of defences not tested in a run are kept from the previous run.
"""
from modules import evaluation as E
from modules.preprocessing import defences as D

GROUP = "augmentation"
DEFENCES = ["random_erasing", "mixup"]
ALIASES = {"erasing": "random_erasing"}


def main():
    args = E.parser_for(__doc__.splitlines()[0], DEFENCES, ALIASES).parse_args()
    names = E.selected(args.defence, DEFENCES, ALIASES)
    configs = E.frozen_configs(names)
    config = E.load_config(args)

    phase1 = E.load_model(E.phase1_model_path(args, config))
    predictors = [(E.BASELINE, {}, D.make_predictor(D.NO_DEFENCE, phase1))]
    for n in names:
        model = E.load_model(E.PHASE3 / configs[n]["checkpoint"])
        predictors.append((n, configs[n]["params"], D.make_predictor(D.NO_DEFENCE, model)))
    E.evaluate(GROUP, predictors, DEFENCES, config)


if __name__ == "__main__":
    main()
