# CLAUDE.md

Adversarial robustness project for a speed-limit sign classifier (DSA4266).
- Phase 1 trains the classifier (TF ResNet50, `src/Phase_1/`).
- Phase 2 attacks it: white-box PGD and black-box Pixle (`src/Phase_2/`).
- Phase 3 defends it (`src/Phase_3/`; see `src/Phase_3/PRD_defences.md`).

Data lives in `data/data_processed/`: `train/`, `val/`, `test/`, plus the attacked test sets `pgd_attacked/` and `pixle_attacked/`.

## Rules

### Every defence group gets its own output directories

Any code that tunes a defence, or produces results for a defence, must create its own directories for those files and write only there. Never write into another group's folder or straight into `checkpoints/` or `tuning_logs/`. All paths below are inside `src/Phase_3/`.

| What | Directory |
|---|---|
| Tuning runs: checkpoints, per-run grids, logs | `checkpoints/<group>/` |
| Tuning record: grids, histories, summaries, loss reports | `tuning_logs/<group>/` |
| Final test results | `test_results/<group>/` |
| Frozen chosen settings | `tuned_defences.yaml` (one entry per defence) |

- Create the directories in the script, e.g. `Path(...).mkdir(parents=True, exist_ok=True)`, or `mkdir -p` in shell scripts. Do not assume they already exist.
- Current groups:
  - `augmentation`: `checkpoints/augmentation/`, `tuning_logs/augmentation/`
  - `preprocessing`: `checkpoints/preprocessing/`, `tuning_logs/preprocessing/`
- A new defence group follows the same pattern. Add it to the output-directory table in `src/Phase_3/PRD_defences.md` (section 5.1) and to the Outputs section of `src/Phase_3/README.md`.

### No attack data or attack knowledge in defences

- Defences are trained on `train/` and tuned on clean `val/` only.
- Attacked images (`pgd_attacked/`, `pixle_attacked/`, or attacked copies of any split) are for final evaluation only.
- No defence setting may be copied from an attack setting (ε, step size, patch size). This is why adversarial training such as TRADES is excluded.
