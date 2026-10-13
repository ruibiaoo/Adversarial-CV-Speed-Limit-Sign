# CLAUDE.md

Adversarial robustness project for a speed-limit sign classifier (DSA4266).
- Phase 1 trains the classifier (TF ResNet50, `src/modelling/`).
- Phase 2 attacks it: white-box PGD and black-box Pixle (`src/attack/`).
- Phase 3 defends it (`src/defenses/`; see `src/defenses/PRD_defences.md`).

Data lives in `data/data_updated/`: `train/`, `val/`, `test/`, plus the attacked test sets `pgd_attacked/` and `pixle_attacked/`.

## Rules

### No attack data or attack knowledge in defences

- Defences are trained on `train/` and tuned on clean `val/` only.
- Attacked images (`pgd_attacked/`, `pixle_attacked/`, or attacked copies of any split) are for final evaluation only.
- No defence setting may be copied from an attack setting (ε, step size, patch size). This is why adversarial training such as TRADES is excluded.
