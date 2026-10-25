# CLAUDE.md

Adversarial robustness project for a speed-limit sign classifier (DSA4266).
- Phase 1 trains the classifier (TF ResNet50, `src/modelling/`).
- Phase 2 attacks it: white-box PGD and black-box Pixle (`src/attack/`).
- Phase 3 defends it (`src/defenses/`; see `src/defenses/PRD_defences.md`).

Data lives in `data/data_updated/`: `train/`, `val/`, `test/`, plus the attacked test sets `pgd_attacked/` and `pixle_attacked/`.
Attacked copies of every split are in `data/data_updated_pgd/` and `data/data_updated_pixle/`; the defences read `data/data_clean_pgd_pixle/` (clean + PGD + Pixle train/val per class, separate `test/`, `test_pgd/`, `test_pixle/`; see `src/defenses/PRD_defences.md` §3.2).

## Rules

### No attack knowledge in defences; attacked test data for final evaluation only

- Defences may be tuned on a val split that includes attacked copies of `val/` (PRD §3.1/§3.2).
- Attacked test images (`test_pgd/`, `test_pixle/`, `pgd_attacked/`, `pixle_attacked/`) are for final evaluation only; no training or tuning script may read them.
- No defence setting may be copied from an attack setting (ε, step size, patch size). This is why adversarial training such as TRADES is excluded.
