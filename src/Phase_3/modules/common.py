"""Shared helpers: config loading and the Phase 1 imports."""
from pathlib import Path

import yaml

PHASE3 = Path(__file__).resolve().parents[1]
PHASE1 = PHASE3.parent / "Phase_1"


def load_config():
    """Phase 1 config (same recipe) + Phase 3 overrides. Paths become absolute."""
    with (PHASE1 / "config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with (PHASE3 / "config.yaml").open(encoding="utf-8") as f:
        config.update(yaml.safe_load(f))
    for key in ("data", "output_dir", "runs_dir", "phase1_model"):
        config[key] = str((PHASE3 / config[key]).resolve())
    return config
