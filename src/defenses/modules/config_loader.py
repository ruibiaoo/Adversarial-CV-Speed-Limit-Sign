"""Config loader: Phase 1 training config + Phase 3 overrides, with absolute paths."""
from pathlib import Path

import yaml

PHASE3 = Path(__file__).resolve().parents[1]
PHASE1 = PHASE3.parent / "modelling"
# Phase 1 moved its training config to configs/train.yaml; the old path is kept first.
PHASE1_CONFIGS = (PHASE1 / "config.yaml", PHASE1 / "configs" / "train.yaml")
# data_updated sits at the repo root (outside src/) when not running under Docker.
DATA_FALLBACK = PHASE3.parents[1] / "data" / "data_updated"


def load_config():
    """Phase 1 config (same recipe) + Phase 3 overrides. Paths become absolute."""
    phase1_config = next((p for p in PHASE1_CONFIGS if p.exists()), None)
    if phase1_config is None:
        raise FileNotFoundError(f"No Phase 1 config found in {[str(p) for p in PHASE1_CONFIGS]}")
    with phase1_config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with (PHASE3 / "config.yaml").open(encoding="utf-8") as f:
        config.update(yaml.safe_load(f))
    for key in ("data", "output_dir", "checkpoints_dir", "phase1_model"):
        config[key] = str((PHASE3 / config[key]).resolve())
    if not Path(config["data"]).exists() and DATA_FALLBACK.exists():
        config["data"] = str(DATA_FALLBACK)
    return config
