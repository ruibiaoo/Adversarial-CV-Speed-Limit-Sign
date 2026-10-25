"""Config loader: Phase 1 training config + Phase 3 overrides, with absolute paths.

The data root's data.yaml (the YOLO spec written by src/data_preparation and
validated by the attack scripts) is the single source of truth for the class
order and for which folders are the train/val/test splits. A run is pointed at
an attacked split by editing data.yaml (e.g. val: val_pgd_attacked) and
re-running, so no tuning or test script hard-codes a split name.
"""
from pathlib import Path

import yaml

PHASE3 = Path(__file__).resolve().parents[1]
PHASE1 = PHASE3.parent / "modelling"
# Phase 1 moved its training config to configs/train.yaml; the old path is kept first.
PHASE1_CONFIGS = (PHASE1 / "config.yaml", PHASE1 / "configs" / "train.yaml")
# data_updated sits at the repo root (outside src/) when not running under Docker.
DATA_FALLBACK = PHASE3.parents[1] / "data" / "data_updated"
DEFAULT_SPLITS = {"train": "train", "val": "val", "test": "test"}


def names_to_list(names):
    """A YOLO data.yaml `names` (list, or {index: name} with int or str keys) as an ordered list."""
    if isinstance(names, dict):
        try:
            return [names[i] if i in names else names[str(i)] for i in range(len(names))]
        except KeyError as error:
            raise ValueError("data.yaml has missing class indices") from error
    return list(names)


def load_data_yaml(data_root):
    """class_names and split folder names from <data_root>/data.yaml, or (None, defaults) if absent."""
    path = Path(data_root) / "data.yaml"
    if not path.exists():
        return None, dict(DEFAULT_SPLITS)
    meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    class_names = names_to_list(meta["names"]) if "names" in meta else None
    if class_names is not None and meta.get("nc") not in (None, len(class_names)):
        raise ValueError(f"{path}: nc={meta.get('nc')} does not match {len(class_names)} names")
    splits = {role: meta.get(role, default) for role, default in DEFAULT_SPLITS.items()}
    return class_names, splits


def load_config():
    """Phase 1 config (same recipe) + Phase 3 overrides. Paths become absolute.

    data.yaml in the resolved data root sets config["class_names"] and
    config["splits"] (train/val/test folder names). config["test_sets"]["clean"]
    is kept in step with the test split named there.
    """
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

    class_names, splits = load_data_yaml(config["data"])
    if class_names is not None:  # data.yaml is the source of truth; config.yaml is the fallback
        config["class_names"] = class_names
    config["splits"] = splits
    if isinstance(config.get("test_sets"), dict) and "clean" in config["test_sets"]:
        config["test_sets"]["clean"] = splits["test"]  # clean test folder also comes from data.yaml
    return config
