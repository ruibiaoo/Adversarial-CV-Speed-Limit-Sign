"""W&B setup for normal training and sweeps."""
import os
from uuid import uuid4


def prepare_wandb_tuning(tuner_config):
    """Check W&B settings for sweeps and set env vars."""
    arguments = tuner_config.copy()
    arguments.pop("tuner", None)
    # space stays in the sweep dict, method/metric/iterations handled by train.py
    wandb_config = arguments.pop("wandb", {})

    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required")

    project = wandb_config.get("project")
    if not project:
        raise ValueError("wandb.project is required in tuner.yaml")

    entity = wandb_config.get("entity") or os.environ.get("WANDB_ENTITY")
    if not entity:
        raise ValueError("Set wandb.entity in tuner.yaml or WANDB_ENTITY")

    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_ENTITY"] = entity
    os.environ.pop("WANDB_NAME", None)
    # Keep each search separate so only its own runs get grouped.
    os.environ["WANDB_RUN_GROUP"] = f"{arguments.get('name', 'tune')}_{uuid4().hex}"
    # dont put wandb project into arguments["project"]
    # because train.py uses project as the local folder (/app/runs)
    return arguments


def prepare_wandb_training(config):
    """Turn on W&B for normal training, or warn and skip it."""
    wandb_config = config.pop("wandb", {}) or {}

    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set; training without W&B tracking.")
        return False

    project = wandb_config.get("project")
    entity = wandb_config.get("entity") or os.environ.get("WANDB_ENTITY")
    if not project or not entity:
        print("wandb project/entity missing; training without W&B tracking.")
        return False

    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_ENTITY"] = entity
    if config.get("name"):
        os.environ["WANDB_NAME"] = config["name"]
        os.environ["WANDB_RUN_GROUP"] = config["name"]
    else:
        os.environ.pop("WANDB_RUN_GROUP", None)
    return True
