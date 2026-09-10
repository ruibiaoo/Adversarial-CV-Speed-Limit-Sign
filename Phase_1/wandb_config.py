"""Prepare W&B settings for Ray Tune without using W&B Sweeps."""

from typing import Any, Mapping, MutableMapping


def prepare_wandb_tuning(
    tuner_config: Mapping[str, Any], environ: MutableMapping[str, str]
) -> dict[str, Any]:
    """Return Ray/Ultralytics tuning arguments and configure W&B environment."""
    arguments = dict(tuner_config)
    arguments.pop("tuner", None)
    wandb_config = arguments.pop("wandb", {})

    if not environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required")

    project = wandb_config.get("project")
    if not project:
        raise ValueError("WANDB_PROJECT is required")

    entity = wandb_config.get("entity") or environ.get("WANDB_ENTITY")
    if not entity:
        raise ValueError("WANDB_ENTITY is required")

    environ["WANDB_PROJECT"] = project
    environ["WANDB_ENTITY"] = entity
    arguments["project"] = project
    return arguments
