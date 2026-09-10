"""Build safe Ray Tune arguments from the declarative tuner configuration."""

import os
from typing import Any, Mapping, MutableMapping

if __package__:
    from .wandb_config import prepare_wandb_tuning
else:
    from wandb_config import prepare_wandb_tuning


def _resolve_tune_api(tune_api: Any | None) -> Any:
    if tune_api is not None:
        return tune_api

    from ray import tune

    return tune


def build_tune_space(
    schema: Mapping[str, Mapping[str, Any]], tune_api: Any | None = None
) -> dict[str, Any]:
    """Convert the supported YAML schema into Ray Tune Domain objects."""
    tune = _resolve_tune_api(tune_api)
    space: dict[str, Any] = {}

    for name, specification in schema.items():
        operation = specification.get("type")
        if operation == "uniform":
            space[name] = tune.uniform(specification["min"], specification["max"])
        elif operation == "choice":
            space[name] = tune.choice(specification["values"])
        else:
            raise ValueError(f"Unsupported Tune operation for {name}: {operation}")

    return space


def prepare_tuning_arguments(
    tuner_config: Mapping[str, Any],
    environ: MutableMapping[str, str] = os.environ,
    tune_api: Any | None = None,
) -> dict[str, Any]:
    """Prepare W&B metadata and replace declarative space values with Ray domains."""
    arguments = prepare_wandb_tuning(tuner_config, environ=environ)
    arguments["space"] = build_tune_space(arguments["space"], tune_api=tune_api)
    return arguments
