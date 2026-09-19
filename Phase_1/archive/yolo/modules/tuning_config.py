"""Convert the tuning YAML settings into arguments for model.tune()."""
from .wandb_config import prepare_wandb_tuning
from ray import tune


def build_tune_space(schema):
    """Turn each uniform range or choice list into a Ray search parameter."""

    space = {}

    for name, specification in schema.items():
        operation = specification.get("type")
        if operation == "uniform":
            space[name] = tune.uniform(specification["min"], specification["max"])
        elif operation == "choice":
            space[name] = tune.choice(specification["values"])
        else:
            raise ValueError(f"Unsupported Tune operation for {name}: {operation}")

    return space


def prepare_tuning_arguments(tuner_config):
    """Set up W&B and convert the search space before tuning starts."""
    arguments = prepare_wandb_tuning(tuner_config)
    arguments["space"] = build_tune_space(arguments["space"])
    return arguments
