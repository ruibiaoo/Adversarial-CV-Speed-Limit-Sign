"""Convert the tuning YAML settings into a W&B sweep dict."""
from .wandb_config import prepare_wandb_tuning


def build_sweep_parameters(schema):
    """Turn each uniform range or choice list into a W&B sweep parameter."""
    params = {}

    for name, spec in (schema or {}).items():
        op = spec.get("type")
        if op == "uniform":
            params[name] = {
                "distribution": "uniform",
                "min": spec["min"],
                "max": spec["max"],
            }
        elif op == "choice":
            params[name] = {"values": spec["values"]}
        else:
            raise ValueError(f"Unsupported sweep operation for {name}: {op}")

    return params


def prepare_sweep(tuner_config):
    """Check W&B, split sweep settings from fixed trial config."""
    args = prepare_wandb_tuning(tuner_config)

    space = args.pop("space", {})
    method = args.pop("method", "bayes")
    metric = args.pop("metric", {"name": "val_accuracy", "goal": "maximize"})
    iterations = int(args.pop("iterations", 20))
    # rest of args (data, epochs, augments, etc) are fixed for every trial
    fixed = args

    sweep = {
        "method": method,
        "metric": metric,
        "parameters": build_sweep_parameters(space),
    }
    if fixed.get("name"):
        sweep["name"] = fixed["name"]

    return sweep, fixed, iterations
