import os
from uuid import uuid4


def prepare_wandb_tuning(tuner_config):
    """Return Ray/Ultralytics tuning arguments and configure W&B environment."""
    arguments = tuner_config.copy()
    arguments.pop("tuner", None)
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
    # Keep each search separate so only its own runs get numbered.
    os.environ["WANDB_RUN_GROUP"] = f"{arguments.get('name', 'tune')}_{uuid4().hex}"
    arguments["project"] = project
    return arguments


def rename_wandb_runs(name):
    """Number this search's W&B runs after tuning has finished."""
    import wandb

    if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        print("W&B run numbering needs online mode; skipping renaming.")
        return

    project = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
    runs = wandb.Api().runs(
        project,
        filters={"group": os.environ["WANDB_RUN_GROUP"]},
        order="+created_at",
    )
    for number, run in enumerate(runs, start=1):
        run.name = f"{name}_run_id_{number}"
        run.update()

    print(f"W&B runs numbered in group: {os.environ['WANDB_RUN_GROUP']}")
