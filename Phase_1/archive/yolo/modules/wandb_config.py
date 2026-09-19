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
    os.environ.pop("WANDB_NAME", None)
    # Keep each search separate so only its own runs get numbered.
    os.environ["WANDB_RUN_GROUP"] = f"{arguments.get('name', 'tune')}_{uuid4().hex}"
    arguments["project"] = project
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


def log_test_metrics(test_metrics):
    """Log final test top-1/top-5 to W&B."""
    import wandb

    if wandb.run is not None:
        wandb.log({
            "test/top1": float(test_metrics.top1),
            "test/top5": float(test_metrics.top5),
        })
        return
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=f"{os.environ.get('WANDB_NAME', 'train')}_test",
        group=os.environ.get("WANDB_NAME"),
        job_type="test",
    )
    run.log({
        "test/top1": float(test_metrics.top1),
        "test/top5": float(test_metrics.top5),
    })
    run.finish()


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
