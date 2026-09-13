from pathlib import Path
from ultralytics import settings
from ultralytics import YOLO
import yaml

if __package__:
    from .modules.tuning_config import prepare_tuning_arguments
    from .modules.tuning_results import save_tuning_results
    from .modules.wandb_config import rename_wandb_runs
else:
    from modules.tuning_config import prepare_tuning_arguments
    from modules.tuning_results import save_tuning_results
    from modules.wandb_config import rename_wandb_runs


def load_config(path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    phase_directory = Path(__file__).resolve().parent
    config = load_config(phase_directory / "config.yaml")
    tuner_config = load_config(phase_directory / "tuner.yaml")
    tuning = tuner_config.get("tuner", False)

    if tuning:
        tuning_arguments = prepare_tuning_arguments(tuner_config)

    # Set integrations before YOLO loads its training callbacks.

    settings.update({"wandb": tuning, "raytune": tuning})
    if tuning:
        settings.update({"runs_dir": str(phase_directory / "tuning")})

    model = YOLO("yolo26m-cls.pt")

    if tuning:
        results = model.tune(**tuning_arguments)
        save_tuning_results(results, phase_directory / "tuning")
        rename_wandb_runs(tuner_config.get("name", "tune"))
        return results

    results = model.train(**config)
    print("\n--- EVALUATING ON TEST SPLIT ---")
    test_metrics = model.val(data=config["data"], split="test")
    print(f"Top-1 Accuracy: {test_metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {test_metrics.top5:.4f}")
    return results


if __name__ == "__main__":
    main()
