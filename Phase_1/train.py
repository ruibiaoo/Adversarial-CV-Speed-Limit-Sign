import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import yaml

if __package__:
    from .tuning_config import prepare_tuning_arguments
else:
    from tuning_config import prepare_tuning_arguments


def run_training(
    model: Any,
    config: Mapping[str, Any],
    tuner_config: Mapping[str, Any],
    prepare_tuning: Callable[[Mapping[str, Any]], dict[str, Any]] = prepare_tuning_arguments,
) -> Any:
    """Run either the configured Ray Tune search or normal train-and-evaluate flow."""
    if tuner_config.get("tuner", False):
        tuning_arguments = prepare_tuning(tuner_config)
        space_types = {
            name: type(domain).__name__
            for name, domain in tuning_arguments["space"].items()
        }
        print("Ray Tune search-space domain types:", space_types)
        return model.tune(**tuning_arguments)

    results = model.train(**config)
    print("\n--- EVALUATING ON TEST SPLIT ---")
    test_metrics = model.val(data=config.get("data", "/app/data"), split="test")
    print("\n--- TEST METRICS ---")
    print(f"Top-1 Accuracy: {test_metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {test_metrics.top5:.4f}")
    return results


def configure_integrations(enabled: bool) -> None:
    """Set Ultralytics callbacks for the selected execution mode."""
    value = str(enabled).lower()
    subprocess.run(["yolo", "settings", f"wandb={value}", f"raytune={value}"], check=True)


def load_mapping(source: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """Load YAML once, or accept a mapping supplied by a caller/test."""
    if isinstance(source, Mapping):
        return source
    with source.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def start_training(
    config_source: Path | Mapping[str, Any],
    tuner_source: Path | Mapping[str, Any],
    *,
    model_factory: Callable[[str], Any],
    environ: Mapping[str, str] | dict[str, str] = os.environ,
    prepare_tuning: Callable[..., dict[str, Any]] = prepare_tuning_arguments,
    integration_runner: Callable[[bool], None] = configure_integrations,
) -> Any:
    """Validate configuration before constructing YOLO, then run the selected mode."""
    config = load_mapping(config_source)
    tuner_config = load_mapping(tuner_source)

    if tuner_config.get("tuner", False):
        tuning_arguments = prepare_tuning(tuner_config, environ=environ)
        integration_runner(True)
        model = model_factory("yolo26n-cls.pt")
        space_types = {name: type(domain).__name__ for name, domain in tuning_arguments["space"].items()}
        print("Ray Tune search-space domain types:", space_types)
        return model.tune(**tuning_arguments)

    integration_runner(False)
    return run_training(model_factory("yolo26n-cls.pt"), config, tuner_config)


def create_yolo(weights: str) -> Any:
    """Import Ultralytics only after startup validation selects a run mode."""
    from ultralytics import YOLO

    return YOLO(weights)


if __name__ == "__main__":
    phase_directory = Path(__file__).resolve().parent
    start_training(
        phase_directory / "config.yaml",
        phase_directory / "tuner.yaml",
        model_factory=create_yolo,
    )
