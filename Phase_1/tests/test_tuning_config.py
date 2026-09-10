"""Behavioral tests for declarative Ray Tune and W&B orchestration."""

import unittest
from pathlib import Path
from subprocess import run
import sys

from Phase_1.train import run_training
from Phase_1.tuning_config import build_tune_space, prepare_tuning_arguments


class FakeTune:
    """Minimal Ray Tune substitute that exposes the requested domain calls."""

    def uniform(self, lower, upper):
        return ("uniform", lower, upper)

    def choice(self, values):
        return ("choice", tuple(values))


class RecordingModel:
    def __init__(self):
        self.tune_arguments = None
        self.train_arguments = None
        self.validation_arguments = None

    def tune(self, **kwargs):
        self.tune_arguments = kwargs
        return "tuned"

    def train(self, **kwargs):
        self.train_arguments = kwargs
        return "trained"

    def val(self, **kwargs):
        self.validation_arguments = kwargs
        return type("Metrics", (), {"top1": 0.9, "top5": 1.0})()


class TuneConfigurationTests(unittest.TestCase):
    def test_train_module_imports_from_the_compose_working_directory(self):
        """A package-only import that breaks ``python train.py`` in Compose is a bug."""
        phase_directory = Path(__file__).resolve().parents[1]

        result = run(
            [sys.executable, "-c", "import train"],
            cwd=phase_directory,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_tune_space_turns_declarative_schema_into_domains(self):
        """Passing YAML mappings through without constructing domains is a bug."""
        schema = {
            "lr0": {"type": "uniform", "min": 1e-5, "max": 1e-1},
            "batch": {"type": "choice", "values": [4, 8, 16, 32]},
        }

        result = build_tune_space(schema, tune_api=FakeTune())

        self.assertEqual(
            result,
            {
                "lr0": ("uniform", 1e-5, 1e-1),
                "batch": ("choice", (4, 8, 16, 32)),
            },
        )

    def test_build_tune_space_rejects_unknown_schema_operations(self):
        """Evaluating arbitrary YAML operations instead of rejecting them is a bug."""
        with self.assertRaisesRegex(ValueError, "Unsupported Tune operation"):
            build_tune_space(
                {"lr0": {"type": "__import__('os').system", "values": []}},
                tune_api=FakeTune(),
            )

    def test_prepare_tuning_arguments_sanitizes_wandb_fields_and_converts_space(self):
        """Forwarding W&B control fields or string search values to YOLO is a bug."""
        config = {
            "tuner": True,
            "use_ray": True,
            "data": "/app/data",
            "epochs": 50,
            "name": "yolo26_speed_limit_tune",
            "seed": 0,
            "space": {"lr0": {"type": "uniform", "min": 1e-5, "max": 1e-1}},
            "grace_period": 10,
            "gpu_per_trial": 1,
            "iterations": 200,
            "search_alg": "optuna",
            "wandb": {
                "project": "adversarial-cv-speed-limit-sign",
                "entity": "hongyaok-hongyaok",
            },
        }
        environ = {"WANDB_API_KEY": "test-api-key"}

        result = prepare_tuning_arguments(config, environ=environ, tune_api=FakeTune())

        self.assertEqual(result["space"], {"lr0": ("uniform", 1e-5, 1e-1)})
        self.assertEqual(result["project"], "adversarial-cv-speed-limit-sign")
        self.assertEqual(result["use_ray"], True)
        self.assertEqual(result["search_alg"], "optuna")
        self.assertNotIn("tuner", result)
        self.assertNotIn("wandb", result)
        self.assertEqual(environ["WANDB_PROJECT"], "adversarial-cv-speed-limit-sign")
        self.assertEqual(environ["WANDB_ENTITY"], "hongyaok-hongyaok")

    def test_disabled_tuning_does_not_prepare_wandb_or_call_tune(self):
        """Preparing W&B when the tuner switch is off is a bug."""
        model = RecordingModel()
        config = {"data": "/app/data"}

        result = run_training(model, config, {"tuner": False})

        self.assertEqual(result, "trained")
        self.assertEqual(model.train_arguments, config)
        self.assertEqual(model.validation_arguments, {"data": "/app/data", "split": "test"})
        self.assertIsNone(model.tune_arguments)

    def test_enabled_tuning_forwards_only_prepared_arguments_to_model_tune(self):
        """Bypassing W&B preparation or forwarding its control fields is a bug."""
        model = RecordingModel()
        tuner_config = {"tuner": True, "wandb": {"project": "ignored"}}
        prepared_arguments = {
            "use_ray": True,
            "search_alg": "optuna",
            "space": {"lr0": ("uniform", 1e-5, 1e-1)},
        }
        received_configs = []

        def prepare_tuning(config):
            received_configs.append(config)
            return prepared_arguments

        result = run_training(
            model,
            {"data": "/app/data"},
            tuner_config,
            prepare_tuning=prepare_tuning,
        )

        self.assertEqual(result, "tuned")
        self.assertEqual(received_configs, [tuner_config])
        self.assertEqual(model.tune_arguments, prepared_arguments)
        self.assertNotIn("tuner", model.tune_arguments)
        self.assertNotIn("wandb", model.tune_arguments)
        self.assertIsNone(model.train_arguments)

if __name__ == "__main__":
    unittest.main()
