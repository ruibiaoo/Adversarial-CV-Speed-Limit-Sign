"""Behavioral checks for tuning startup and documented Compose credentials."""

from pathlib import Path
import unittest

import yaml

from Phase_1.train import start_training
from Phase_1.tuning_config import prepare_tuning_arguments


PHASE_DIRECTORY = Path(__file__).resolve().parents[1]


class FakeTune:
    def uniform(self, lower, upper):
        return ("uniform", lower, upper)

    def choice(self, values):
        return ("choice", tuple(values))


class FakeModel:
    def __init__(self):
        self.tune_arguments = None

    def tune(self, **kwargs):
        self.tune_arguments = kwargs
        return "tuned"

    def train(self, **kwargs):
        return "trained"

    def val(self, **kwargs):
        return type("Metrics", (), {"top1": 0.9, "top5": 1.0})()


class StartupWiringTests(unittest.TestCase):
    def test_real_tuner_config_validates_before_constructing_model_and_forwards_project(self):
        """Constructing YOLO before missing-key validation or dropping project is a bug."""
        model = FakeModel()
        factory_calls = []
        integration_calls = []

        def model_factory(weights):
            factory_calls.append(weights)
            return model

        def prepare(config, environ):
            return prepare_tuning_arguments(config, environ, tune_api=FakeTune())

        result = start_training(
            PHASE_DIRECTORY / "config.yaml",
            PHASE_DIRECTORY / "tuner.yaml",
            model_factory=model_factory,
            environ={"WANDB_API_KEY": "test-api-key"},
            prepare_tuning=prepare,
            integration_runner=integration_calls.append,
        )

        self.assertEqual(result, "tuned")
        self.assertEqual(factory_calls, ["yolo26n-cls.pt"])
        self.assertEqual(integration_calls, [True])
        self.assertEqual(model.tune_arguments["project"], "adversarial-cv-speed-limit-sign")
        self.assertNotIn("tuner", model.tune_arguments)
        self.assertNotIn("wandb", model.tune_arguments)

    def test_missing_tuning_key_prevents_model_construction(self):
        """Creating or downloading model weights before credential validation is a bug."""
        factory_calls = []

        with self.assertRaisesRegex(ValueError, "WANDB_API_KEY"):
            start_training(
                PHASE_DIRECTORY / "config.yaml",
                PHASE_DIRECTORY / "tuner.yaml",
                model_factory=factory_calls.append,
                environ={},
                integration_runner=lambda enabled: self.fail("integration was configured"),
            )

        self.assertEqual(factory_calls, [])

    def test_non_tuning_run_needs_no_wandb_key_and_disables_integrations(self):
        """Requiring W&B for ordinary training or leaving callbacks enabled is a bug."""
        model = FakeModel()
        integration_calls = []

        result = start_training(
            PHASE_DIRECTORY / "config.yaml",
            {"tuner": False},
            model_factory=lambda weights: model,
            environ={},
            integration_runner=integration_calls.append,
        )

        self.assertEqual(result, "trained")
        self.assertEqual(integration_calls, [False])


class ComposeCredentialContractTests(unittest.TestCase):
    def test_example_key_and_compose_default_entity_form_a_complete_documented_setup(self):
        """Requiring an undocumented entity or key at Compose rendering is a bug."""
        compose = yaml.safe_load((PHASE_DIRECTORY / "docker-compose.yml").read_text())
        environment = compose["services"]["phase1-classifier"]["environment"]
        example_lines = (PHASE_DIRECTORY / ".env.example").read_text().splitlines()

        self.assertEqual(example_lines, ["WANDB_API_KEY=replace_with_your_wandb_api_key"])
        self.assertEqual(environment["WANDB_API_KEY"], "${WANDB_API_KEY:-}")
        self.assertEqual(environment["WANDB_ENTITY"], "hongyaok-hongyaok")


if __name__ == "__main__":
    unittest.main()
