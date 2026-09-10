"""Behavioral tests for W&B setup used by Ray Tune."""

import unittest

from Phase_1.wandb_config import prepare_wandb_tuning


class PrepareWandbTuningTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "tuner": True,
            "use_ray": True,
            "data": "/app/data",
            "space": {"lr0": "tune.uniform(1e-5, 1e-1)"},
            "grace_period": 10,
            "gpu_per_trial": 1,
            "iterations": 200,
            "search_alg": "optuna",
            "wandb": {
                "project": "adversarial-cv-speed-limit-sign",
                "entity": "hongyaok-hongyaok",
            },
        }
        self.environ = {"WANDB_API_KEY": "test-api-key"}

    def test_returns_ray_arguments_and_configures_wandb_environment(self):
        """Removing a Ray argument or configuring W&B incorrectly is a bug."""
        result = prepare_wandb_tuning(self.config, self.environ)

        self.assertEqual(
            result,
            {
                "use_ray": True,
                "data": "/app/data",
                "space": {"lr0": "tune.uniform(1e-5, 1e-1)"},
                "grace_period": 10,
                "gpu_per_trial": 1,
                "iterations": 200,
                "search_alg": "optuna",
                "project": "adversarial-cv-speed-limit-sign",
            },
        )
        self.assertEqual(self.environ["WANDB_PROJECT"], "adversarial-cv-speed-limit-sign")
        self.assertEqual(self.environ["WANDB_ENTITY"], "hongyaok-hongyaok")

    def test_does_not_mutate_the_supplied_tuning_mapping(self):
        """Popping fields from caller-owned config is a bug."""
        original = {
            "tuner": True,
            "wandb": {
                "project": "adversarial-cv-speed-limit-sign",
                "entity": "hongyaok-hongyaok",
            },
            "search_alg": "optuna",
        }

        prepare_wandb_tuning(original, self.environ)

        self.assertEqual(
            original,
            {
                "tuner": True,
                "wandb": {
                    "project": "adversarial-cv-speed-limit-sign",
                    "entity": "hongyaok-hongyaok",
                },
                "search_alg": "optuna",
            },
        )

    def test_requires_a_non_empty_wandb_api_key_without_leaking_it(self):
        """Accepting an absent or empty credential is a bug."""
        for api_key in (None, ""):
            with self.subTest(api_key=api_key):
                environ = {} if api_key is None else {"WANDB_API_KEY": api_key}

                with self.assertRaisesRegex(ValueError, "WANDB_API_KEY") as error:
                    prepare_wandb_tuning(self.config, environ)

                self.assertNotIn("test-api-key", str(error.exception))

    def test_requires_a_project(self):
        """Running W&B without a target project is a bug."""
        config = {**self.config, "wandb": {"entity": "hongyaok-hongyaok"}}

        with self.assertRaisesRegex(ValueError, "WANDB_PROJECT"):
            prepare_wandb_tuning(config, self.environ)

    def test_uses_environment_entity_when_config_entity_is_absent(self):
        """Ignoring WANDB_ENTITY fallback is a bug."""
        config = {
            **self.config,
            "wandb": {"project": "adversarial-cv-speed-limit-sign"},
        }
        environ = {"WANDB_API_KEY": "test-api-key", "WANDB_ENTITY": "hongyaok-hongyaok"}

        result = prepare_wandb_tuning(config, environ)

        self.assertEqual(result["project"], "adversarial-cv-speed-limit-sign")
        self.assertEqual(environ["WANDB_ENTITY"], "hongyaok-hongyaok")

    def test_requires_an_entity_when_not_in_config_or_environment(self):
        """Starting W&B with no entity source is a bug."""
        config = {
            **self.config,
            "wandb": {"project": "adversarial-cv-speed-limit-sign"},
        }

        with self.assertRaisesRegex(ValueError, "WANDB_ENTITY"):
            prepare_wandb_tuning(config, self.environ)


if __name__ == "__main__":
    unittest.main()
