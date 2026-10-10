import unittest

import numpy as np

try:
    from .pixle_attack import PixleAttack, PixleConfig, attack_with_original_l0
except ImportError:  # Allows direct execution from the Phase_2 directory.
    from pixle_attack import PixleAttack, PixleConfig, attack_with_original_l0


class PixleAttackTests(unittest.TestCase):
    def test_query_budget_and_l0_accounting(self):
        calls = []

        def oracle(image):
            calls.append(image.copy())
            return np.array([0.9, 0.1], dtype=float)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        attacker = PixleAttack(
            oracle,
            PixleConfig(restarts=2, iterations=3, patch_size=1, seed=1),
        )
        result = attack_with_original_l0(
            attacker,
            image,
            label=0,
        )

        self.assertEqual(result.queries, 1 + 2 * 3)
        self.assertEqual(len(calls), result.queries)
        self.assertEqual(result.l0, 0)
        self.assertFalse(result.success)

    def test_initial_query_can_be_reused(self):
        call_count = 0

        def oracle(image):
            nonlocal call_count
            call_count += 1
            return np.array([0.6, 0.4], dtype=float)

        image = np.zeros((3, 3, 3), dtype=np.uint8)
        initial = oracle(image)
        attacker = PixleAttack(
            oracle,
            PixleConfig(restarts=1, iterations=1, patch_size=1, seed=2),
        )
        result = attacker.attack(image, label=0, initial_probabilities=initial)

        self.assertEqual(result.queries, 2)
        self.assertEqual(call_count, 2)

    def test_progress_callback_receives_query_counts(self):
        progress = []

        def oracle(image):
            return np.array([0.9, 0.1], dtype=float)

        image = np.zeros((3, 3, 3), dtype=np.uint8)
        attacker = PixleAttack(
            oracle,
            PixleConfig(restarts=1, iterations=2, patch_size=1, seed=3),
        )
        result = attacker.attack(
            image,
            label=0,
            progress_callback=lambda queries, limit: progress.append(
                (queries, limit)
            ),
        )

        self.assertEqual(result.queries, 3)
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])

    def test_last_perturbed_image_is_retained(self):
        calls = []

        def oracle(image):
            calls.append(image.copy())
            return np.array([0.9, 0.1], dtype=float)

        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        attacker = PixleAttack(
            oracle,
            PixleConfig(restarts=1, iterations=1, patch_size=1, seed=4),
        )
        result = attack_with_original_l0(attacker, image, label=0)

        self.assertFalse(result.success)
        self.assertTrue(np.array_equal(result.last_perturbed_image, calls[-1]))
        self.assertFalse(np.array_equal(result.last_perturbed_image, image))


if __name__ == "__main__":
    unittest.main()
