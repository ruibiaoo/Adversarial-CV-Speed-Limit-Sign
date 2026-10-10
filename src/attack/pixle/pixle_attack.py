"""Black-box Pixle attacks.

The implementation follows the paper's Restart-Iterative procedure:

* sample a contiguous source patch;
* map each source pixel to a destination coordinate;
* query the black-box model on the candidate image;
* keep the candidate that improves the attack loss the most.

The attack only requires a callable that returns class probabilities. It does
not require gradients, model weights, or a training procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np


MappingName = Literal[
    "random",
    "similarity",
    "distance",
    "similarity_distribution",
    "distance_distribution",
]
AlgorithmName = Literal["restart-iterative", "iterative"]
OperationName = Literal["overwrite", "swap"]
ProgressCallback = Callable[[int, int], None]


class ProbabilityModel(Protocol):
    """Protocol for a model exposed as a black-box probability oracle."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return one probability per class for one HWC image."""


@dataclass(frozen=True)
class PixleConfig:
    """Parameters controlling a Pixle attack."""

    restarts: int = 100
    iterations: int = 50
    patch_size: int | tuple[int, int] = 3
    mapping: MappingName = "random"
    algorithm: AlgorithmName = "restart-iterative"
    operation: OperationName = "overwrite"
    seed: int | None = 4266
    max_queries: int | None = None

    def __post_init__(self) -> None:
        if self.restarts < 1:
            raise ValueError("restarts must be at least 1")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if isinstance(self.patch_size, int):
            patch_height = patch_width = self.patch_size
        else:
            patch_height, patch_width = self.patch_size
        if patch_height < 1 or patch_width < 1:
            raise ValueError("patch dimensions must be positive")
        if self.mapping not in {
            "random",
            "similarity",
            "distance",
            "similarity_distribution",
            "distance_distribution",
        }:
            raise ValueError(f"Unknown mapping: {self.mapping}")
        if self.algorithm not in {"restart-iterative", "iterative"}:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        if self.operation not in {"overwrite", "swap"}:
            raise ValueError(f"Unknown operation: {self.operation}")
        if self.max_queries is not None and self.max_queries < 1:
            raise ValueError("max_queries must be positive when supplied")


@dataclass
class AttackResult:
    """Result and accounting information for one attacked image."""

    adversarial_image: np.ndarray
    original_label: int
    adversarial_label: int
    original_confidence: float
    adversarial_confidence: float
    last_perturbed_image: np.ndarray
    last_perturbed_label: int
    last_perturbed_confidence: float
    success: bool
    queries: int
    l0: int
    loss: float
    target_label: int | None


class PixleAttack:
    """Run Pixle against a probability-query black-box model."""

    def __init__(
        self,
        predict_proba: ProbabilityModel | Callable[[np.ndarray], np.ndarray],
        config: PixleConfig | None = None,
    ) -> None:
        self.predict_proba = predict_proba
        self.config = config or PixleConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def attack(
        self,
        image: np.ndarray,
        label: int,
        *,
        target_label: int | None = None,
        initial_probabilities: np.ndarray | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AttackResult:
        """Attack one image.

        Parameters
        ----------
        image:
            HWC image array. The dtype/range is preserved and passed unchanged
            to the black-box oracle.
        label:
            Correct original class index.
        target_label:
            Optional target class. If supplied, the attack is targeted and
            succeeds only when the prediction equals this class.
        initial_probabilities:
            Optional result of an already-counted initial model query. This is
            useful for evaluation pipelines that filter initially incorrect
            images without querying the model twice.
        """
        original = self._validate_image(image)
        target = self._validate_target(target_label, label)
        query_limit = self.config.max_queries or (
            self.config.restarts * self.config.iterations + 1
        )

        if initial_probabilities is None:
            initial = self._query(original)
            queries = 1
        else:
            initial = self._validate_probabilities(initial_probabilities)
            queries = 1  # the caller supplied one already-counted query

        if progress_callback is not None:
            progress_callback(queries, query_limit)

        original_label = int(np.argmax(initial))
        best_image = original.copy()
        best_probabilities = initial.copy()
        best_loss = self._loss(initial, label, target)
        last_perturbed_image = original.copy()
        last_perturbed_probabilities = initial.copy()

        if self._success(original_label, label, target):
            return self._result(
                best_image,
                original_label,
                best_probabilities,
                initial,
                True,
                queries,
                best_loss,
                target,
                last_perturbed_image,
                last_perturbed_probabilities,
            )

        patch_height, patch_width = self._patch_shape()
        if patch_height > original.shape[0] or patch_width > original.shape[1]:
            raise ValueError(
                f"patch size {(patch_height, patch_width)} exceeds image size "
                f"{original.shape[:2]}"
            )
        for _restart in range(self.config.restarts):
            # This is the current best image at the beginning of a restart.
            # Candidate images inside a restart are sampled from this same
            # base, matching the paper's random-search pseudocode.
            restart_base = best_image.copy()
            restart_best = best_image.copy()

            for _iteration in range(self.config.iterations):
                if queries >= query_limit:
                    return self._result(
                        best_image,
                        original_label,
                        best_probabilities,
                        initial,
                        self._success(
                            int(np.argmax(best_probabilities)), label, target
                        ),
                        queries,
                        best_loss,
                        target,
                        last_perturbed_image,
                        last_perturbed_probabilities,
                    )

                patch = self._sample_patch(
                    original.shape[:2], patch_height, patch_width
                )
                candidate_base = (
                    best_image if self.config.algorithm == "iterative" else restart_base
                )
                mapping = self._build_mapping(candidate_base, patch)
                candidate = self._apply_mapping(candidate_base, mapping)
                probabilities = self._query(candidate)
                queries += 1
                last_perturbed_image = candidate.copy()
                last_perturbed_probabilities = probabilities.copy()
                if progress_callback is not None:
                    progress_callback(queries, query_limit)
                candidate_label = int(np.argmax(probabilities))

                if self._success(candidate_label, label, target):
                    candidate_loss = self._loss(probabilities, label, target)
                    return self._result(
                        candidate,
                        original_label,
                        probabilities,
                        initial,
                        True,
                        queries,
                        candidate_loss,
                        target,
                        last_perturbed_image,
                        last_perturbed_probabilities,
                    )

                candidate_loss = self._loss(probabilities, label, target)
                if candidate_loss < best_loss:
                    best_image = candidate.copy()
                    best_probabilities = probabilities.copy()
                    best_loss = candidate_loss
                    restart_best = candidate.copy()

            best_image = restart_best

        final_label = int(np.argmax(best_probabilities))
        return self._result(
            best_image,
            original_label,
            best_probabilities,
            initial,
            self._success(final_label, label, target),
            queries,
            best_loss,
            target,
            last_perturbed_image,
            last_perturbed_probabilities,
        )

    def _query(self, image: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(image)
        return self._validate_probabilities(probabilities)

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
            raise ValueError("image must have shape (height, width, channels)")
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError("image dimensions must be positive")
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        if array.shape[2] == 4:
            array = array[:, :, :3]
        return array.copy()

    @staticmethod
    def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64).squeeze()
        if values.ndim != 1 or values.size < 2:
            raise ValueError("predict_proba must return a 1D class-probability vector")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("probabilities must be finite and non-negative")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("probabilities must have a positive sum")
        values = values / total
        return values

    @staticmethod
    def _validate_target(target_label: int | None, label: int) -> int | None:
        if target_label is None:
            return None
        if target_label < 0:
            raise ValueError("target_label must be non-negative")
        if target_label == label:
            raise ValueError("target_label must differ from the original label")
        return int(target_label)

    def _patch_shape(self) -> tuple[int, int]:
        if isinstance(self.config.patch_size, int):
            return self.config.patch_size, self.config.patch_size
        return self.config.patch_size

    def _sample_patch(
        self,
        image_shape: tuple[int, int],
        patch_height: int,
        patch_width: int,
    ) -> tuple[int, int, int, int]:
        height, width = image_shape
        top = int(self.rng.integers(0, height - patch_height + 1))
        left = int(self.rng.integers(0, width - patch_width + 1))
        return top, left, patch_height, patch_width

    def _build_mapping(
        self,
        image: np.ndarray,
        patch: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        top, left, patch_height, patch_width = patch
        height, width = image.shape[:2]
        flat_pixels = image[:, :, :3].reshape(-1, 3).astype(np.float64)
        mapping = []

        for source_y in range(top, top + patch_height):
            for source_x in range(left, left + patch_width):
                source_index = source_y * width + source_x
                destination_index = self._destination_index(
                    flat_pixels, source_index, height, width
                )
                destination_y, destination_x = divmod(destination_index, width)
                mapping.append(
                    (source_y, source_x, destination_y, destination_x)
                )
        return mapping

    def _destination_index(
        self,
        pixels: np.ndarray,
        source_index: int,
        height: int,
        width: int,
    ) -> int:
        mapping = self.config.mapping
        total_pixels = height * width

        if mapping == "random":
            destination = int(self.rng.integers(0, total_pixels - 1))
            if destination >= source_index:
                destination += 1
            return destination

        source_pixel = pixels[source_index]
        distances = np.sum((pixels - source_pixel) ** 2, axis=1)
        distances[source_index] = np.inf if mapping in {
            "similarity",
            "similarity_distribution",
        } else -np.inf

        if mapping == "similarity":
            return int(np.argmin(distances))
        if mapping == "distance":
            return int(np.argmax(distances))

        if mapping == "similarity_distribution":
            finite = np.isfinite(distances)
            weights = np.zeros_like(distances)
            weights[finite] = 1.0 / (distances[finite] + 1e-12)
        elif mapping == "distance_distribution":
            finite = np.isfinite(distances)
            weights = np.zeros_like(distances)
            weights[finite] = np.maximum(distances[finite], 0.0)
        else:  # guarded by PixleConfig, retained for defensive programming
            raise ValueError(f"Unknown mapping: {mapping}")

        weights[source_index] = 0.0
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            candidates = np.flatnonzero(weights == 0)
            candidates = candidates[candidates != source_index]
            return int(self.rng.choice(candidates))
        probabilities = weights / weight_sum
        return int(self.rng.choice(total_pixels, p=probabilities))

    def _apply_mapping(
        self,
        base: np.ndarray,
        mapping: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        candidate = base.copy()
        if self.config.operation == "overwrite":
            for source_y, source_x, destination_y, destination_x in mapping:
                candidate[destination_y, destination_x] = base[source_y, source_x]
            return candidate

        # Read every value from the same base image so a swap does not depend
        # on the order in which mappings are applied.
        for source_y, source_x, destination_y, destination_x in mapping:
            source_value = base[source_y, source_x].copy()
            destination_value = base[destination_y, destination_x].copy()
            candidate[source_y, source_x] = destination_value
            candidate[destination_y, destination_x] = source_value
        return candidate

    @staticmethod
    def _loss(
        probabilities: np.ndarray,
        label: int,
        target_label: int | None,
    ) -> float:
        if target_label is None:
            return float(probabilities[label])
        return float(1.0 - probabilities[target_label])

    @staticmethod
    def _success(
        predicted_label: int,
        label: int,
        target_label: int | None,
    ) -> bool:
        if target_label is None:
            return predicted_label != label
        return predicted_label == target_label

    @staticmethod
    def _result(
        image: np.ndarray,
        original_label: int,
        probabilities: np.ndarray,
        initial_probabilities: np.ndarray,
        success: bool,
        queries: int,
        loss: float,
        target_label: int | None,
        last_perturbed_image: np.ndarray,
        last_perturbed_probabilities: np.ndarray,
    ) -> AttackResult:
        adversarial_label = int(np.argmax(probabilities))
        original_confidence = float(initial_probabilities[original_label])
        adversarial_confidence = float(probabilities[adversarial_label])
        last_perturbed_label = int(np.argmax(last_perturbed_probabilities))
        last_perturbed_confidence = float(
            last_perturbed_probabilities[last_perturbed_label]
        )
        # The original image is not available here, so l0 is computed by the
        # caller-facing wrapper in attack_with_original_l0 below.
        return AttackResult(
            adversarial_image=image,
            original_label=original_label,
            adversarial_label=adversarial_label,
            original_confidence=original_confidence,
            adversarial_confidence=adversarial_confidence,
            last_perturbed_image=last_perturbed_image,
            last_perturbed_label=last_perturbed_label,
            last_perturbed_confidence=last_perturbed_confidence,
            success=success,
            queries=queries,
            l0=0,
            loss=loss,
            target_label=target_label,
        )


def count_l0(original: np.ndarray, adversarial: np.ndarray) -> int:
    """Count image pixels whose channel values changed."""
    original_array = np.asarray(original)
    adversarial_array = np.asarray(adversarial)
    if original_array.shape != adversarial_array.shape:
        raise ValueError("original and adversarial images must have the same shape")
    if original_array.ndim != 3:
        raise ValueError("images must have shape (height, width, channels)")
    return int(np.count_nonzero(np.any(original_array != adversarial_array, axis=2)))


def attack_with_original_l0(
    attacker: PixleAttack,
    image: np.ndarray,
    label: int,
    *,
    target_label: int | None = None,
    initial_probabilities: np.ndarray | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AttackResult:
    """Run an attack and fill in the final (L_0) distance."""
    original = np.asarray(image).copy()
    result = attacker.attack(
        original,
        label,
        target_label=target_label,
        initial_probabilities=initial_probabilities,
        progress_callback=progress_callback,
    )
    result.l0 = count_l0(original, result.adversarial_image)
    return result


def parse_patch_size(value: str) -> int | tuple[int, int]:
    """Parse ``3`` or ``3x5`` from a CLI argument."""
    text = value.lower().replace(" ", "")
    if "x" not in text:
        return int(text)
    height, width = text.split("x", maxsplit=1)
    return int(height), int(width)
