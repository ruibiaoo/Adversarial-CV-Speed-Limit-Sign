from itertools import product
from pathlib import Path
import random
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
import tensorflow as tf
import yaml
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier

def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config


def resolve_repo_path(path_value: str, repo_directory: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_directory / path


def validate_class_mapping(config: dict, repo_directory: Path) -> None:
    names = config["data"]["class_names"]
    expected = [f"{speed}_MPH" for speed in range(10, 71, 10)]
    if names != expected:
        raise ValueError(f"Invalid class order: expected {expected}, got {names}")
    root = resolve_repo_path(config["data"]["root"], repo_directory)
    metadata = load_config(root / "data.yaml")
    mapped = metadata.get("names")
    if isinstance(mapped, dict):
        try:
            mapped = [mapped[index] if index in mapped else mapped[str(index)] for index in range(len(names))]
        except KeyError as error:
            raise ValueError("data.yaml has missing class indices") from error
    if metadata.get("nc") != len(names) or mapped != names:
        raise ValueError(f"Class mapping in {root / 'data.yaml'} does not match pgd.yaml")
    for name in names:
        if not (root / "val" / name).is_dir():
            raise FileNotFoundError(f"Validation class directory not found: {root / 'val' / name}")


def validate_attack_settings(config: dict) -> None:
    attack = config["attack"]
    required = {
        "name": "pgd", "objective": "untargeted", "norm": "linf",
        "loss": "cross_entropy", "label_source": "ground_truth",
        "selection": "highest_loss_per_image",
    }
    for key, expected in required.items():
        if str(attack.get(key, "")).lower() != expected:
            raise ValueError(f"attack.{key} must be {expected}")
    if attack.get("random_start") is not True:
        raise ValueError("PGD sweep requires attack.random_start: true")
    if float(attack["clip_min"]) != 0.0 or float(attack["clip_max"]) != 1.0:
        raise ValueError("PGD expects attack.clip_min=0 and attack.clip_max=1")
    if str(config["experiment"]["split"]).lower() != "val":
        raise ValueError("This sweep must use experiment.split: val")
    if int(config["data"]["batch_size"]) < 1:
        raise ValueError("data.batch_size must be positive")


def load_validation_batches(config: dict, repo_directory: Path):
    data_config = config["data"]
    image_size = int(config["model"]["image_size"])
    validation_directory = resolve_repo_path(
        data_config["root"], repo_directory
    ) / "val"

    if not validation_directory.is_dir():
        raise FileNotFoundError(
            f"Validation directory does not exist: {validation_directory}"
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        validation_directory,
        labels="inferred",
        label_mode="int",
        class_names=data_config["class_names"],
        image_size=(image_size, image_size),
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
    )

    batches = []
    for images, labels in dataset:
        batches.append(
            (
                np.asarray(images, dtype=np.float32) / 255.0,
                np.asarray(labels, dtype=np.int64),
            )
        )
    if not batches:
        raise ValueError(f"No validation images found in {validation_directory}")
    return batches


def load_phase_1_model(model_path: Path, repo_directory: Path, num_classes: int):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (ModuleNotFoundError, TypeError, ValueError) as load_error:
        if str(repo_directory) not in sys.path:
            sys.path.insert(0, str(repo_directory))
        from src.Phase_1.modules.model import build_model

        train_config_path = repo_directory / "src" / "Phase_1" / "configs" / "train.yaml"
        train_config = load_config(train_config_path)
        train_config["pretrained"] = False
        model = build_model(train_config, num_classes)
        try:
            model.load_weights(model_path)
        except Exception as weight_error:
            raise RuntimeError(
                f"Could not load model {model_path}: {load_error}"
            ) from weight_error
        return model


def load_art_classifier(config:dict, repo_directory: Path):
    # Load configurations and path
    model_config = config["model"]
    attack_config = config["attack"]
    model_path = resolve_repo_path(model_config["path"], repo_directory)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    # Load image size and number of classes
    image_size = int(model_config["image_size"])
    num_classes = len(config["data"]["class_names"])

    # Loading the model and setting it to non-trainable
    model = load_phase_1_model(model_path, repo_directory, num_classes)
    model.trainable = False
    if tuple(model.input_shape[1:]) != (image_size, image_size, 3):
        raise ValueError(f"Checkpoint input shape {model.input_shape} does not match pgd.yaml")
    if model.output_shape[-1] != num_classes:
        raise ValueError(f"Checkpoint outputs {model.output_shape[-1]} classes, expected {num_classes}")

    # PGD modifies raw RGB images in the [0,1] range
    raw_images = tf.keras.Input(shape=(image_size, image_size, 3), name="raw_images")
    processed_images = (tf.keras.applications.resnet.preprocess_input(raw_images * 255.0))

    predictions = model(processed_images, training=False)

    attack_model = tf.keras.Model(inputs=raw_images, outputs=predictions)

    # Loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=bool(model_config["from_logits"])
    )

    # Load the ART classifier
    art_classifier = TensorFlowV2Classifier(
        model=attack_model,
        nb_classes=num_classes,
        input_shape=(image_size, image_size, 3),
        loss_object = loss,
        channels_first=False,
        clip_values=(float(attack_config["clip_min"]), float(attack_config["clip_max"]))
    )

    return art_classifier

def convert_norm(norm_name: str):
    norm_mapping = {
        "linf": np.inf,
        "l2": 2,
        "l1": 1,
    }
    normalized_name = norm_name.lower()
    if normalized_name not in norm_mapping:
        raise ValueError(f"Unsupported norm: {norm_name}")
    return norm_mapping[normalized_name]


def pgd_attack(art_classifier: TensorFlowV2Classifier, config: dict, parameters: dict):
    attack_configs = config["attack"]

    epsilon = float(parameters["epsilon_255"]) / 255.0
    alpha = epsilon * float(parameters["alpha_ratio"])
    targeted = attack_configs["objective"].lower() == "targeted"
    if targeted:
        raise NotImplementedError(
            "The current evaluation pipeline supports untargeted PGD only."
        )

    number_of_random_starts = 1 if attack_configs["random_start"] else 0

    return ProjectedGradientDescent(
        estimator=art_classifier,
        eps=epsilon,
        eps_step=alpha,
        max_iter=int(parameters["iterations"]),
        targeted=targeted,
        num_random_init=number_of_random_starts,
        norm=convert_norm(attack_configs["norm"]),
        batch_size=int(config["data"]["batch_size"]),
        random_eps=False,
        verbose=False,
    )


def per_image_loss(labels, predictions, from_logits):
    losses = tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        predictions,
        from_logits=from_logits,
    )
    return np.asarray(losses, dtype=np.float32)


def generate_best_restart(attack, classifier, images, labels, parameters, config):
    random_start = bool(config["attack"]["random_start"])
    attempts = int(parameters["restarts"]) if random_start else 1
    from_logits = bool(config["model"]["from_logits"])
    best_images = None
    best_predictions = None
    best_losses = np.full(len(labels), -np.inf, dtype=np.float32)

    for _ in range(attempts):
        candidate_images = attack.generate(x=images, y=labels)
        candidate_predictions = classifier.predict(candidate_images)
        candidate_losses = per_image_loss(
            labels, candidate_predictions, from_logits
        )
        if not np.isfinite(candidate_losses).all():
            raise RuntimeError("PGD produced a non-finite cross-entropy loss")
        better = candidate_losses > best_losses
        best_losses[better] = candidate_losses[better]
        if best_images is None:
            best_images = candidate_images.copy()
            best_predictions = candidate_predictions.copy()
        else:
            best_images[better] = candidate_images[better]
            best_predictions[better] = candidate_predictions[better]

    return best_images, best_predictions, best_losses


def evaluate_configuration(classifier, validation_batches, config, parameters):
    """ASR/loss use clean-correct images; robust accuracy uses the full validation split."""
    attack = pgd_attack(classifier, config, parameters)
    epsilon = float(parameters["epsilon_255"]) / 255.0
    total = 0
    clean_correct = 0
    successful_attacks = 0
    loss_sum = 0.0
    linf_sum = 0.0
    maximum_linf = 0.0
    start_time = perf_counter()

    for images, labels in validation_batches:
        clean_outputs = classifier.predict(images)
        clean_predictions = np.argmax(clean_outputs, axis=1)
        eligible_mask = clean_predictions == labels
        total += len(labels)
        clean_correct += int(np.sum(eligible_mask))
        if not np.any(eligible_mask):
            continue

        eligible_images = images[eligible_mask]
        eligible_labels = labels[eligible_mask]
        adversarial_images, adversarial_outputs, losses = generate_best_restart(
            attack,
            classifier,
            eligible_images,
            eligible_labels,
            parameters,
            config,
        )
        adversarial_predictions = np.argmax(adversarial_outputs, axis=1)
        successful_attacks += int(
            np.sum(adversarial_predictions != eligible_labels)
        )
        perturbations = np.max(
            np.abs(adversarial_images - eligible_images), axis=(1, 2, 3)
        )
        observed_maximum = float(np.max(perturbations))
        if observed_maximum > epsilon + 1e-5:
            raise RuntimeError(
                f"PGD exceeded epsilon: {observed_maximum} > {epsilon}"
            )
        if np.min(adversarial_images) < -1e-6 or np.max(adversarial_images) > 1.0 + 1e-6:
            raise RuntimeError("PGD produced pixels outside [0, 1].")

        loss_sum += float(np.sum(losses))
        linf_sum += float(np.sum(perturbations))
        maximum_linf = max(maximum_linf, observed_maximum)

    robust_correct = clean_correct - successful_attacks
    if total == 0 or clean_correct == 0:
        raise ValueError("Validation split has no images correctly classified by the Phase 1 model")
    clean_accuracy = clean_correct / total
    adversarial_accuracy = robust_correct / total
    asr = successful_attacks / clean_correct
    return {
        **parameters,
        "alpha_255": float(parameters["epsilon_255"]) * float(parameters["alpha_ratio"]),
        "total_images": total,
        "clean_correct_images": clean_correct,
        "successful_attacks": successful_attacks,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "accuracy_drop": clean_accuracy - adversarial_accuracy,
        "attack_success_rate": asr,
        "mean_adversarial_loss": loss_sum / clean_correct,
        "mean_linf": linf_sum / clean_correct,
        "maximum_linf": maximum_linf,
        "runtime_seconds": perf_counter() - start_time,
    }


def generate_sweep_parameters(config: dict):
    sweep = config["sweep"]
    required = ("iterations", "epsilon_255", "alpha_ratio", "restarts")
    if any(not isinstance(sweep.get(key), list) or not sweep[key] for key in required):
        raise ValueError(f"sweep must contain nonempty lists for {required}")
    for iterations, epsilon_255, alpha_ratio, restarts in product(
        sweep["iterations"],
        sweep["epsilon_255"],
        sweep["alpha_ratio"],
        sweep["restarts"],
    ):
        parameters = {
            "iterations": int(iterations),
            "epsilon_255": float(epsilon_255),
            "alpha_ratio": float(alpha_ratio),
            "restarts": int(restarts),
        }
        if (
            parameters["iterations"] < 1 or parameters["restarts"] < 1
            or not 0 < parameters["epsilon_255"] <= 255
            or not 0 < parameters["alpha_ratio"] <= 1
        ):
            raise ValueError(f"Invalid PGD parameters: {parameters}")
        yield parameters


def save_iteration_plots(results_frame: pd.DataFrame, output_directory: Path) -> None:
    """At R=3, draw one curve per alpha; epsilon is fixed by this sweep."""
    subset = results_frame.loc[results_frame["random_restarts"] == 3]
    if subset.empty or subset["epsilon"].nunique() != 1:
        raise ValueError("R=3 plots require one epsilon and at least one R=3 configuration")
    if subset.duplicated(["iterations", "step_size"]).any():
        raise ValueError("R=3 plot points are not unique by iteration and step size")
    plot_specs = (
        ("mean_cross_entropy_loss", "Mean cross-entropy loss", "pgd_loss_vs_iterations.png"),
        ("robust_accuracy", "Robust accuracy", "pgd_robust_accuracy_vs_iterations.png"),
    )
    epsilon_255 = subset["epsilon"].iloc[0] * 255
    for metric, ylabel, filename in plot_specs:
        figure, axis = plt.subplots(figsize=(7, 5))
        for alpha, curve in subset.groupby("step_size", sort=True):
            curve = curve.sort_values("iterations")
            axis.plot(
                curve["iterations"], curve[metric], marker="o", linewidth=2,
                label=f"α = {alpha * 255:g}/255",
            )
        axis.set_xticks(sorted(subset["iterations"].unique()))
        axis.set_xlabel("PGD iterations (K)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} vs PGD iterations (ε={epsilon_255:g}/255, R=3)")
        if metric == "robust_accuracy":
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.set_ylim(0, 1)
        axis.legend(title="Step size")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_directory / filename, dpi=200)
        plt.close(figure)



def main():
    phase_directory = Path(__file__).resolve().parent
    repo_directory = phase_directory.parents[1]
    pgd_config = load_config(phase_directory / "configs" / "pgd.yaml")
    validate_attack_settings(pgd_config)
    validate_class_mapping(pgd_config, repo_directory)

    seed = int(pgd_config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    output_directory = resolve_repo_path(
        pgd_config["experiment"]["output_dir"], repo_directory
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    art_classifier = load_art_classifier(pgd_config, repo_directory)
    validation_batches = load_validation_batches(
        pgd_config, repo_directory
    )
    parameter_sets = list(generate_sweep_parameters(pgd_config))
    keys = [tuple(parameters.values()) for parameters in parameter_sets]
    if len(parameter_sets) != 36 or len(set(keys)) != 36:
        raise ValueError(f"Expected 36 unique PGD configurations; found {len(set(keys))}")

    results = []
    for index, parameters in enumerate(parameter_sets, start=1):
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        print(f"\nConfiguration {index}/{len(parameter_sets)}: {parameters}")
        result = evaluate_configuration(
            art_classifier,
            validation_batches,
            pgd_config,
            parameters,
        )
        results.append(result)
        print(
            f"ASR={result['attack_success_rate']:.4f}, "
            f"adversarial accuracy={result['adversarial_accuracy']:.4f}"
        )

    if len(results) != 36:
        raise RuntimeError(f"Only {len(results)} of 36 configurations completed")
    results_frame = pd.DataFrame([
        {
            "epsilon": result["epsilon_255"] / 255.0,
            "iterations": result["iterations"],
            "step_size": result["alpha_255"] / 255.0,
            "random_restarts": result["restarts"],
            "mean_cross_entropy_loss": result["mean_adversarial_loss"],
            "robust_accuracy": result["adversarial_accuracy"],
            "attack_success_rate": result["attack_success_rate"],
            "epsilon_255": result["epsilon_255"],
            "alpha_ratio": result["alpha_ratio"],
            "total_images": result["total_images"],
            "clean_correct_images": result["clean_correct_images"],
            "successful_attacks": result["successful_attacks"],
            "runtime_seconds": result["runtime_seconds"],
        }
        for result in results
    ])
    results_path = output_directory / "pgd_tuning_results.csv"
    results_frame.to_csv(results_path, index=False)
    save_iteration_plots(results_frame, output_directory)

    print(f"\nResults saved to: {results_path}")
    print(f"Plots saved to: {output_directory}")
    print("Select the PGD parameters manually and save them in configs/pgd_selected.yaml.")

if __name__ == "__main__":
    main()
