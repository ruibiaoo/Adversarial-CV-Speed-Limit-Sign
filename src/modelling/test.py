import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.resnet import preprocess_input


def load_config(path):
    """Load the test settings from yaml."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(value, base_directory):
    """Use Phase_1 as the base for relative paths."""
    path = Path(value)
    if path.is_absolute():
        return path
    return base_directory / path


def build_test_dataset(config):
    """Load the test split without shuffling or augmentation."""
    image_size = int(config["image_size"])
    batch = int(config["batch"])
    class_names = list(config["class_names"])
    test_dir = Path(config["data"]) / config["split"]

    dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(image_size, image_size),
        batch_size=batch,
        shuffle=False,
    )
    file_paths = list(dataset.file_paths)

    def preprocess(image, label):
        image = preprocess_input(tf.cast(image, tf.float32))
        return image, label

    dataset = dataset.map(preprocess, num_parallel_calls=1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, file_paths


def calculate_metrics(true_labels, predicted_labels, probabilities, class_names):
    """Calculate overall and per-class classification metrics."""
    labels = list(range(len(class_names)))
    metrics = {
        "number_of_test_images": len(true_labels),
        "cross_entropy_loss": float(
            log_loss(true_labels, probabilities, labels=labels)
        ),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "top_3_accuracy": float(
            top_k_accuracy_score(
                true_labels,
                probabilities,
                k=min(3, len(class_names)),
                labels=labels,
            )
        ),
        "top_5_accuracy": float(
            top_k_accuracy_score(
                true_labels,
                probabilities,
                k=min(5, len(class_names)),
                labels=labels,
            )
        ),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(true_labels, predicted_labels)
        ),
        "cohen_kappa": float(
            cohen_kappa_score(true_labels, predicted_labels)
        ),
    }

    for average in ("macro", "micro", "weighted"):
        metrics[f"precision_{average}"] = float(
            precision_score(
                true_labels,
                predicted_labels,
                average=average,
                zero_division=0,
            )
        )
        metrics[f"recall_{average}"] = float(
            recall_score(
                true_labels,
                predicted_labels,
                average=average,
                zero_division=0,
            )
        )
        metrics[f"f1_{average}"] = float(
            f1_score(
                true_labels,
                predicted_labels,
                average=average,
                zero_division=0,
            )
        )

    binary_labels = label_binarize(true_labels, classes=labels)
    try:
        metrics["roc_auc_macro_ovr"] = float(
            roc_auc_score(
                true_labels,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        metrics["roc_auc_weighted_ovr"] = float(
            roc_auc_score(
                true_labels,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError as error:
        metrics["roc_auc_note"] = f"Not available: {error}"

    try:
        metrics["average_precision_macro"] = float(
            average_precision_score(binary_labels, probabilities, average="macro")
        )
        metrics["average_precision_weighted"] = float(
            average_precision_score(binary_labels, probabilities, average="weighted")
        )
    except ValueError as error:
        metrics["average_precision_note"] = f"Not available: {error}"

    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        zero_division=0,
    )

    per_class = {}
    total = matrix.sum()
    for index, class_name in enumerate(class_names):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        specificity_total = true_negative + false_positive
        specificity = (
            true_negative / specificity_total if specificity_total > 0 else 0.0
        )

        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "specificity": float(specificity),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return metrics, per_class, matrix


def save_confusion_matrix(path, matrix, class_names, normalized=False):
    """Save a labelled confusion matrix as a text table."""
    name_width = max(12, max(len(name) for name in class_names) + 2)

    with path.open("w", encoding="utf-8") as file:
        title = "Normalized confusion matrix" if normalized else "Confusion matrix"
        file.write(f"{title}\n\n")
        file.write("Actual \\ Predicted".ljust(name_width))
        for name in class_names:
            file.write(name.rjust(name_width))
        file.write("\n")

        for row_index, name in enumerate(class_names):
            file.write(name.ljust(name_width))
            for value in matrix[row_index]:
                if normalized:
                    file.write(f"{value:.4f}".rjust(name_width))
                else:
                    file.write(str(int(value)).rjust(name_width))
            file.write("\n")


def save_metrics(path, metrics, per_class):
    """Save overall and per-class metrics as readable text."""
    with path.open("w", encoding="utf-8") as file:
        file.write("OVERALL TEST METRICS\n")
        file.write("====================\n")
        for name, value in metrics.items():
            if isinstance(value, float):
                file.write(f"{name}: {value:.6f}\n")
            else:
                file.write(f"{name}: {value}\n")

        file.write("\nPER-CLASS METRICS\n")
        file.write("=================\n")
        for class_name, class_metrics in per_class.items():
            file.write(f"\n{class_name}\n")
            for name, value in class_metrics.items():
                if isinstance(value, float):
                    file.write(f"  {name}: {value:.6f}\n")
                else:
                    file.write(f"  {name}: {value}\n")


def save_predictions(
    path,
    file_paths,
    true_labels,
    predicted_labels,
    probabilities,
    class_names,
):
    """Save every prediction and class probability to CSV."""
    probability_columns = [f"probability_{name}" for name in class_names]
    columns = [
        "file_path",
        "actual_class",
        "predicted_class",
        "confidence",
        "correct",
        *probability_columns,
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for index, file_path in enumerate(file_paths):
            actual = int(true_labels[index])
            predicted = int(predicted_labels[index])
            row = {
                "file_path": file_path,
                "actual_class": class_names[actual],
                "predicted_class": class_names[predicted],
                "confidence": float(probabilities[index, predicted]),
                "correct": actual == predicted,
            }
            for class_index, class_name in enumerate(class_names):
                row[f"probability_{class_name}"] = float(
                    probabilities[index, class_index]
                )
            writer.writerow(row)


def load_classifier(model_path, image_size=256, num_classes=7):
    """Load model, falling back to architecture reconstruction for Keras 2/3 cross-compatibility."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        pass

    import tempfile
    import zipfile

    if __package__:
        from .modules.model import build_model
    else:
        try:
            from modules.model import build_model
        except ModuleNotFoundError:
            import sys
            sys.path.append(str(Path(__file__).resolve().parent))
            from modules.model import build_model

    model = build_model(
        {"model_name": "resnet50", "image_size": image_size, "pretrained": False},
        num_classes=num_classes,
    )
    if zipfile.is_zipfile(model_path):
        with zipfile.ZipFile(model_path, "r") as z:
            with tempfile.TemporaryDirectory() as tmp_dir:
                extracted_path = z.extract("model.weights.h5", tmp_dir)
                model.load_weights(extracted_path)
    else:
        model.load_weights(model_path)
    return model


def _import_convae_helpers():
    if __package__:
        from .modules.convae import (
            is_wrapped_with_convae,
            load_convae,
            parse_convae_config,
            resolve_convae_path,
            wrap_with_convae,
        )
    else:
        try:
            from modules.convae import (
                is_wrapped_with_convae,
                load_convae,
                parse_convae_config,
                resolve_convae_path,
                wrap_with_convae,
            )
        except ModuleNotFoundError:
            import sys

            sys.path.append(str(Path(__file__).resolve().parent))
            from modules.convae import (
                is_wrapped_with_convae,
                load_convae,
                parse_convae_config,
                resolve_convae_path,
                wrap_with_convae,
            )
    return (
        is_wrapped_with_convae,
        load_convae,
        parse_convae_config,
        resolve_convae_path,
        wrap_with_convae,
    )


def main():
    phase_directory = Path(__file__).resolve().parent
    config = load_config(phase_directory / "configs" / "test.yaml")

    config["model_path"] = resolve_path(
        config["model_path"], phase_directory
    )
    config["data"] = resolve_path(config["data"], phase_directory)
    output_directory = resolve_path(config["output_dir"], phase_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    model_path = Path(config["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    image_size = int(config["image_size"])
    num_classes = len(config["class_names"])
    model = load_classifier(model_path, image_size=image_size, num_classes=num_classes)

    (
        is_wrapped_with_convae,
        load_convae,
        parse_convae_config,
        resolve_convae_path,
        wrap_with_convae,
    ) = _import_convae_helpers()
    convae_cfg = parse_convae_config(config)
    convae_used_path = None
    if convae_cfg["enabled"]:
        if is_wrapped_with_convae(model):
            print(
                f"Note: {model_path} already contains a ConvAE preprocessor; "
                "using its embedded ConvAE without wrapping a second time."
            )
            convae_used_path = f"embedded ({model_path})"
        else:
            raw_ae_path = convae_cfg["model_path"]
            if not raw_ae_path:
                raise ValueError(
                    "convae.enabled is true in test.yaml, but convae.model_path is not set."
                )
            resolved_ae = resolve_convae_path(resolve_path(raw_ae_path, phase_directory))
            convae = load_convae(resolved_ae, img_size=image_size)
            model = wrap_with_convae(model, convae, trainable=False)
            convae_used_path = str(resolved_ae)
            print(f"ConvAE preprocessing enabled for evaluation: {resolved_ae}")

    input_height = model.input_shape[1]
    input_width = model.input_shape[2]
    if input_height != image_size or input_width != image_size:
        raise ValueError(
            f"Model expects {input_height}x{input_width} images, "
            f"but test.yaml uses {image_size}x{image_size}."
        )

    test_ds, file_paths = build_test_dataset(config)
    true_labels = np.concatenate(
        [labels.numpy() for _, labels in test_ds]
    )
    probabilities = model.predict(test_ds, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    class_names = list(config["class_names"])
    metrics, per_class, matrix = calculate_metrics(
        true_labels,
        predicted_labels,
        probabilities,
        class_names,
    )
    if convae_used_path is not None:
        metrics = {"convae_preprocessor": convae_used_path, **metrics}

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    save_confusion_matrix(
        output_directory / "confusion_matrix.txt",
        matrix,
        class_names,
    )
    save_confusion_matrix(
        output_directory / "confusion_matrix_normalized.txt",
        normalized_matrix,
        class_names,
        normalized=True,
    )
    save_metrics(
        output_directory / "test_metrics.txt",
        metrics,
        per_class,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=6,
        zero_division=0,
    )
    (output_directory / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    save_predictions(
        output_directory / "predictions.csv",
        file_paths,
        true_labels,
        predicted_labels,
        probabilities,
        class_names,
    )

    print("\n--- TEST RESULTS ---")
    print(f"Model: {model_path}")
    if convae_used_path is not None:
        print(f"ConvAE preprocessor: {convae_used_path}")
    print(f"Test images: {metrics['number_of_test_images']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    main()
