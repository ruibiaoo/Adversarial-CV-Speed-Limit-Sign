import gc
import os
from pathlib import Path

import tensorflow as tf
import wandb
import yaml
from sklearn.metrics import f1_score

if __package__:
    from .modules.data import build_datasets, compute_class_weights
    from .modules.model import build_model, get_optimizer
    from .modules.tuning_config import prepare_sweep
    from .modules.tuning_results import make_output_directory, update_live_results
    from .modules.wandb_config import prepare_wandb_training
else:
    from modules.data import build_datasets, compute_class_weights
    from modules.model import build_model, get_optimizer
    from modules.tuning_config import prepare_sweep
    from modules.tuning_results import make_output_directory, update_live_results
    from modules.wandb_config import prepare_wandb_training


def load_config(path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def clear_training_memory():
    """Clear old Keras models after a training run."""
    tf.keras.backend.clear_session()
    gc.collect()


class ValidationMacroF1(tf.keras.callbacks.Callback):
    """Calculate macro F1 on the validation set after each epoch."""

    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        true_labels = []
        predicted_labels = []

        for images, labels in self.val_ds:
            predictions = self.model.predict_on_batch(images)
            predicted = tf.argmax(predictions, axis=1)
            true_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.numpy())

        score = f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
        logs = logs if logs is not None else {}
        logs["val_macro_f1"] = float(score)
        print(f" - val_macro_f1: {score:.4f}")


class WandbLog(tf.keras.callbacks.Callback):
    """Log training and validation metrics to W&B after each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        learning_rate = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "loss": logs.get("loss"),
            "accuracy": logs.get("accuracy"),
            "top5_acc": logs.get("top5_acc"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy"),
            "val_top5_acc": logs.get("val_top5_acc"),
            "val_macro_f1": logs.get("val_macro_f1"),
        })


def train_once(cfg, track):
    """Train one model from a config dict, return best validation macro F1."""
    seed = int(cfg.get("seed", 4266))
    tf.keras.utils.set_random_seed(seed)

    # Let GPU grow instead of grabbing all memory at once.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    train_ds, val_ds, class_names, num_classes = build_datasets(cfg)

    # rare signs get bigger loss so model does not just guess 40_MPH
    class_weights = compute_class_weights(cfg, class_names)

    model = build_model(cfg, num_classes)
    optimizer = get_optimizer(cfg)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )

    epochs = int(cfg.get("epochs", 50))
    project = cfg.get("project", "/app/runs")
    name = cfg.get("name", "tf_resnet_speed_limit")
    save_dir = Path(project) / name
    # sweep trials share the same name, so give each run its own folder
    # single training keeps project/name, sweep gets project/name/run_id
    if track:
        run_id = getattr(getattr(wandb, "run", None), "id", None)
        if run_id:
            save_dir = save_dir / str(run_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    reduce_lr_factor = float(cfg.get("reduce_lr_factor", 0.5))
    reduce_lr_patience = int(cfg.get("reduce_lr_patience", 2))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 5))
    min_learning_rate = float(cfg.get("min_learning_rate", 0.000001))

    callbacks = [
        ValidationMacroF1(val_ds),
        tf.keras.callbacks.ModelCheckpoint(
            str(save_dir / "best.keras"),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=early_stopping_patience,
            verbose=1,
        ),
    ]

    if track:
        callbacks.append(WandbLog())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    val_macro_f1 = history.history.get("val_macro_f1", [0.0])
    best_val = float(max(val_macro_f1)) if val_macro_f1 else 0.0
    print(f"Best validation macro F1: {best_val:.4f}")
    print(f"Best model saved to: {save_dir / 'best.keras'}")

    return best_val


def run_sweep_trial(base, output_directory):
    """Run one W&B sweep trial and always clean it up."""
    run = None
    succeeded = False
    try:
        run = wandb.init(
            group=os.environ.get("WANDB_RUN_GROUP"),
        )
        run.name = f"{base.get('name', 'sweep')}_{run.id}"

        cfg = base.copy()
        cfg.update(dict(wandb.config))
        best = train_once(cfg, track=True)
        run.log({"val_macro_f1": best})

        update_live_results(
            output_directory,
            run.id,
            run.name,
            best,
            cfg,
        )
        succeeded = True
        return best
    finally:
        try:
            if run is not None:
                run.finish(exit_code=0 if succeeded else 1)
        finally:
            clear_training_memory()


def start_sweep_agent(sweep_id, iterations, base, output_directory):
    """Start the W&B agent and run each requested trial."""
    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]

    def trial_function():
        run_sweep_trial(base, output_directory)

    wandb.agent(
        sweep_id,
        function=trial_function,
        entity=entity,
        project=project,
        count=iterations,
    )


def main():
    phase_directory = Path(__file__).resolve().parent
    config = load_config(phase_directory / "config.yaml")
    tuner_config = load_config(phase_directory / "tuner.yaml")
    tuning = tuner_config.get("tuner", False)

    if tuning:
        sweep, fixed, iterations = prepare_sweep(tuner_config)

        # fixed has data/epochs/augments, but not model_name etc.
        # So keep config.yaml as base, fixed overrides it, sweep overrides both.
        base = config.copy()
        base.pop("wandb", None)
        base.update(fixed)

        sweep_id = wandb.sweep(
            sweep,
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
        )
        print(f"Sweep created: {sweep_id}")

        # one folder for the whole sweep, live updated after each trial
        output_directory = make_output_directory(phase_directory / "tuning")
        print(f"Live tuning summary at: {output_directory}")

        start_sweep_agent(sweep_id, iterations, base, output_directory)
        print(f"\nSweep done, final summary at: {output_directory}")
        return

    track = prepare_wandb_training(config)
    if track:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=config.get("name"),
            group=os.environ.get("WANDB_RUN_GROUP"),
            config=config,
            job_type="train",
        )

    train_once(config, track=track)

    if track:
        wandb.finish()


if __name__ == "__main__":
    main()
