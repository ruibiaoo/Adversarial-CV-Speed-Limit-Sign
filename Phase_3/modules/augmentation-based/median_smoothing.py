"""Subcategory 1 (input transformation): median smoothing.

Test-time defence, no retraining. Each uint8 RGB image is median filtered
with a k x k window before preprocess_input, and the frozen model classifies
the filtered image.

Why median smoothing for this project:
- The black-box attack (Pixle) overwrites 3x3 patches of pixels. A median
  filter replaces each pixel with the median of its window, so a patch that
  covers less than half the window is removed. With k >= 5, a 3x3 patch
  (9 of 25 pixels) is removed completely.
- Against the white-box attack (L-inf PGD, eps 8/255), it averages out
  high-frequency noise like JPEG does, but it also handles the L0 attack, which
  JPEG does not.
- Speed-limit digits are large, high-contrast strokes. The median keeps edges,
  so clean accuracy holds up better than with Gaussian blur.

Calibration ("training") uses clean val only (PRD section 5): the largest k
whose clean-val accuracy is within TOLERANCE of no defence. The tolerance is
fixed here, before any results are seen.

  python modules/augmentation-based/median_smoothing.py                 # Phase 1 model
  python modules/augmentation-based/median_smoothing.py --model path/to/best.keras
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import _trainer as T

KERNELS = (3, 5, 7)
TOLERANCE = 0.01  # max clean-val accuracy drop (1 pt), fixed a priori


def median_smooth(image, kernel=3):
    """Median filter one uint8 HxWx3 image, per channel. Returns uint8."""
    return cv2.medianBlur(np.ascontiguousarray(image, dtype=np.uint8), int(kernel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="frozen model to calibrate on (default: Phase 1 model)")
    args = parser.parse_args()

    import tensorflow as tf

    config = T.load_config()
    T.setup_gpu()
    model_path = Path(args.model) if args.model else Path(config["phase1_model"])
    if not model_path.is_absolute():
        model_path = (T.PHASE3 / model_path).resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    val_ds = T.val_batches(config)

    base = T.evaluate(model, val_ds)
    rows = [{"kernel": 0, **base, "drop": 0.0}]
    print(f"no defence: val_acc={base['accuracy']:.4f} val_macro_f1={base['macro_f1']:.4f}")
    for k in KERNELS:
        m = T.evaluate(model, val_ds, transform=lambda im, k=k: median_smooth(im, k))
        rows.append({"kernel": k, **m, "drop": round(base["accuracy"] - m["accuracy"], 4)})
        print(f"median k={k}: val_acc={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f}")

    out = Path(config["runs_dir"]) / "median_smoothing_grid.csv"
    T.write_csv(out, rows)
    T.write_csv(T.TUNING / "median_smoothing" / "grid.csv", rows)
    ok = [r for r in rows[1:] if r["drop"] <= TOLERANCE]
    if not ok:  # nothing within tolerance: smallest kernel, flagged
        print("No kernel within tolerance, using the smallest one.")
    chosen = max(ok, key=lambda r: r["kernel"]) if ok else rows[1]
    T.save_choice("median_smoothing", {
        "params": {"kernel": chosen["kernel"]},
        "calibration_model": str(model_path.relative_to(T.PHASE3.parent)).replace("\\", "/"),
        "val_accuracy": round(chosen["accuracy"], 4),
        "val_macro_f1": round(chosen["macro_f1"], 4),
        "no_defence_val_accuracy": round(base["accuracy"], 4),
        "within_tolerance": bool(ok),
        "selected_on": f"largest kernel with clean val accuracy drop <= {TOLERANCE}",
    })
    print(f"Chosen kernel: {chosen['kernel']}  (grid saved to {out})")


if __name__ == "__main__":
    main()
