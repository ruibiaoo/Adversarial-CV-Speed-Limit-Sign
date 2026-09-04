"""
prepare_classification_data.py

Converts an object detection dataset (YOLO bounding box / segmentation format)
into a cropped image dataset ready for YOLO classification (Ultralytics YOLOv8/v11).

Key Features:
- Crops detected signs using normalized bbox or polygon annotations.
- Stratified splitting by class with optional group-based splitting
  to avoid data leakage between augmented/derived images.
- Parallel processing using multiprocessing / ProcessPoolExecutor for high throughput.
- Robust coordinate handling, EXIF orientation correction, and RGB conversion.
- Generates data.yaml in the output directory for direct use with `yolo classify train`.
"""

import argparse
import os
import re
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_base_name(filename: str) -> str:
    """
    Extracts the underlying base name of an image, stripping Roboflow hash
    suffixes (e.g. '.rf.d72176f98cada2a874d8a13f07869167') or extensions.
    This enables grouping augmented variants of the same physical sign.
    """
    stem = Path(filename).stem
    # Strip .rf.<hash> if present
    base = re.sub(r"\.rf\.[a-f0-9]+", "", stem, flags=re.IGNORECASE)
    # Strip double extension artifacts (e.g. '0000005_jpg')
    base = re.sub(r"_(jpg|jpeg|png)$", "", base, flags=re.IGNORECASE)
    return base


def load_yaml_names(yaml_path: str) -> Dict[int, str]:
    """Loads class index to name mapping from a YOLO data.yaml file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    raw_names = data.get("names", [])
    if isinstance(raw_names, list):
        return {i: str(name) for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    else:
        raise ValueError(f"Unexpected 'names' format in {yaml_path}: {type(raw_names)}")


def parse_label_file(label_path: str) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """
    Parses a YOLO annotation file. Supports:
    1. Standard 5-token YOLO bounding box: <class_id> <xc> <yc> <w> <h>
    2. Multi-point polygon/segmentation: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    Returns list of (class_id, (xmin, ymin, xmax, ymax)) in normalized coordinates [0, 1].
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 5:
                continue
            cls_id = int(tokens[0])
            coords = [float(x) for x in tokens[1:]]

            if len(coords) == 4:
                # Standard YOLO bbox: xc, yc, w, h
                xc, yc, w, h = coords
                xmin = xc - w / 2.0
                ymin = yc - h / 2.0
                xmax = xc + w / 2.0
                ymax = yc + h / 2.0
            else:
                # Polygon coordinates: x1, y1, x2, y2, ...
                xs = coords[0::2]
                ys = coords[1::2]
                xmin = min(xs)
                ymin = min(ys)
                xmax = max(xs)
                ymax = max(ys)

            # Clamp normalized coords to [0.0, 1.0]
            xmin = max(0.0, min(1.0, xmin))
            ymin = max(0.0, min(1.0, ymin))
            xmax = max(0.0, min(1.0, xmax))
            ymax = max(0.0, min(1.0, ymax))

            if xmax > xmin and ymax > ymin:
                boxes.append((cls_id, (xmin, ymin, xmax, ymax)))

    return boxes


def crop_single_task(task_args: dict) -> Tuple[bool, str]:
    """
    Worker task to crop and save sign images from one source image.
    Args dictionary contains:
        img_path: Path to source image
        boxes: List of (class_id, (xmin, ymin, xmax, ymax))
        dest_dirs: Mapping of class_id to target directory
        padding: Relative padding ratio around bounding box
        target_size: Optional (width, height) to resize
    """
    img_path = task_args["img_path"]
    boxes = task_args["boxes"]
    dest_dirs = task_args["dest_dirs"]
    padding = task_args.get("padding", 0.0)
    target_size = task_args.get("target_size", None)

    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            img_w, img_h = im.size

            multi_box = len(boxes) > 1
            stem = Path(img_path).stem
            ext = Path(img_path).suffix or ".jpg"

            for idx, (cls_id, (n_xmin, n_ymin, n_xmax, n_ymax)) in enumerate(boxes):
                # Convert normalized coords to pixel coords
                px_xmin = n_xmin * img_w
                px_ymin = n_ymin * img_h
                px_xmax = n_xmax * img_w
                px_ymax = n_ymax * img_h

                box_w = px_xmax - px_xmin
                box_h = px_ymax - px_ymin

                if padding > 0.0:
                    pad_x = box_w * padding
                    pad_y = box_h * padding
                    px_xmin = max(0.0, px_xmin - pad_x)
                    px_ymin = max(0.0, px_ymin - pad_y)
                    px_xmax = min(float(img_w), px_xmax + pad_x)
                    px_ymax = min(float(img_h), px_ymax + pad_y)

                crop_box = (
                    int(round(px_xmin)),
                    int(round(px_ymin)),
                    int(round(px_xmax)),
                    int(round(px_ymax)),
                )

                # Ensure non-zero crop area
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    continue

                crop = im.crop(crop_box)

                if target_size:
                    crop = crop.resize(target_size, Image.Resampling.LANCZOS)

                # Ensure RGB mode for JPEG saving
                if crop.mode != "RGB":
                    crop = crop.convert("RGB")

                dest_dir = dest_dirs[cls_id]
                if multi_box:
                    out_filename = f"{stem}_crop{idx}{ext}"
                else:
                    out_filename = f"{stem}{ext}"

                out_path = os.path.join(dest_dir, out_filename)
                crop.save(out_path, format="JPEG", quality=95)

        return True, img_path
    except Exception as e:
        return False, f"Error processing {img_path}: {e}"


def split_dataset(
    samples_per_class: Dict[int, List[dict]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    min_test_per_class: int = 5,
    group_by_base: bool = True,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """
    Splits samples into 'train', 'val', (and optionally 'test') splits.
    If group_by_base is True, groups samples with the same base name so that
    augmented versions of the same sign are never split across subsets.
    Ensures that the test set has at least min_test_per_class images per class.
    """
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    for cls_id, samples in samples_per_class.items():
        if group_by_base:
            base_groups = defaultdict(list)
            for s in samples:
                base = get_base_name(s["img_name"])
                base_groups[base].append(s)

            groups = list(base_groups.values())
            rng.shuffle(groups)

            total_cls_samples = len(samples)
            target_test = max(min_test_per_class, int(round(total_cls_samples * test_ratio))) if test_ratio > 0 else 0
            target_val = int(round(total_cls_samples * val_ratio)) if val_ratio > 0 else 0

            val_group_samples = []
            test_group_samples = []
            train_group_samples = []

            cur_test = 0
            cur_val = 0

            for g in groups:
                # Prioritize satisfying the minimum test quota first, then target test
                if test_ratio > 0 and (cur_test < target_test or cur_test < min_test_per_class):
                    test_group_samples.extend(g)
                    cur_test += len(g)
                elif val_ratio > 0 and cur_val < target_val:
                    val_group_samples.extend(g)
                    cur_val += len(g)
                else:
                    train_group_samples.extend(g)

            splits["train"].extend(train_group_samples)
            splits["val"].extend(val_group_samples)
            splits["test"].extend(test_group_samples)
        else:
            cls_samples = list(samples)
            rng.shuffle(cls_samples)
            n = len(cls_samples)
            n_test = max(min_test_per_class, int(round(n * test_ratio))) if test_ratio > 0 else 0
            n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0

            n_test = min(n, n_test)
            n_val = min(n - n_test, n_val)

            splits["test"].extend(cls_samples[:n_test])
            splits["val"].extend(cls_samples[n_test : n_test + n_val])
            splits["train"].extend(cls_samples[n_test + n_val :])

    return splits


def run_pipeline(
    raw_dir: str,
    output_dir: str,
    yaml_file: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    min_test_per_class: int = 5,
    no_split: bool = False,
    group_by_base: bool = True,
    padding: float = 0.0,
    target_size: Optional[Tuple[int, int]] = None,
    replace_spaces: bool = True,
    workers: int = 8,
    seed: int = 42,
    clean: bool = True,
):
    raw_path = Path(raw_dir).resolve()
    output_path = Path(output_dir).resolve()
    images_dir = raw_path / "images"
    labels_dir = raw_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Images or labels folder missing in {raw_path}. Expected {images_dir} and {labels_dir}."
        )

    # Resolve data.yaml
    if yaml_file:
        yaml_path = Path(yaml_file).resolve()
    else:
        yaml_path = raw_path / "data.yaml"

    class_names = load_yaml_names(str(yaml_path))
    print(f"Loaded {len(class_names)} classes from {yaml_path.name}: {class_names}")

    # Sanitize class names if requested
    sanitized_names = {}
    for cid, name in class_names.items():
        if replace_spaces:
            sanitized_names[cid] = name.replace(" ", "_")
        else:
            sanitized_names[cid] = name

    # Index image files
    img_files = {}
    for f in os.listdir(images_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            stem = os.path.splitext(f)[0]
            img_files[stem] = f

    print(f"Found {len(img_files)} images in {images_dir}.")

    # Gather label tasks
    samples_per_class = defaultdict(list)
    total_boxes = 0

    for stem, img_fname in img_files.items():
        lbl_file = labels_dir / f"{stem}.txt"
        if not lbl_file.exists():
            continue

        boxes = parse_label_file(str(lbl_file))
        if not boxes:
            continue

        # Group by primary class for splitting (or all classes)
        primary_cls = boxes[0][0]
        sample = {
            "img_name": img_fname,
            "img_path": str(images_dir / img_fname),
            "boxes": boxes,
            "primary_class": primary_cls,
        }
        samples_per_class[primary_cls].append(sample)
        total_boxes += len(boxes)

    print(f"Parsed {sum(len(v) for v in samples_per_class.values())} labeled images ({total_boxes} bounding boxes).")

    # Clean output dir if requested
    if clean and output_path.exists():
        print(f"Cleaning existing contents in output directory: {output_path}")
        for item in output_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    output_path.mkdir(parents=True, exist_ok=True)

    # Determine splits and directory structure
    task_list = []
    split_counts = defaultdict(lambda: defaultdict(int))

    if no_split:
        # Flat structure: data_processed/<class_name>/
        for cid, s_name in sanitized_names.items():
            (output_path / s_name).mkdir(parents=True, exist_ok=True)

        for cls_id, samples in samples_per_class.items():
            for s in samples:
                dest_dirs = {cid: str(output_path / sanitized_names[cid]) for cid in class_names.keys()}
                task_list.append({
                    "img_path": s["img_path"],
                    "boxes": s["boxes"],
                    "dest_dirs": dest_dirs,
                    "padding": padding,
                    "target_size": target_size,
                })
                for cid, _ in s["boxes"]:
                    split_counts["all"][sanitized_names[cid]] += 1
    else:
        # YOLO classification format: data_processed/<split>/<class_name>/
        splits = split_dataset(
            samples_per_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            min_test_per_class=min_test_per_class,
            group_by_base=group_by_base,
            seed=seed,
        )

        for split_name, split_samples in splits.items():
            if not split_samples:
                continue
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for cid, s_name in sanitized_names.items():
                (split_dir / s_name).mkdir(parents=True, exist_ok=True)

            for s in split_samples:
                dest_dirs = {cid: str(split_dir / sanitized_names[cid]) for cid in class_names.keys()}
                task_list.append({
                    "img_path": s["img_path"],
                    "boxes": s["boxes"],
                    "dest_dirs": dest_dirs,
                    "padding": padding,
                    "target_size": target_size,
                })
                for cid, _ in s["boxes"]:
                    split_counts[split_name][sanitized_names[cid]] += 1

    print(f"\nDispatching {len(task_list)} image crop tasks across {workers} worker processes...")

    # Parallel Execution
    successful = 0
    failed = 0
    errors = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(crop_single_task, task) for task in task_list]
            if tqdm:
                for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping dataset"):
                    ok, msg = future.result()
                    if ok:
                        successful += 1
                    else:
                        failed += 1
                        errors.append(msg)
            else:
                for future in as_completed(futures):
                    ok, msg = future.result()
                    if ok:
                        successful += 1
                    else:
                        failed += 1
                        errors.append(msg)
                    if (successful + failed) % 100 == 0 or (successful + failed) == len(futures):
                        print(f"Processed {successful + failed}/{len(futures)} images...")
    else:
        # Serial fallback
        for task in task_list:
            ok, msg = crop_single_task(task)
            if ok:
                successful += 1
            else:
                failed += 1
                errors.append(msg)

    # Generate data.yaml in output directory for YOLO classification
    out_yaml = {
        "path": ".",
        "nc": len(class_names),
        "names": {cid: sanitized_names[cid] for cid in sorted(class_names.keys())},
    }
    if not no_split:
        out_yaml["train"] = "train"
        out_yaml["val"] = "val"
        if test_ratio > 0:
            out_yaml["test"] = "test"

    yaml_out_path = output_path / "data.yaml"
    with open(yaml_out_path, "w", encoding="utf-8") as f:
        yaml.dump(out_yaml, f, sort_keys=False)

    # Print summary table
    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Output Directory: {output_path}")
    print(f"YAML Config:      {yaml_out_path}")
    print(f"Successful crops: {successful}")
    print(f"Failed crops:     {failed}")

    if errors:
        print("\nErrors encountered:")
        for err in errors[:5]:
            print(f" - {err}")
        if len(errors) > 5:
            print(f" ... and {len(errors) - 5} more.")

    print("\nClass Distribution per Split:")
    all_classes = [sanitized_names[cid] for cid in sorted(class_names.keys())]
    header = f"{'Class Name':<15}" + "".join([f"{s.upper():<12}" for s in split_counts.keys()]) + f"{'TOTAL':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_per_split = defaultdict(int)
    for cname in all_classes:
        row = f"{cname:<15}"
        row_total = 0
        for sname in split_counts.keys():
            cnt = split_counts[sname][cname]
            total_per_split[sname] += cnt
            row_total += cnt
            row += f"{cnt:<12}"
        row += f"{row_total:<10}"
        print(row)

    print("-" * len(header))
    totals_row = f"{'TOTAL':<15}" + "".join([f"{total_per_split[s]:<12}" for s in split_counts.keys()]) + f"{sum(total_per_split.values()):<10}"
    print(totals_row)
    print("=" * 60)
    print("\nReady for YOLO classification training! Example command:")
    print(f"  yolo classify train data=\"{output_path}\" model=yolov8n-cls.pt epochs=50 imgsz=224")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a detection dataset into cropped images for YOLO classification."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data_raw",
        help="Path to raw detection dataset containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_processed",
        help="Path to output directory for classification dataset.",
    )
    parser.add_argument(
        "--yaml-file",
        type=str,
        default=None,
        help="Optional custom path to data.yaml (default: <raw_dir>/data.yaml).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of dataset for training (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of dataset for validation (default: 0.2).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of dataset for testing (default: 0.1).",
    )
    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=5,
        help="Minimum number of test images required per class (default: 5).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable train/val splitting and put all cropped images directly into class folders.",
    )
    parser.add_argument(
        "--no-group-by-base",
        action="store_true",
        help="Disable grouping augmented variants of base images when splitting.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Relative padding ratio around bounding box before cropping (default: 0.0).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional resize resolution (e.g. --target-size 224 224). Default is native crop size.",
    )
    parser.add_argument(
        "--keep-spaces",
        action="store_true",
        help="Keep spaces in class folder names instead of replacing them with underscores.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Number of parallel worker processes (default: max 16 or cpu count).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting (default: 42).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean output directory before running.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        yaml_file=args.yaml_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_test_per_class=args.min_test_per_class,
        no_split=args.no_split,
        group_by_base=not args.no_group_by_base,
        padding=args.padding,
        target_size=tuple(args.target_size) if args.target_size else None,
        replace_spaces=not args.keep_spaces,
        workers=args.workers,
        seed=args.seed,
        clean=not args.no_clean,
    )
