"""Build data/data_clean_pgd_pixle/: clean + PGD + Pixle copies of every train/val image, per class.

Sources (same images, same class folders, same file stems):
  data/data_updated/        clean
  data/data_updated_pgd/    white-box PGD copy of every train/val/test image (robust-training branch)
  data/data_updated_pixle/  black-box Pixle copy of every train/val/test image (robust-training branch)

Output layout:
  train/<class>/  clean_*, pgd_*, pixle_*   3 x 1080 images
  val/<class>/    clean_*, pgd_*, pixle_*   3 x 220 images   (defence tuning split)
  test/<class>/        clean test, original names        115 images
  test_pgd/<class>/    PGD copy of the test images        115 images
  test_pixle/<class>/  Pixle copy of the test images      115 images
  data.yaml       train: train, val: val, test: test (+ class names)

The filename prefix (clean_/pgd_/pixle_) is what src/defenses/modules/evaluation.source_of reads to
report per-source val scores. The three test sets stay separate so each is scored on its own.
Unlike data_combined/ on the robust-training branch, the PGD-on-Pixle copy (data_pgdpixle) is not
included.

Run from the repo root:

  python src/data_preparation/combine_clean_pgd_pixle.py
  python src/data_preparation/combine_clean_pgd_pixle.py --overwrite   # rebuild from scratch
"""
import argparse
import shutil
from pathlib import Path

DATA_DIR = Path("data")
SOURCES = {"clean": "data_updated", "pgd": "data_updated_pgd", "pixle": "data_updated_pixle"}
TARGET = DATA_DIR / "data_clean_pgd_pixle"
COMBINED_SPLITS = ["train", "val"]
TEST_SETS = {"test": "clean", "test_pgd": "pgd", "test_pixle": "pixle"}  # target folder -> source


def class_files(split_dir):
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            yield class_dir.name, path


def copy_split(src_dir, dst_dir, prefix=""):
    if not src_dir.is_dir():
        raise SystemExit(f"Missing source split: {src_dir}")
    n = 0
    for class_name, path in class_files(src_dir):
        out = dst_dir / class_name / f"{prefix}{path.name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true", help="delete and rebuild the target")
    args = parser.parse_args()
    if TARGET.exists():
        if not args.overwrite:
            raise SystemExit(f"{TARGET} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(TARGET)

    for split in COMBINED_SPLITS:
        counts = {src: copy_split(DATA_DIR / folder / split, TARGET / split, f"{src}_")
                  for src, folder in SOURCES.items()}
        print(f"{split}: {counts}")
    for target, src in TEST_SETS.items():
        print(f"{target}: {copy_split(DATA_DIR / SOURCES[src] / 'test', TARGET / target)} ({src})")

    meta = (DATA_DIR / SOURCES["clean"] / "data.yaml").read_text(encoding="utf-8")
    header = ("# Built by src/data_preparation/combine_clean_pgd_pixle.py.\n"
              "# train/ and val/ hold clean_, pgd_ and pixle_ copies of every image; test/ is clean only.\n"
              "# The attacked test copies are test_pgd/ and test_pixle/ (src/defenses/config.yaml test_sets).\n")
    (TARGET / "data.yaml").write_text(header + meta, encoding="utf-8")
    print(f"Done: {TARGET}")


if __name__ == "__main__":
    main()
