from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

DATA_DIR = Path("data")
SOURCE_DATASETS = [
    "data_updated",
    "data_updated_pgd",
    "data_updated_pixle",
    "data_pgdpixle",
]
TARGET_DIR = DATA_DIR / "data_combined"
SPLITS = ["train", "val", "test"]


def copy_file(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)


def main() -> None:
    tasks = []

    for ds_name in SOURCE_DATASETS:
        ds_dir = DATA_DIR / ds_name
        if not ds_dir.exists():
            print(f"Skipping missing directory: {ds_dir}")
            continue

        for split in SPLITS:
            split_dir = ds_dir / split
            if not split_dir.exists():
                continue

            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                dest_class_dir = TARGET_DIR / split / class_dir.name
                dest_class_dir.mkdir(parents=True, exist_ok=True)

                for file_path in class_dir.iterdir():
                    if file_path.is_file():
                        # Prefix with dataset name to avoid filename collisions
                        dest_path = dest_class_dir / f"{ds_name}_{file_path.name}"
                        tasks.append((file_path, dest_path))

    print(f"Copying {len(tasks)} files into {TARGET_DIR} using multi-threading...")
    with ThreadPoolExecutor() as executor:
        list(executor.map(lambda pair: copy_file(*pair), tasks))

    # Copy data.yaml from the first available dataset
    for ds_name in SOURCE_DATASETS:
        yaml_src = DATA_DIR / ds_name / "data.yaml"
        if yaml_src.exists():
            shutil.copy2(yaml_src, TARGET_DIR / "data.yaml")
            break

    print(f"Done! Combined dataset created at: {TARGET_DIR}")


if __name__ == "__main__":
    main()

