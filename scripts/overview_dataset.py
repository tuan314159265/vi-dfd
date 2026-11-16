
from pathlib import Path
from collections import Counter

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset"

SPLITS = {
    "Train": "train_images",
    "Test": "test_image",
    "Unseen Test": "unseen_test_images",
}

def main():
    print("=" * 50)
    print("VINTEXT DATASET OVERVIEW")
    print("=" * 50)
    print(f"Dataset root: {DATASET_ROOT}\n")

    total_images = 0

    # 1. Image overview
    print("[IMAGE SUMMARY]")

    for split, folder_name in SPLITS.items():
        folder = DATASET_ROOT / folder_name
        images = list(folder.glob("*.jpg"))
        total_images += len(images)

        print(f"{split:<15}: {len(images)} images")

    print(f"{'Total':<15}: {total_images} images")

    # 2. Annotation overview
    label_dir = DATASET_ROOT / "labels"
    label_files = sorted(label_dir.glob("*.txt"))

    total_instances = 0
    ignored_instances = 0
    valid_instances = 0
    malformed_rows = 0

    for label_file in label_files:
        with label_file.open(
            "r", encoding="utf-8-sig"
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split(",", maxsplit=8)

                if len(parts) != 9:
                    malformed_rows += 1
                    continue

                total_instances += 1
                transcript = parts[8].strip()

                if transcript == "###":
                    ignored_instances += 1
                elif transcript:
                    valid_instances += 1

    print("\n[ANNOTATION SUMMARY]")
    print(f"Annotation files  : {len(label_files)}")
    print(f"Total instances   : {total_instances}")
    print(f"Text instances    : {valid_instances}")
    print(f"Ignored (###)     : {ignored_instances}")
    print(f"Malformed rows    : {malformed_rows}")

    # 3. File types
    print("\n[FILE FORMATS]")

    for folder_name in [*SPLITS.values(), "labels"]:
        folder = DATASET_ROOT / folder_name
        extensions = Counter(
            p.suffix.lower()
            for p in folder.iterdir()
            if p.is_file()
        )
        print(f"{folder_name:<20}: {dict(extensions)}")

    # 4. Sample annotation
    sample_file = label_dir / "gt_1.txt"

    if sample_file.exists():
        print("\n[ANNOTATION SAMPLE: gt_1.txt]")

        with sample_file.open(
            "r", encoding="utf-8-sig"
        ) as f:
            for _, line in zip(range(5), f):
                print(line.strip())

    print("\n" + "=" * 50)
    print("OVERVIEW COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
