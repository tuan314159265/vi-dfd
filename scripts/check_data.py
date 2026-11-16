"""Check that this machine has the group's reference data build (and teacher) before running.

    python scripts/check_data.py --data-root <processed>/vintext_notebook [--teacher-checkpoint <.ckpt>]
    python scripts/check_data.py --data-root <reference root> --compare-assignments <other root>

Compares SHA-256 hashes with configs/data_reference.json. Exit code 1 on any mismatch.
--compare-assignments checks that another data build (e.g. the one a teacher was trained on)
puts every image into the same split as the reference build; it reads only image ids.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assignments(root: Path) -> dict[str, str]:
    path = root / "splits/manifests/image_split_assignments.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["image_id"]: row["final_split"] for row in csv.DictReader(handle)}


def compare_assignments(reference_root: Path, other_root: Path) -> int:
    reference, other = assignments(reference_root), assignments(other_root)
    differ = sorted((image for image in reference.keys() | other.keys() if reference.get(image) != other.get(image)),
                    key=lambda image: int(image))
    for image in differ[:20]:
        print(f"image {image}: reference {reference.get(image)}, other {other.get(image)}")
    print(f"{len(reference) - len(differ)}/{len(reference)} images in the same split; {len(differ)} differ")
    return 1 if differ else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="Folder containing lmdb/ and splits/")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "configs/data_reference.json")
    parser.add_argument("--compare-assignments", type=Path, metavar="OTHER_ROOT",
                        help="Another data root whose image-to-split assignment must equal --data-root's")
    args = parser.parse_args()
    if args.compare_assignments:
        return compare_assignments(args.data_root, args.compare_assignments)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    checks = []

    freeze = args.data_root / "splits" / "freeze_manifest.json"
    checks.append(("splits/freeze_manifest.json", freeze, reference["freeze_manifest_sha256"]))
    if freeze.exists():
        manifest = json.loads(freeze.read_text(encoding="utf-8"))
        for name, expected in manifest["manifest_sha256"].items():
            checks.append((f"splits/{name}", args.data_root / "splits" / name, expected))
    for split, expected in reference["lmdb_data_mdb_sha256"].items():
        checks.append((f"lmdb/{split}/data.mdb", args.data_root / "lmdb" / split / "data.mdb", expected))
    if args.teacher_checkpoint:
        checks.append(("teacher checkpoint", args.teacher_checkpoint, reference["teacher_checkpoint"]["sha256"]))

    failed = 0
    for label, path, expected in checks:
        if not path.exists():
            status = "MISSING"
        else:
            status = "ok" if sha256(path) == expected else "DIFFERENT"
        failed += status != "ok"
        print(f"{status:9} {label}")
    print(f"\n{len(checks) - failed}/{len(checks)} files match the reference build")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
