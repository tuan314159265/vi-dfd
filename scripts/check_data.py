"""Check that this machine has the group's reference data build (and teacher) before running.

    python scripts/check_data.py --data-root <processed>/vintext_notebook [--teacher-checkpoint <.ckpt>]

Compares SHA-256 hashes with configs/data_reference.json. Exit code 1 on any mismatch.
"""

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="Folder containing lmdb/ and splits/")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "configs/data_reference.json")
    args = parser.parse_args()
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
