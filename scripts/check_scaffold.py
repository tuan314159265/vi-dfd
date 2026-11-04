"""Check the artifact's paths and standalone run-config/manifest coverage."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfd.config import load_config  # noqa: E402


REQUIRED = (
    "configs", "data/crop_vintext.py", "data/make_splits.py", "data/subsets.py",
    "dfd/vietchar.py", "dfd/heads.py", "dfd/distill.py", "dfd/metrics.py",
    "train.py", "eval.py", "bench", "serve/Dockerfile", "results", "README.md",
    "requirements.txt", "configs/run_manifest.csv",
)


def main() -> int:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing artifact paths: {missing}")
    with (ROOT / "configs/run_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = [row["config"] for row in rows]
    expected = {str(path.relative_to(ROOT)) for path in (ROOT / "configs/runs").rglob("*.yaml")}
    if not rows or len(set(paths)) != len(paths) or set(paths) != expected:
        raise SystemExit("Run manifest is empty, duplicated, or differs from YAML files")
    for row in rows:
        config = load_config(ROOT / row["config"])
        for name in ("experiment", "variant", "seed", "stage"):
            if str(config[name]) != row[name]:
                raise SystemExit(f"Manifest mismatch: {row['config']}: {name}")
        if config["seed"] not in {42, 43, 44}:
            raise SystemExit(f"Unexpected seed: {row['config']}")
    if {row["experiment"] for row in rows} != {f"E{i}" for i in range(1, 8)}:
        raise SystemExit("Missing experiment coverage in E1–E7")
    print(f"All {len(REQUIRED)} required artifact/support paths exist; {len(rows)} run configs match the manifest.")
    print("Scaffold coverage only. Research backends and experiments remain pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
