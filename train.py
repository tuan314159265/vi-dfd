"""Validate an experiment or prepare its artifacts. PARSeq training is not wired yet."""

import argparse
import subprocess
from pathlib import Path

from dfd.config import load_config, validate_config
from dfd.runtime import prepare_run, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, help="Override the fixed seed (study defaults: 42/43/44)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-config", action="store_true", help="Validate template schema; report unresolved choices")
    action.add_argument("--prepare-run", action="store_true", help="Save metadata only; does not train a model")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        unresolved = validate_config(config)
        if args.check_config:
            print(f"Valid template: {args.config}")
            print("Unresolved: " + (", ".join(unresolved) if unresolved else "none"))
            return 0
        if args.prepare_run:
            seed_everything(config["seed"])
            folder = prepare_run(config, Path(__file__).resolve().parent)
            print(f"Prepared artifacts: {folder}")
            print("Status: prepared; no model has been trained and no metrics have been produced.")
            return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    parser.error("PARSeq training backend is not implemented. Use --check-config or --prepare-run; see docs/IMPLEMENTATION.md")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
