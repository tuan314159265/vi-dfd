"""Validate configs, prepare artifacts, fine-tune the PARSeq-S teacher, or train PARSeq-Ti students."""

import argparse
import subprocess
from pathlib import Path

from dfd.config import load_config, validate_config
from dfd.runtime import prepare_run, seed_everything

DATA_DEFAULTS = {"vocab_path": "vocabulary.json", "split_manifest": "splits/freeze_manifest.json",
                 "subset_manifest": "splits/subsets"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, help="Override the fixed seed (study defaults: 42/43/44)")
    parser.add_argument("--data-root", type=Path, help="Processed VinText root containing lmdb/, splits/, vocabulary.json")
    parser.add_argument("--teacher-checkpoint", type=Path, help="Frozen PARSeq-S .ckpt for KD/DFD students (sets model.teacher_checkpoint)")
    parser.add_argument("--resume", type=Path, help="Resume the exact original configuration from a local trusted checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="One train/validation batch only; recorded as a diagnostic")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-config", action="store_true", help="Validate template schema; report unresolved choices")
    action.add_argument("--prepare-run", action="store_true", help="Save metadata only; does not train a model")
    action.add_argument("--probe-batch", type=Path, help="Measure CUDA micro-batches 16/32/64 and save a JSON report")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        if args.data_root is not None:
            config["data"]["root"] = str(args.data_root.expanduser().resolve())
            # Standard layout of the processed root (same defaults as the teacher config).
            for field, default in DATA_DEFAULTS.items():
                if config["data"].get(field) is None:
                    config["data"][field] = default
        if args.teacher_checkpoint is not None:
            config["model"]["teacher_checkpoint"] = str(args.teacher_checkpoint.expanduser().resolve())
        if args.smoke_test:
            config["diagnostic"] = "fast_dev_run"
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
        if args.probe_batch or config["variant"] == "teacher":
            from dfd.teacher import probe_batches, run_teacher
            if args.probe_batch:
                probe_batches(config, args.probe_batch)
                return 0
            if unresolved:
                parser.error("Unresolved training fields: " + ", ".join(unresolved))
            run_teacher(config, Path(__file__).resolve().parent, args.resume, args.smoke_test)
            return 0
        if config["model"]["backbone"] == "parseq_tiny" and config.get("stage", "train") == "train":
            if unresolved:
                parser.error("Unresolved training fields: " + ", ".join(unresolved))
            from dfd.student import run_student
            run_student(config, Path(__file__).resolve().parent, args.resume, args.smoke_test)
            return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    parser.error("Only the PARSeq-S teacher and PARSeq-Ti students can be trained; see docs/IMPLEMENTATION.md")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
