"""Verify a completed teacher run and export its best weights plus a CSV summary."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dfd.runtime import METRIC_FIELDS  # noqa: E402
from dfd.vietchar import DecompositionTable  # noqa: E402


def finish(folder):
    folder = Path(folder).resolve()
    training = json.loads((folder / "training.json").read_text())
    if training["status"] != "completed" or training["smoke_test"]:
        raise ValueError("Only a successfully completed full teacher run can be exported")
    metadata = json.loads((folder / "metadata.json").read_text())
    checkpoint_path = Path(training["best_checkpoint"])
    # This checkpoint was generated locally by our own training process.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    table = DecompositionTable.from_characters()
    if checkpoint["vocabulary_fingerprint"] != table.fingerprint():
        raise ValueError("Checkpoint vocabulary differs from the current fixed table")
    if checkpoint["dfd_run_identity"]["commit_hash"] != metadata["commit_hash"]:
        raise ValueError("Checkpoint and run metadata refer to different source commits")
    state = {name.removeprefix("model."): value for name, value in checkpoint["state_dict"].items() if name.startswith("model.")}
    if state["head.weight"].shape != (229, 384) or state["text_embed.embedding.weight"].shape != (231, 384):
        raise ValueError("Wrong teacher output-head or context-embedding shape")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Checkpoint contains a non-finite tensor")
    config = checkpoint["dfd_config"]
    with (folder / "validation.csv").open(encoding="utf-8", newline="") as handle:
        validations = list(csv.DictReader(handle))
    best_epoch = checkpoint["epoch"] + 1
    best = next(row for row in validations if int(row["epoch"]) == best_epoch)
    if abs(float(best["word_acc"]) - training["best_validation_word_acc"]) > 1e-6:
        raise ValueError("Best checkpoint does not match its validation CSV row")
    export_path = folder / "teacher_parseq_s_229.pt"
    payload = {
        "state_dict": state, "characters": table.characters,
        "vocabulary_fingerprint": table.fingerprint(), "parseq_commit": checkpoint["parseq_commit"],
        "commit_hash": metadata["commit_hash"], "config_hash": metadata["config_hash"],
        "seed": config["seed"], "best_epoch": best_epoch,
    }
    if not export_path.exists():
        torch.save(payload, export_path)
    loaded = torch.load(export_path, map_location="cpu", weights_only=True)
    if loaded["commit_hash"] != metadata["commit_hash"] or set(loaded["state_dict"]) != set(state):
        raise ValueError("Existing weight export differs from the selected teacher")
    if not all(torch.equal(loaded["state_dict"][name], value) for name, value in state.items()):
        raise ValueError("Exported weights do not exactly match the best checkpoint")
    # PARSeq's ViT/LayerNorm model state contains parameters, without BatchNorm statistics.
    parameters = sum(value.numel() for value in state.values())
    summary = {
        "run_id": metadata["run_id"], "commit_hash": metadata["commit_hash"],
        "config_hash": metadata["config_hash"], "model": "PARSeq-S", "seed": config["seed"],
        "epochs_completed": config["training"]["epochs"], "best_epoch": best_epoch,
        "batch_size": config["training"]["batch_size"], "train_samples": training["train_audit"]["kept"],
        "validation_samples": int(best["num_words"]), "validation_word_acc": float(best["word_acc"]),
        "heavy_samples": int(best["heavy_words"]), "validation_heavy_acc": float(best["heavy_word_acc"]),
        "validation_ned": float(best["ned_score"]), "parameters": parameters,
        "elapsed_seconds": training["elapsed_seconds"], "best_checkpoint": str(checkpoint_path),
        "weights_path": str(export_path), "weights_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (folder / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    metric_base = {"run_id": summary["run_id"], "commit_hash": summary["commit_hash"],
                   "config_hash": summary["config_hash"], "seed": summary["seed"],
                   "split": "validation", "precision": "fp32_weights_amp_inference", "params": parameters}
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(dict(metric_base, subset="all", num_words=summary["validation_samples"],
                             word_acc=summary["validation_word_acc"], ned_score=summary["validation_ned"]))
        writer.writerow(dict(metric_base, subset="diacritic_heavy", num_words=summary["heavy_samples"],
                             word_acc=summary["validation_heavy_acc"]))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    finish(args.run)
