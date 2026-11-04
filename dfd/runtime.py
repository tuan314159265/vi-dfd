"""Prepare reproducible run artifacts; this module never fabricates training metrics."""

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

from dfd.config import validate_config


RUN_FIELDS = [
    "run_id", "experiment", "variant", "seed", "commit_hash", "config_hash",
    "status", "created_utc", "platform", "python_version", "torch_version",
    "gpu", "unresolved_fields",
]
METRIC_FIELDS = [
    "run_id", "commit_hash", "config_hash", "seed", "split", "subset", "precision",
    "num_words", "word_acc", "ned_score", "base_er", "modifier_er", "tone_er",
    "insertions", "deletions", "params", "cpu_latency_ms", "gpu_throughput",
    "peak_memory_mb",
]


def seed_everything(seed: int) -> None:
    """Seed installed libraries. Cross-device bitwise determinism is not guaranteed."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_revision(repo: Path) -> str:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE).strip()
    actual = Path(git("rev-parse", "--show-toplevel")).resolve()
    if actual != repo.resolve():
        raise ValueError("Expected a Git repository rooted at this project")
    revision = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("Commit code/config changes before preparing a run so commit_hash identifies the exact source")
    return revision


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def prepare_run(config: dict, repo: Path, output_root: Path | None = None) -> Path:
    unresolved = validate_config(config)
    revision = git_revision(repo)
    now = datetime.now(timezone.utc)
    identity = config_hash(config)
    run_id = f"{now:%Y%m%dT%H%M%SZ}_{config['variant']}_s{config['seed']}_{uuid.uuid4().hex[:8]}"
    root = output_root or repo / config["output"]["root"]
    folder = root / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    gpu = "unavailable"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    row = {
        "run_id": run_id, "experiment": config["experiment"], "variant": config["variant"],
        "seed": config["seed"], "commit_hash": revision, "config_hash": identity,
        "status": "prepared", "created_utc": now.isoformat(),
        "platform": os.environ.get("DFD_PLATFORM", platform.system()),
        "python_version": platform.python_version(), "torch_version": _version("torch"),
        "gpu": gpu, "unresolved_fields": ";".join(unresolved),
    }
    (folder / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, fields, rows in (("run.csv", RUN_FIELDS, [row]), ("metrics.csv", METRIC_FIELDS, [])):
        with (folder / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    environment = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions() if dist.metadata.get("Name"))
    (folder / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    return folder
