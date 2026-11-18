"""Train, score and upload a list of runs one after another (Colab / Kaggle / a local GPU).

    python run_queue.py --repo /content/vi-dfd --data-root $DATA --teacher-checkpoint "$TEACHER" \
        --drive-runs "/content/drive/MyDrive/BTL DL/runs" --require-tag exp-v1 --platform colab \
        configs/tuning_runs/E2/ti_dfd_*.yaml

For every config, in order:
  1. skip it if <drive-runs> already holds a finished copy (queue.json with the same config file, the same
     config bytes and the same commit), so a new Colab session simply continues the queue;
  2. `train.py --config ... --data-root ... [--teacher-checkpoint ...]` (the teacher only when KD is enabled);
  3. check that the run completed and left exactly one best-*.ckpt;
  4. `eval.py --run-dir <run> --checkpoint <best> --device <device>` (splits from the config) and check that
     metrics.csv has an `all` row for every split;
  5. write queue.json, copy the run folder to <drive-runs>/<run_id> (via <run_id>.partial, renamed at the end,
     so a half-copied folder never looks finished). last.ckpt is left out unless --upload-last.

Interrupted runs are NOT resumed: they are rerun from scratch. A resumed run gets a new run folder, and
Lightning's ModelCheckpoint then forgets the best score seen before the interruption (it reloads it only when
the checkpoint folder is unchanged), so best-*.ckpt would be the best of the epochs after the resume only.

Copy this file out of the repository before checking out the tag (train.py refuses a dirty working tree),
or run it from any other folder: it only needs the standard library and PyYAML.
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Configs that must never be trained: no_validity_mask scores the ti_dfd s42 checkpoint (train.py refuses it), and the
# temperature sweep reuses the DFD tuning trials; training them would only create duplicate runs for summarize_results.
EVAL_ONLY = {"no_validity_mask", "temperature_sweep"}
TRAIN_STAGES = {"train", "tune"}  # stage "evaluate" (E4-E7 scoring configs) only re-scores checkpoints
RUN_DIR_PREFIX = "Run directory:"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(repo: Path, require_tag: str | None) -> dict:
    """The checkout every run of this queue will record: clean, and on the required tag."""
    status = git(repo, "status", "--porcelain")
    if status:
        raise SystemExit(f"{repo} has uncommitted or untracked files; train.py would refuse to run:\n{status}")
    commit = git(repo, "rev-parse", "HEAD")
    tags = set(git(repo, "tag", "--points-at", "HEAD").split())
    if require_tag and require_tag not in tags:
        raise SystemExit(f"HEAD {commit[:12]} carries tags {sorted(tags) or 'none'}, not {require_tag}: "
                         f"git -C {repo} checkout {require_tag}")
    return {"commit": commit, "tags": sorted(tags)}


def expand(repo: Path, patterns: list[str]) -> list[Path]:
    configs = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(str(repo / pattern)))
        if not matches:
            raise SystemExit(f"No config matches {pattern!r} under {repo}")
        configs += [path for path in matches if path not in configs]
    return configs


def finished(drive_runs: Path, key: dict) -> str | None:
    """Run id of a finished copy of this exact config and commit on the drive, if any."""
    if not drive_runs.is_dir():
        return None
    for marker in sorted(drive_runs.glob("*/queue.json")):
        if marker.parent.name.endswith(".partial"):
            continue
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") == "done" and all(record.get(field) == value for field, value in key.items()):
            return marker.parent.name
    return None


def stream(command: list[str], cwd: Path, env: dict, log: Path) -> tuple[int, Path | None]:
    """Run `command`, echo and log its output, and return (exit code, run folder announced by train.py)."""
    run_dir = None
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            if line.startswith(RUN_DIR_PREFIX):
                run_dir = Path(line[len(RUN_DIR_PREFIX):].strip())
        code = process.wait()
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = cwd / run_dir
    return code, run_dir


def check_trained(run_dir: Path, smoke: bool) -> Path | None:
    """The best checkpoint of a run that finished training (None for a smoke test: it saves no checkpoint)."""
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    want = "smoke_completed" if smoke else "completed"
    if meta.get("status") != want:
        raise RuntimeError(f"run status is {meta.get('status')!r}, expected {want!r}")
    best = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    if smoke:
        return best[0] if len(best) == 1 else None
    if len(best) != 1:
        raise RuntimeError(f"expected exactly one best-*.ckpt in {run_dir / 'checkpoints'}, found {[p.name for p in best]}")
    return best[0]


def check_scored(run_dir: Path, splits: list[str]) -> dict:
    """validation / test / ... -> word accuracy on subset all, from metrics.csv. Fails if a split is missing."""
    import csv
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["subset"] == "all"]
    found = {row["split"]: float(row["word_acc"]) for row in rows}
    missing = [split for split in splits if split not in found]
    if missing:
        raise RuntimeError(f"metrics.csv has no 'all' row for {missing}")
    return {split: found[split] for split in splits}


def upload(run_dir: Path, drive_runs: Path, keep_last: bool) -> Path:
    target = drive_runs / run_dir.name
    partial = drive_runs / (run_dir.name + ".partial")
    if target.exists():
        raise RuntimeError(f"{target} already exists")
    if partial.exists():
        shutil.rmtree(partial)
    ignore = None if keep_last else shutil.ignore_patterns("last.ckpt", "last-v*.ckpt")
    shutil.copytree(run_dir, partial, ignore=ignore)
    partial.rename(target)
    return target


def run_one(config_path: Path, args, checkout: dict, env: dict) -> dict:
    repo = args.repo
    relative = config_path.resolve().relative_to(repo.resolve()).as_posix()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    key = {"config_path": relative, "config_sha256": sha256(config_path), "commit": checkout["commit"],
           "smoke_test": bool(args.smoke_test)}
    result = {"config": relative, "status": "", "run_id": "", "word_acc": {}}
    if config.get("variant") in EVAL_ONLY or config.get("stage", "train") not in TRAIN_STAGES:
        return {**result, "status": "skipped: not a training config (score an existing checkpoint with eval.py)"}
    done = finished(args.drive_runs, key)
    if done:
        return {**result, "status": "already done", "run_id": done}

    command = [sys.executable, "train.py", "--config", relative, "--data-root", str(args.data_root)]
    if (config.get("distillation") or {}).get("enabled"):
        if args.teacher_checkpoint is None:
            return {**result, "status": "failed: KD config needs --teacher-checkpoint"}
        command += ["--teacher-checkpoint", str(args.teacher_checkpoint)]
    if args.smoke_test:
        command.append("--smoke-test")
    log = args.log_dir / (config_path.stem + ".train.log")
    print(f"\n===== {relative} =====", flush=True)
    started = time.time()
    code, run_dir = stream(command, repo, env, log)
    train_seconds = time.time() - started
    if run_dir is not None:
        result["run_id"] = run_dir.name
    try:
        if code != 0:
            raise RuntimeError(f"train.py exited with code {code} (log: {log})")
        if run_dir is None or not run_dir.is_dir():
            raise RuntimeError(f"train.py did not announce a run folder (log: {log})")
        shutil.copy2(log, run_dir / "train_log.txt")
        best = check_trained(run_dir, args.smoke_test)
        splits = list((config.get("evaluation") or {}).get("splits") or [])
        scores, eval_seconds = {}, 0.0
        if best is not None:
            eval_log = args.log_dir / (config_path.stem + ".eval.log")
            started = time.time()
            code, _ = stream([sys.executable, "eval.py", "--run-dir", str(run_dir), "--checkpoint", str(best),
                              "--device", args.device], repo, env, eval_log)
            eval_seconds = time.time() - started
            shutil.copy2(eval_log, run_dir / "eval_log.txt")
            if code != 0:
                raise RuntimeError(f"eval.py exited with code {code} (log: {eval_log})")
            scores = check_scored(run_dir, splits)
        record = {**key, "status": "done", "run_id": run_dir.name, "tags": checkout["tags"],
                  "best_checkpoint": best.name if best else None, "word_acc_all": scores,
                  "train_seconds": round(train_seconds, 1), "eval_seconds": round(eval_seconds, 1),
                  "platform": env.get("DFD_PLATFORM"), "finished_utc": datetime.now(timezone.utc).isoformat()}
        (run_dir / "queue.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        target = upload(run_dir, args.drive_runs, args.upload_last)
        return {**result, "status": "done", "word_acc": scores, "uploaded": str(target)}
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        return {**result, "status": f"failed: {error}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="config paths or globs, relative to --repo")
    parser.add_argument("--repo", type=Path, required=True, help="clean checkout of the shared tag")
    parser.add_argument("--data-root", type=Path, required=True, help="processed VinText root with lmdb/ and splits/")
    parser.add_argument("--teacher-checkpoint", type=Path, help="frozen PARSeq-S .ckpt (needed by KD configs)")
    parser.add_argument("--drive-runs", type=Path, required=True, help="shared folder that receives finished run folders")
    parser.add_argument("--require-tag", help="refuse to start unless HEAD carries this tag (exp-v1, exp-v2)")
    parser.add_argument("--platform", default=os.environ.get("DFD_PLATFORM", "local"), help="written as DFD_PLATFORM")
    parser.add_argument("--device", default="cuda", help="eval.py device")
    parser.add_argument("--log-dir", type=Path, help="default: <repo>/../queue_logs")
    parser.add_argument("--max-runs", type=int, help="stop after training this many runs (session budget)")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--upload-last", action="store_true", help="also upload last.ckpt (only needed to resume)")
    parser.add_argument("--smoke-test", action="store_true", help="pass --smoke-test to train.py: wiring check only, "
                        "never counted by select/summarize; use a separate --drive-runs folder")
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    args.data_root = args.data_root.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve() if args.teacher_checkpoint else None
    args.log_dir = (args.log_dir or args.repo.parent / "queue_logs").resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.drive_runs.mkdir(parents=True, exist_ok=True)

    checkout = preflight(args.repo, args.require_tag)
    configs = expand(args.repo, args.configs)
    env = {**os.environ, "DFD_PLATFORM": args.platform}
    print(f"Queue of {len(configs)} configs at {checkout['commit'][:12]} {checkout['tags']} -> {args.drive_runs}", flush=True)

    results, trained = [], 0
    for config_path in configs:
        if args.max_runs is not None and trained >= args.max_runs:
            results.append({"config": config_path.relative_to(args.repo).as_posix(), "status": "not started (--max-runs)",
                            "run_id": "", "word_acc": {}})
            continue
        preflight(args.repo, args.require_tag)  # a run must not leave the checkout dirty for the next one
        result = run_one(config_path, args, checkout, env)
        results.append(result)
        if result["status"] == "done" or result["status"].startswith("failed"):
            trained += 1
        if result["status"].startswith("failed") and args.stop_on_failure:
            break

    print("\n===== queue summary =====")
    for result in results:
        scores = " ".join(f"{split}={value:.4f}" for split, value in result["word_acc"].items())
        print(f"{result['status'][:60]:<60} {result['config']} {result['run_id']} {scores}")
    failed = [result for result in results if result["status"].startswith("failed")]
    print(f"{sum(r['status'] == 'done' for r in results)} done now, "
          f"{sum(r['status'] == 'already done' for r in results)} already done, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
