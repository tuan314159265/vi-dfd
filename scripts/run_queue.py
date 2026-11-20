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

Several Colab tabs (from different Google accounts) can run the SAME queue at the same time without training a
config twice. Before step 2 a worker claims the config by writing <drive-runs>/_claims/<config>-<hash>/<worker>.json:
  - a config with a live claim of another worker (heartbeat younger than --stale-minutes) is skipped;
  - after writing its claim the worker waits --claim-settle-seconds (Drive needs time to sync files between
    accounts), re-reads the folder, and keeps the config only if its claim is the oldest live one;
  - while training, a thread rewrites the claim every --heartbeat-seconds. A tab that dies stops beating, so after
    --stale-minutes another worker takes the config over (and reruns it from scratch);
  - a failed run leaves a claim with status "failed" that blocks every worker, the one that failed included (so a
    broken config is not retried by every tab, and a run whose eval failed is rescored instead of trained twice)
    until it is deleted or a worker passes --retry-failed. A finished or interrupted run removes its claim.
Each worker only writes and deletes its own file, because one Drive folder can hold two files with the same name.
Drive sync between accounts is not atomic, so this is a best-effort lock: a duplicate is still possible if two tabs
claim one config within the settle window; `e2_tuning.py lock` and summarize_results.py refuse two runs of one cell
(--exclude all but one), so it cannot slip into the results. --shuffle (each worker its own order) makes it rarer; --shard i/n splits the queue strictly instead.

--time-budget-minutes stops starting new configs when the time left in the session is shorter than one run
(--run-minutes, or the longest run seen in this session if longer), so a ~3 h Colab T4 session does not die
half-way through a run.

eval.py at exp-v1 imports strhub without putting the pinned PARSeq checkout on sys.path (train.py does that via
dfd/teacher.py), so the runner adds $PARSEQ_ROOT (default ~/.cache/vi-dfd/parseq, checked against the commit pinned
in dfd/parseq.py) to PYTHONPATH for train.py and eval.py. No file of the tagged checkout changes.

--rescore <folder> trains nothing: it scores run folders that finished training but were never uploaded (copied out
of a Colab VM after eval.py failed), with the same checks, queue.json and upload as a normal run. Each folder is
matched to the queue config it was trained from (every non-null value of the config must appear unchanged in the
run's resolved.yaml, and the run's commit must be the checkout's), so a rescored run counts as done for that config.
Run it in one tab only: it takes no claims.

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
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Configs that must never be trained: no_validity_mask scores the ti_dfd s42 checkpoint (train.py refuses it), and the
# temperature sweep reuses the DFD tuning trials; training them would only create duplicate runs for summarize_results.
EVAL_ONLY = {"no_validity_mask", "temperature_sweep"}
TRAIN_STAGES = {"train", "tune"}  # stage "evaluate" (E4-E7 scoring configs) only re-scores checkpoints
RUN_DIR_PREFIX = "Run directory:"
CLAIMS_DIR = "_claims"


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
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                if line.startswith(RUN_DIR_PREFIX):
                    run_dir = Path(line[len(RUN_DIR_PREFIX):].strip())
            code = process.wait()
        finally:  # interrupted (notebook stop button, SIGTERM): do not leave train.py running on the GPU
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = cwd / run_dir
    return code, run_dir


def parseq_env(env: dict, repo: Path) -> tuple[dict, Path | None]:
    """Put the pinned PARSeq checkout on PYTHONPATH for the child scripts (eval.py does not add it itself)."""
    root = Path(env.get("PARSEQ_ROOT") or Path.home() / ".cache/vi-dfd/parseq").expanduser()
    if not (root / "strhub").is_dir():
        return env, None
    source = repo / "dfd" / "parseq.py"
    pinned = re.search(r'PARSEQ_COMMIT\s*=\s*"([0-9a-f]{40})"', source.read_text(encoding="utf-8")) \
        if source.is_file() else None
    if pinned and (root / ".git").exists():
        revision = git(root, "rev-parse", "HEAD")
        if revision != pinned.group(1):
            raise SystemExit(f"PARSeq at {root} is at {revision[:12]}, the checkout pins {pinned.group(1)[:12]}")
        if git(root, "status", "--porcelain", "--untracked-files=no"):
            raise SystemExit(f"PARSeq at {root} has local changes")
    paths = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    if str(root) not in paths:
        paths.insert(0, str(root))
    return {**env, "PYTHONPATH": os.pathsep.join(paths)}, root


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def claim_folder(drive_runs: Path, key: dict) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return drive_runs / CLAIMS_DIR / f"{Path(key['config_path']).stem}-{digest}"


def read_claims(folder: Path, stale_seconds: float) -> list[dict]:
    """Every claim file in `folder`, with `live` = still beating (a failed claim never goes stale)."""
    claims = []
    now = time.time()
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            beat = datetime.fromisoformat(record["heartbeat_utc"]).timestamp()
        except (OSError, ValueError, KeyError, TypeError):  # half-synced file: judge it by its modification time
            try:
                beat = path.stat().st_mtime
            except OSError:
                continue
            record = {"worker": path.stem, "status": "running", "claimed_utc": ""}
        record["path"] = path
        record["live"] = record.get("status") == "failed" or now - beat < stale_seconds
        claims.append(record)
    return claims


class Claim:
    """This worker's claim file on one config, rewritten by a heartbeat thread while the run trains."""

    def __init__(self, folder: Path, worker: str, key: dict, heartbeat_seconds: float):
        self.path = folder / f"{worker}.json"
        self.record = {"worker": worker, "config_path": key["config_path"], "host": socket.gethostname(),
                       "status": "running", "claimed_utc": now_utc()}
        self.heartbeat_seconds = heartbeat_seconds
        self.stop_event = threading.Event()
        self.thread = None

    def write(self, **fields) -> None:
        self.record.update(fields, heartbeat_utc=now_utc())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.record, indent=2) + "\n", encoding="utf-8")

    def start(self) -> None:
        def beat():
            while not self.stop_event.wait(self.heartbeat_seconds):
                try:
                    self.write()
                except OSError as error:  # Drive hiccup: try again at the next beat
                    print(f"[claim] heartbeat failed: {error}", flush=True)
        self.thread = threading.Thread(target=beat, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def release(self) -> None:
        self.stop()
        try:
            self.path.unlink()
        except OSError:
            pass

    def fail(self, error: str) -> None:
        self.stop()
        try:
            self.write(status="failed", error=error[:500])
        except OSError:
            pass


def acquire(drive_runs: Path, key: dict, args) -> tuple[Claim | None, str]:
    """Claim the config for this worker, or return why not (another worker holds it, or it is already done)."""
    folder = claim_folder(drive_runs, key)
    stale = args.stale_minutes * 60
    claims = [c for c in read_claims(folder, stale) if c["live"]]
    running = [c for c in claims if c.get("status") != "failed" and c.get("worker") != args.worker]
    # this worker's own failed claim blocks too: otherwise a tab retrains every config it failed (19/11: eval.py
    # failed after training, the tab trained tau2_lam1_wt1 a second time while the first run sat in runs_rescue)
    failed = [c for c in claims if c.get("status") == "failed"]
    if running:
        return None, f"busy: claimed by {running[0].get('worker')}"
    if failed and not args.retry_failed:
        return None, f"blocked: failed on {failed[0].get('worker')} (delete {failed[0]['path']} or --retry-failed)"
    claim = Claim(folder, args.worker, key, args.heartbeat_seconds)
    claim.write()
    if args.claim_settle_seconds > 0:
        print(f"[claim] {key['config_path']}: waiting {args.claim_settle_seconds:g}s for competing claims", flush=True)
        time.sleep(args.claim_settle_seconds)
    contenders = [c for c in read_claims(folder, stale) if c["live"] and c.get("status") != "failed"]
    if not any(c.get("worker") == args.worker for c in contenders):  # our file is not visible yet: count it anyway
        contenders.append(dict(claim.record))
    first = min(contenders, key=lambda c: (c.get("claimed_utc") or "~", c.get("worker") or ""))
    if first.get("worker") != args.worker:
        claim.release()
        return None, f"busy: claimed by {first.get('worker')}"
    done = finished(drive_runs, key)
    if done:
        claim.release()
        return None, f"already done ({done})"
    claim.start()
    return claim, ""


def last_validated_epoch(run_dir: Path) -> int:
    """The last epoch train.py validated (validation.csv numbers epochs from 1); 0 without the file."""
    import csv
    path = run_dir / "validation.csv"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        return max((int(row["epoch"]) for row in csv.DictReader(handle) if row.get("epoch")), default=0)


def check_trained(run_dir: Path, smoke: bool) -> Path | None:
    """The best checkpoint of a run that finished training (None for a smoke test: it saves no checkpoint)."""
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    want = "smoke_completed" if smoke else "completed"
    if meta.get("status") != want:
        raise RuntimeError(f"run status is {meta.get('status')!r}, expected {want!r}")
    best = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    if smoke:
        return best[0] if len(best) == 1 else None
    last = last_validated_epoch(run_dir)
    epochs = (yaml.safe_load((run_dir / "resolved.yaml").read_text(encoding="utf-8")).get("training") or {}).get("epochs")
    if epochs and last < int(epochs):  # a run stopped early can still say "completed" (19/11: 5.7 of 20 epochs)
        raise RuntimeError(f"trained {last} of {epochs} epochs (validation.csv) although its status is 'completed'")
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
    claim, reason = acquire(args.drive_runs, key, args)
    if claim is None:
        return {**result, "status": "already done" if reason.startswith("already done") else reason}
    outcome = {}
    try:
        outcome = train_and_upload(config_path, config, relative, key, args, checkout, env, result)
        return outcome
    finally:  # done: drop the claim; failed: keep it as a marker; interrupted: drop it so another worker can rerun
        if outcome.get("status", "").startswith("failed"):
            claim.fail(outcome["status"])
        else:
            claim.release()


def train_and_upload(config_path: Path, config: dict, relative: str, key: dict, args, checkout: dict, env: dict,
                     result: dict) -> dict:
    repo = args.repo
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
                  "platform": env.get("DFD_PLATFORM"), "worker": args.worker, "finished_utc": now_utc()}
        (run_dir / "queue.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        target = upload(run_dir, args.drive_runs, args.upload_last)
        return {**result, "status": "done", "word_acc": scores, "uploaded": str(target)}
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        return {**result, "status": f"failed: {error}"}


def leaves(node, prefix=()):
    if isinstance(node, dict):
        for name, value in node.items():
            yield from leaves(value, (*prefix, name))
    else:
        yield prefix, node


def trained_from(config: dict, resolved: dict) -> bool:
    """train.py only fills null fields (data paths, teacher) and adds `diagnostic`: every set value must survive."""
    for path, value in leaves(config):
        if value is None:
            continue
        node = resolved
        for name in path:
            if not isinstance(node, dict) or name not in node:
                return False
            node = node[name]
        if node != value:
            return False
    return True


def rescore_one(source: Path, candidates: list, args, checkout: dict, env: dict) -> dict:
    """Score one rescued run folder and upload it as if the queue had finished it."""
    result = {"config": "", "status": "", "run_id": source.name, "word_acc": {}}
    try:
        meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        commit = meta.get("commit_hash", checkout["commit"])
        if commit != checkout["commit"]:
            return {**result, "status": f"skipped: trained at {commit[:12]}, checkout is {checkout['commit'][:12]}"}
        resolved = yaml.safe_load((source / "resolved.yaml").read_text(encoding="utf-8"))
        if resolved.get("diagnostic"):
            return {**result, "status": "skipped: smoke test"}
        found = [(path, config) for path, config in candidates if trained_from(config, resolved)]
        if len(found) != 1:
            return {**result, "status": f"skipped: matches {len(found)} queue configs {[p.name for p, _ in found]}"}
        config_path, config = found[0]
        relative = config_path.relative_to(args.repo).as_posix()
        result["config"] = relative
        key = {"config_path": relative, "config_sha256": sha256(config_path), "commit": checkout["commit"],
               "smoke_test": False}
        done = finished(args.drive_runs, key)
        if done:
            return {**result, "status": "already done", "run_id": done}
        check_trained(source, False)
        run_dir = args.log_dir.parent / "rescore" / source.name  # a local copy: eval.py writes into the run folder
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(source, run_dir)
        best = check_trained(run_dir, False)
        print(f"\n===== rescore {source.name} ({relative}) =====", flush=True)
        eval_log = args.log_dir / f"{source.name}.rescore.log"
        started = time.time()
        code, _ = stream([sys.executable, "eval.py", "--run-dir", str(run_dir), "--checkpoint", str(best),
                          "--device", args.device], args.repo, env, eval_log)
        eval_seconds = time.time() - started
        shutil.copy2(eval_log, run_dir / "eval_log.txt")
        if code != 0:
            raise RuntimeError(f"eval.py exited with code {code} (log: {eval_log})")
        scores = check_scored(run_dir, list((config.get("evaluation") or {}).get("splits") or []))
        record = {**key, "status": "done", "run_id": run_dir.name, "tags": checkout["tags"], "best_checkpoint": best.name,
                  "word_acc_all": scores, "train_seconds": meta.get("elapsed_seconds"),
                  "eval_seconds": round(eval_seconds, 1), "platform": meta.get("platform"), "worker": args.worker,
                  "finished_utc": now_utc(), "rescored_from": str(source)}
        (run_dir / "queue.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        target = upload(run_dir, args.drive_runs, args.upload_last)
        shutil.rmtree(run_dir)
        return {**result, "status": "done", "word_acc": scores, "uploaded": str(target)}
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError) as error:
        return {**result, "status": f"failed: {error}"}


def rescore(args, checkout: dict, env: dict, configs: list[Path]) -> list[dict]:
    candidates = []
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config.get("variant") not in EVAL_ONLY and config.get("stage", "train") in TRAIN_STAGES:
            candidates.append((path, config))
    folders = sorted(path for path in args.rescore.iterdir() if (path / "metadata.json").is_file())
    print(f"Rescoring {len(folders)} run folders from {args.rescore} against {len(candidates)} queue configs", flush=True)
    return [rescore_one(folder, candidates, args, checkout, env) for folder in folders]


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
    parser.add_argument("--worker", help="name of this tab in the claim files (default: <host>-<random>); "
                        "tabs running at the same time must use different names")
    parser.add_argument("--claim-settle-seconds", type=float, default=90.0,
                        help="wait after writing a claim before checking for competing claims (Drive sync delay)")
    parser.add_argument("--heartbeat-seconds", type=float, default=120.0, help="how often a running claim is refreshed")
    parser.add_argument("--stale-minutes", type=float, default=15.0,
                        help="a claim not refreshed for this long belongs to a dead tab and is taken over")
    parser.add_argument("--retry-failed", action="store_true", help="also take configs that failed on another worker")
    parser.add_argument("--shuffle", action="store_true",
                        help="walk the queue in an order seeded by --worker, so parallel tabs start on different configs")
    parser.add_argument("--rescore", type=Path, help="score the run folders in this folder that trained but were never "
                        "uploaded (eval.py failed); trains nothing, ignores --shard")
    parser.add_argument("--shard", help="i/n: only the configs at positions i, i+n, ... of the expanded queue")
    parser.add_argument("--time-budget-minutes", type=float,
                        help="do not start a config when less than one run of time is left (Colab T4 session ~180)")
    parser.add_argument("--run-minutes", type=float, default=45.0,
                        help="expected train+eval minutes of one run, for --time-budget-minutes")
    args = parser.parse_args(argv)
    args.worker = (args.worker or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}").strip()
    if not args.worker or any(char in args.worker for char in '/\\:*?"<>|'):
        parser.error(f"--worker {args.worker!r} must be usable as a file name")
    shard = None
    if args.shard:
        try:
            shard = tuple(int(part) for part in args.shard.split("/"))
            if len(shard) != 2 or not 0 <= shard[0] < shard[1]:
                raise ValueError
        except ValueError:
            parser.error(f"--shard {args.shard!r} is not i/n with 0 <= i < n")
    args.repo = args.repo.resolve()
    args.data_root = args.data_root.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve() if args.teacher_checkpoint else None
    args.log_dir = (args.log_dir or args.repo.parent / "queue_logs").resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.drive_runs.mkdir(parents=True, exist_ok=True)

    checkout = preflight(args.repo, args.require_tag)
    configs = expand(args.repo, args.configs)
    env, parseq_root = parseq_env({**os.environ, "DFD_PLATFORM": args.platform}, args.repo)
    print(f"PARSeq on PYTHONPATH: {parseq_root or 'not found (eval.py needs strhub importable)'}", flush=True)
    if args.rescore:  # the whole queue (no shard): a rescued run may come from any tab
        if not args.rescore.is_dir():
            parser.error(f"--rescore {args.rescore} is not a folder")
        return summarize(rescore(args, checkout, env, configs))
    if shard:
        configs = configs[shard[0]::shard[1]]
    if args.shuffle:
        random.Random(args.worker).shuffle(configs)
    print(f"Queue of {len(configs)} configs at {checkout['commit'][:12]} {checkout['tags']} -> {args.drive_runs} "
          f"(worker {args.worker})", flush=True)

    started = time.time()
    results, trained, longest = [], 0, 0.0
    for config_path in configs:
        relative = config_path.relative_to(args.repo).as_posix()
        if args.max_runs is not None and trained >= args.max_runs:
            results.append({"config": relative, "status": "not started (--max-runs)", "run_id": "", "word_acc": {}})
            continue
        if args.time_budget_minutes is not None:
            left = args.time_budget_minutes - (time.time() - started) / 60
            need = max(args.run_minutes, longest)
            if left < need:
                results.append({"config": relative, "status": f"not started (time: {left:.0f} min left < {need:.0f})",
                                "run_id": "", "word_acc": {}})
                continue
        preflight(args.repo, args.require_tag)  # a run must not leave the checkout dirty for the next one
        run_started = time.time()
        result = run_one(config_path, args, checkout, env)
        results.append(result)
        if result["status"] == "done" or result["status"].startswith("failed"):
            trained += 1
            longest = max(longest, (time.time() - run_started) / 60)
        if result["status"].startswith("failed") and args.stop_on_failure:
            break
    return summarize(results)


def summarize(results: list[dict]) -> int:
    print("\n===== queue summary =====")
    for result in results:
        scores = " ".join(f"{split}={value:.4f}" for split, value in result["word_acc"].items())
        status = result["status"] if result["status"].startswith(("blocked", "skipped: ")) else result["status"][:60]
        print(f"{status:<60} {result['config']} {result['run_id']} {scores}")
    failed = [result for result in results if result["status"].startswith("failed")]
    print(f"{sum(r['status'] == 'done' for r in results)} done now, "
          f"{sum(r['status'] == 'already done' for r in results)} already done, "
          f"{sum(r['status'].startswith(('busy', 'blocked')) for r in results)} held by other workers, "
          f"{sum(r['status'].startswith('not started') for r in results)} not started, {len(failed)} failed")
    return 1 if failed else 0


def interrupted(signum, frame):
    """SIGTERM (notebook stop, Colab shutdown) unwinds like Ctrl-C: train.py is stopped and the claim dropped."""
    raise KeyboardInterrupt(f"signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
