import importlib.util
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_queue", ROOT / "scripts" / "run_queue.py")
run_queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_queue)

# Stand-ins for train.py / eval.py with the same outputs the real scripts leave in a run folder.
FAKE_TRAIN = textwrap.dedent('''
    import argparse, json, os, uuid, yaml
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--data-root"); p.add_argument("--teacher-checkpoint")
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()
    config = yaml.safe_load(open(a.config, encoding="utf-8"))
    mode = config.get("fake", "ok")
    run = Path("results/runs") / f"run_{config['variant']}_{uuid.uuid4().hex[:6]}"
    (run / "checkpoints").mkdir(parents=True)
    if a.smoke_test:
        config["diagnostic"] = "fast_dev_run"
    (run / "resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "metrics.csv").write_text("run_id,split,subset,word_acc\\n", encoding="utf-8")
    print("Run directory:", run.resolve(), flush=True)
    Path("calls.log").open("a").write(json.dumps({"teacher": a.teacher_checkpoint, "platform": os.environ.get("DFD_PLATFORM"), "pythonpath": os.environ.get("PYTHONPATH", "")}) + "\\n")
    if mode == "crash":
        raise SystemExit(3)
    if not a.smoke_test:
        (run / "checkpoints" / "best-epoch=03-validation_word_acc=0.7000.ckpt").write_bytes(b"best")
        (run / "checkpoints" / "last.ckpt").write_bytes(b"last")
        if mode == "two_best":
            (run / "checkpoints" / "best-epoch=04-validation_word_acc=0.7100.ckpt").write_bytes(b"best")
        epochs = 1 if mode == "short" else config["training"]["epochs"]  # "short": stopped early, still "completed"
        (run / "validation.csv").write_text("epoch,word_acc\\n" + "".join(f"{e},0.7\\n" for e in range(1, epochs + 1)),
                                            encoding="utf-8")
    status = "smoke_completed" if a.smoke_test else ("interrupted" if mode == "interrupted" else "completed")
    (run / "metadata.json").write_text(json.dumps({"run_id": run.name, "status": status}), encoding="utf-8")
''')
FAKE_EVAL = textwrap.dedent('''
    import argparse, os, yaml
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir"); p.add_argument("--checkpoint"); p.add_argument("--device")
    a = p.parse_args()
    if os.environ.get("FAKE_EVAL_FAIL"):
        raise SystemExit("ImportError: cannot import name 'DistilledPARSeq'")
    run = Path(a.run_dir)
    config = yaml.safe_load((run / "resolved.yaml").read_text(encoding="utf-8"))
    splits = [] if config.get("fake") == "no_rows" else config["evaluation"]["splits"]
    with (run / "metrics.csv").open("a", encoding="utf-8") as handle:
        for split in splits:
            handle.write(f"{run.name},{split},all,0.7123\\n")
''')


def make_config(repo, name, variant="ti_dfd", kd=True, fake="ok", splits=("validation",), stage="tune"):
    config = {"experiment": "E2", "stage": stage, "variant": variant, "seed": 42, "fake": fake,
              "distillation": {"enabled": kd}, "evaluation": {"splits": list(splits)}, "training": {"epochs": 3}}
    path = repo / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text(FAKE_TRAIN, encoding="utf-8")
    (repo / "eval.py").write_text(FAKE_EVAL, encoding="utf-8")
    (repo / ".gitignore").write_text("results/\ncalls.log\n__pycache__/\n", encoding="utf-8")
    make_config(repo, "a_dfd")
    make_config(repo, "b_ce", variant="ti_ce", kd=False)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call([*git, "add", "-A"])
    subprocess.check_call([*git, "commit", "-qm", "init"])
    subprocess.check_call([*git, "tag", "exp-v1"])
    return repo


def commit_all(repo):
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.check_call([*git, "add", "-A"])
    subprocess.check_call([*git, "commit", "-qm", "more"])
    subprocess.check_call([*git, "tag", "-f", "exp-v1"], stdout=subprocess.DEVNULL)


def queue(repo, tmp_path, *configs, extra=()):
    argv = [*configs, "--repo", str(repo), "--data-root", str(tmp_path), "--teacher-checkpoint", str(tmp_path / "t.ckpt"),
            "--drive-runs", str(tmp_path / "drive"), "--require-tag", "exp-v1", "--platform", "colab", "--device", "cpu",
            "--log-dir", str(tmp_path / "logs"), "--claim-settle-seconds", "0", *extra]
    return run_queue.main(argv)


def drive_runs(tmp_path):
    return sorted(path for path in (tmp_path / "drive").iterdir() if path.name != run_queue.CLAIMS_DIR)


def test_runs_scores_uploads_and_skips_finished_configs(repo, tmp_path):
    assert queue(repo, tmp_path, "configs/*.yaml") == 0
    uploaded = drive_runs(tmp_path)
    assert len(uploaded) == 2 and not any(path.name.endswith(".partial") for path in uploaded)
    for folder in uploaded:
        record = json.loads((folder / "queue.json").read_text(encoding="utf-8"))
        assert record["status"] == "done" and record["word_acc_all"] == {"validation": 0.7123}
        assert record["commit"] == subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        assert (folder / "checkpoints").is_dir() and not (folder / "checkpoints" / "last.ckpt").exists()
        assert len(list((folder / "checkpoints").glob("best-*.ckpt"))) == 1
        assert (folder / "train_log.txt").exists() and (folder / "eval_log.txt").exists()
    calls = [json.loads(line) for line in (repo / "calls.log").read_text().splitlines()]
    assert [call["teacher"] is not None for call in calls] == [True, False]  # teacher only for the KD config
    assert {call["platform"] for call in calls} == {"colab"}
    # A new session continues the queue instead of training the same configs again.
    assert queue(repo, tmp_path, "configs/*.yaml") == 0
    assert len(drive_runs(tmp_path)) == 2 and len((repo / "calls.log").read_text().splitlines()) == 2


def test_changed_config_bytes_or_commit_means_not_done(repo, tmp_path):
    assert queue(repo, tmp_path, "configs/a_dfd.yaml") == 0
    make_config(repo, "a_dfd", splits=("validation", "test"))
    commit_all(repo)
    assert queue(repo, tmp_path, "configs/a_dfd.yaml") == 0
    assert len(drive_runs(tmp_path)) == 2


@pytest.mark.parametrize("fake, reason", [("crash", "exited with code 3"), ("interrupted", "status is 'interrupted'"),
                                          ("two_best", "exactly one best"), ("no_rows", "no 'all' row"),
                                          ("short", "trained 1 of 3 epochs")])
def test_bad_runs_fail_and_are_not_uploaded(repo, tmp_path, fake, reason, capsys):
    make_config(repo, "c_bad", fake=fake)
    commit_all(repo)
    assert queue(repo, tmp_path, "configs/c_bad.yaml", "configs/b_ce.yaml") == 1
    out = capsys.readouterr().out
    assert reason in out
    assert [json.loads((p / "queue.json").read_text())["config_path"] for p in drive_runs(tmp_path)] == ["configs/b_ce.yaml"]


def test_stop_on_failure_and_max_runs(repo, tmp_path):
    make_config(repo, "0_crash", fake="crash")
    commit_all(repo)
    assert queue(repo, tmp_path, "configs/*.yaml", extra=("--stop-on-failure",)) == 1
    assert len((repo / "calls.log").read_text().splitlines()) == 1
    assert queue(repo, tmp_path, "configs/a_dfd.yaml", "configs/b_ce.yaml", extra=("--max-runs", "1")) == 0
    assert len(drive_runs(tmp_path)) == 1


def test_preflight_refuses_dirty_tree_and_wrong_tag(repo, tmp_path):
    (repo / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="uncommitted or untracked"):
        queue(repo, tmp_path, "configs/*.yaml")
    (repo / "notes.txt").unlink()
    with pytest.raises(SystemExit, match="not exp-v2"):
        run_queue.preflight(repo, "exp-v2")
    with pytest.raises(SystemExit, match="No config matches"):
        queue(repo, tmp_path, "configs/missing_*.yaml")


def test_eval_only_and_smoke_runs(repo, tmp_path):
    make_config(repo, "d_mask", variant="no_validity_mask")
    make_config(repo, "e_sweep", variant="temperature_sweep")
    make_config(repo, "f_score", variant="ti_ce", kd=False, stage="evaluate")
    commit_all(repo)
    assert queue(repo, tmp_path, "configs/d_mask.yaml", "configs/e_sweep.yaml", "configs/f_score.yaml", "configs/a_dfd.yaml",
                 extra=("--smoke-test",)) == 0
    assert len((repo / "calls.log").read_text().splitlines()) == 1  # only a_dfd reached train.py
    (folder,) = drive_runs(tmp_path)  # the eval-only configs are skipped, the smoke run has no checkpoint to score
    record = json.loads((folder / "queue.json").read_text(encoding="utf-8"))
    assert record["smoke_test"] is True and record["best_checkpoint"] is None and record["word_acc_all"] == {}
    assert yaml.safe_load((folder / "resolved.yaml").read_text(encoding="utf-8"))["diagnostic"] == "fast_dev_run"
    # a smoke copy never counts as a finished real run
    assert run_queue.finished(tmp_path / "drive", {**{k: record[k] for k in ("config_path", "config_sha256", "commit")},
                                                   "smoke_test": False}) is None


drive_runs_done = drive_runs


def claim_key(repo, name, smoke=False):
    path = repo / "configs" / f"{name}.yaml"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return {"config_path": f"configs/{name}.yaml", "config_sha256": run_queue.sha256(path), "commit": commit,
            "smoke_test": smoke}


def write_claim(tmp_path, repo, name, worker, status="running", age_seconds=0.0, claimed="2000-01-01T00:00:00+00:00"):
    from datetime import datetime, timedelta, timezone
    folder = run_queue.claim_folder(tmp_path / "drive", claim_key(repo, name))
    folder.mkdir(parents=True, exist_ok=True)
    beat = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    path = folder / f"{worker}.json"
    path.write_text(json.dumps({"worker": worker, "status": status, "claimed_utc": claimed, "heartbeat_utc": beat}),
                    encoding="utf-8")
    return path


def test_finished_run_leaves_no_claim_and_records_worker(repo, tmp_path):
    assert queue(repo, tmp_path, "configs/a_dfd.yaml", extra=("--worker", "vinh-1")) == 0
    (folder,) = drive_runs_done(tmp_path)
    assert json.loads((folder / "queue.json").read_text(encoding="utf-8"))["worker"] == "vinh-1"
    assert not list((tmp_path / "drive" / run_queue.CLAIMS_DIR).rglob("*.json"))


def test_live_claim_of_another_worker_is_skipped_and_stale_one_taken_over(repo, tmp_path, capsys):
    write_claim(tmp_path, repo, "a_dfd", "tuan-1")
    assert queue(repo, tmp_path, "configs/*.yaml", extra=("--worker", "vinh-1")) == 0
    assert "busy: claimed by tuan-1" in capsys.readouterr().out
    assert [json.loads((p / "queue.json").read_text())["config_path"] for p in drive_runs_done(tmp_path)] == ["configs/b_ce.yaml"]
    # the other tab died: its heartbeat is older than --stale-minutes, so this worker reruns the config
    write_claim(tmp_path, repo, "a_dfd", "tuan-1", age_seconds=3600)
    assert queue(repo, tmp_path, "configs/*.yaml", extra=("--worker", "vinh-1")) == 0
    assert len(drive_runs_done(tmp_path)) == 2


def test_older_competing_claim_wins_after_settling(repo, tmp_path, monkeypatch, capsys):
    # Another tab writes its (older) claim while this worker waits for Drive to sync.
    monkeypatch.setattr(run_queue.time, "sleep", lambda seconds: write_claim(tmp_path, repo, "a_dfd", "tien-1"))
    assert queue(repo, tmp_path, "configs/a_dfd.yaml", extra=("--worker", "vinh-1", "--claim-settle-seconds", "5")) == 0
    assert "busy: claimed by tien-1" in capsys.readouterr().out
    assert not drive_runs_done(tmp_path) and not (repo / "calls.log").exists()
    folder = run_queue.claim_folder(tmp_path / "drive", claim_key(repo, "a_dfd"))
    assert [path.name for path in folder.iterdir()] == ["tien-1.json"]  # our losing claim was withdrawn


def test_failed_claim_blocks_other_workers_until_retry(repo, tmp_path, capsys):
    make_config(repo, "c_bad", fake="crash")
    commit_all(repo)
    assert queue(repo, tmp_path, "configs/c_bad.yaml", extra=("--worker", "vinh-1")) == 1
    folder = run_queue.claim_folder(tmp_path / "drive", claim_key(repo, "c_bad"))
    (marker,) = folder.iterdir()
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "failed"
    capsys.readouterr()
    assert queue(repo, tmp_path, "configs/c_bad.yaml", extra=("--worker", "tuan-1")) == 0
    assert "blocked: failed on vinh-1" in capsys.readouterr().out
    assert len((repo / "calls.log").read_text().splitlines()) == 1
    assert queue(repo, tmp_path, "configs/c_bad.yaml", extra=("--worker", "tuan-1", "--retry-failed")) == 1
    assert len((repo / "calls.log").read_text().splitlines()) == 2


def test_own_failed_claim_blocks_the_same_worker_too(repo, tmp_path, capsys):
    """19/11: a tab retrained a config whose eval had failed in that same tab, leaving two runs of one trial."""
    make_config(repo, "c_bad", fake="crash")
    commit_all(repo)
    assert queue(repo, tmp_path, "configs/c_bad.yaml", extra=("--worker", "vinh-1")) == 1
    capsys.readouterr()
    assert queue(repo, tmp_path, "configs/c_bad.yaml", extra=("--worker", "vinh-1")) == 0
    assert "blocked: failed on vinh-1" in capsys.readouterr().out
    assert len((repo / "calls.log").read_text().splitlines()) == 1
    assert queue(repo, tmp_path, "configs/c_bad.yaml", extra=("--worker", "vinh-1", "--retry-failed")) == 1
    assert len((repo / "calls.log").read_text().splitlines()) == 2


def test_interrupted_worker_drops_its_claim(repo, tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(run_queue, "train_and_upload", stop)
    with pytest.raises(KeyboardInterrupt):
        queue(repo, tmp_path, "configs/a_dfd.yaml", extra=("--worker", "vinh-1"))
    assert not list(run_queue.claim_folder(tmp_path / "drive", claim_key(repo, "a_dfd")).iterdir())


def test_shard_shuffle_and_time_budget(repo, tmp_path, capsys):
    make_config(repo, "c_ce", variant="ti_ce", kd=False)
    commit_all(repo)
    assert queue(repo, tmp_path, "configs/*.yaml", extra=("--shard", "1/2")) == 0
    assert [json.loads((p / "queue.json").read_text())["config_path"] for p in drive_runs_done(tmp_path)] == ["configs/b_ce.yaml"]
    capsys.readouterr()
    assert queue(repo, tmp_path, "configs/*.yaml", extra=("--time-budget-minutes", "10", "--run-minutes", "20")) == 0
    out = capsys.readouterr().out
    assert out.count("not started (time: 10 min left < 20)") == 3  # b_ce was already done, but the budget is checked first
    configs = [Path(f"configs/{c}.yaml") for c in "abcdefgh"]
    orders = set()
    for worker in ("w1", "w2", "w3"):
        shuffled = list(configs)
        run_queue.random.Random(worker).shuffle(shuffled)
        orders.add(tuple(shuffled))
    assert len(orders) > 1  # different tabs start on different configs
    with pytest.raises(SystemExit):
        queue(repo, tmp_path, "configs/*.yaml", extra=("--shard", "2/2"))


def test_two_parallel_workers_split_the_queue_without_duplicates(repo, tmp_path):
    import sys
    for name in ("c_ce", "d_ce", "e_ce", "f_ce"):
        make_config(repo, name, variant="ti_ce", kd=False)
    commit_all(repo)
    workers = []
    for worker in ("tab-a", "tab-b"):
        argv = [sys.executable, str(ROOT / "scripts" / "run_queue.py"), "configs/*.yaml", "--repo", str(repo),
                "--data-root", str(tmp_path), "--teacher-checkpoint", str(tmp_path / "t.ckpt"),
                "--drive-runs", str(tmp_path / "drive"), "--require-tag", "exp-v1", "--device", "cpu",
                "--log-dir", str(tmp_path / f"logs-{worker}"), "--worker", worker, "--shuffle",
                "--claim-settle-seconds", "1"]
        workers.append(subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outputs = [process.communicate(timeout=240)[0] for process in workers]
    assert [process.returncode for process in workers] == [0, 0], outputs
    records = [json.loads((p / "queue.json").read_text(encoding="utf-8")) for p in drive_runs(tmp_path)]
    configs = sorted(record["config_path"] for record in records)
    assert configs == sorted(f"configs/{n}.yaml" for n in ("a_dfd", "b_ce", "c_ce", "d_ce", "e_ce", "f_ce"))
    assert {record["worker"] for record in records} == {"tab-a", "tab-b"}  # both tabs did part of the work
    assert len((repo / "calls.log").read_text().splitlines()) == 6  # no config trained twice


def test_parseq_checkout_is_put_on_pythonpath(repo, tmp_path, monkeypatch):
    parseq = tmp_path / "parseq"
    (parseq / "strhub").mkdir(parents=True)
    monkeypatch.setenv("PARSEQ_ROOT", str(parseq))
    assert queue(repo, tmp_path, "configs/a_dfd.yaml") == 0
    (call,) = [json.loads(line) for line in (repo / "calls.log").read_text().splitlines()]
    assert str(parseq) in call["pythonpath"].split(run_queue.os.pathsep)


def test_rescore_scores_and_uploads_a_rescued_run(repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_EVAL_FAIL", "1")
    assert queue(repo, tmp_path, "configs/a_dfd.yaml", extra=("--worker", "vinh-acc3")) == 1
    assert not drive_runs(tmp_path)  # eval.py failed: nothing uploaded, the trained run is only on the VM
    (local,) = (repo / "results" / "runs").iterdir()
    rescue = tmp_path / "drive_rescue"
    rescue.mkdir()
    shutil.copytree(local, rescue / local.name)
    (rescue / "queue_logs_vinh-acc3").mkdir()  # the rescue cell also copies the logs: not a run folder
    smoke = rescue / "smoke_run"
    shutil.copytree(local, smoke)
    resolved = yaml.safe_load((smoke / "resolved.yaml").read_text(encoding="utf-8"))
    (smoke / "resolved.yaml").write_text(yaml.safe_dump({**resolved, "diagnostic": "fast_dev_run"}), encoding="utf-8")
    monkeypatch.delenv("FAKE_EVAL_FAIL")
    capsys.readouterr()
    assert queue(repo, tmp_path, "configs/*.yaml", extra=("--worker", "vinh-acc1", "--rescore", str(rescue))) == 0
    assert "skipped: smoke test" in capsys.readouterr().out
    (folder,) = drive_runs(tmp_path)
    assert folder.name == local.name
    record = json.loads((folder / "queue.json").read_text(encoding="utf-8"))
    assert record["status"] == "done" and record["config_path"] == "configs/a_dfd.yaml"
    assert record["word_acc_all"] == {"validation": 0.7123} and record["rescored_from"] == str(rescue / local.name)
    assert not (folder / "checkpoints" / "last.ckpt").exists()
    assert len((repo / "calls.log").read_text().splitlines()) == 1  # rescoring trains nothing
    # the queue now counts the config as done (the failed claim no longer matters), and a second rescore is a no-op
    capsys.readouterr()
    assert queue(repo, tmp_path, "configs/a_dfd.yaml", extra=("--worker", "vinh-acc2")) == 0
    assert "already done" in capsys.readouterr().out
    assert queue(repo, tmp_path, "configs/*.yaml", extra=("--rescore", str(rescue))) == 0
    assert len(drive_runs(tmp_path)) == 1 and len((repo / "calls.log").read_text().splitlines()) == 1


def test_rescore_refuses_runs_it_cannot_attribute(repo, tmp_path, capsys):
    assert queue(repo, tmp_path, "configs/a_dfd.yaml") == 0
    (local,) = (repo / "results" / "runs").iterdir()
    rescue = tmp_path / "rescue"
    shutil.copytree(local, rescue / "other_commit")
    meta = json.loads((rescue / "other_commit" / "metadata.json").read_text(encoding="utf-8"))
    (rescue / "other_commit" / "metadata.json").write_text(json.dumps({**meta, "commit_hash": "f" * 40}), encoding="utf-8")
    shutil.copytree(local, rescue / "unknown_config")
    (rescue / "unknown_config" / "resolved.yaml").write_text(yaml.safe_dump({"variant": "ti_kd", "seed": 1}), encoding="utf-8")
    shutil.copytree(local, rescue / "interrupted")
    (rescue / "interrupted" / "metadata.json").write_text(json.dumps({**meta, "status": "interrupted"}), encoding="utf-8")
    shutil.copytree(local, rescue / "short")  # "completed" after 1 of 3 epochs, like 20261119T153000Z_..._6e828460
    (rescue / "short" / "validation.csv").write_text("epoch,word_acc\n1,0.7\n", encoding="utf-8")
    shutil.rmtree(tmp_path / "drive")
    capsys.readouterr()
    assert queue(repo, tmp_path, "configs/*.yaml", extra=("--rescore", str(rescue))) == 1
    out = capsys.readouterr().out
    assert "skipped: trained at ffffffffffff" in out and "skipped: matches 0 queue configs" in out
    assert "status is 'interrupted'" in out and "trained 1 of 3 epochs" in out
    assert not drive_runs(tmp_path)
