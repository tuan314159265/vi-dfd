import importlib.util
import json
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
    Path("calls.log").open("a").write(json.dumps({"teacher": a.teacher_checkpoint, "platform": os.environ.get("DFD_PLATFORM")}) + "\\n")
    if mode == "crash":
        raise SystemExit(3)
    if not a.smoke_test:
        (run / "checkpoints" / "best-epoch=03-validation_word_acc=0.7000.ckpt").write_bytes(b"best")
        (run / "checkpoints" / "last.ckpt").write_bytes(b"last")
        if mode == "two_best":
            (run / "checkpoints" / "best-epoch=04-validation_word_acc=0.7100.ckpt").write_bytes(b"best")
    status = "smoke_completed" if a.smoke_test else ("interrupted" if mode == "interrupted" else "completed")
    (run / "metadata.json").write_text(json.dumps({"run_id": run.name, "status": status}), encoding="utf-8")
''')
FAKE_EVAL = textwrap.dedent('''
    import argparse, yaml
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir"); p.add_argument("--checkpoint"); p.add_argument("--device")
    a = p.parse_args()
    run = Path(a.run_dir)
    config = yaml.safe_load((run / "resolved.yaml").read_text(encoding="utf-8"))
    splits = [] if config.get("fake") == "no_rows" else config["evaluation"]["splits"]
    with (run / "metrics.csv").open("a", encoding="utf-8") as handle:
        for split in splits:
            handle.write(f"{run.name},{split},all,0.7123\\n")
''')


def make_config(repo, name, variant="ti_dfd", kd=True, fake="ok", splits=("validation",)):
    config = {"experiment": "E2", "stage": "tune", "variant": variant, "seed": 42, "fake": fake,
              "distillation": {"enabled": kd}, "evaluation": {"splits": list(splits)}}
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
            "--log-dir", str(tmp_path / "logs"), *extra]
    return run_queue.main(argv)


def drive_runs(tmp_path):
    return sorted(path for path in (tmp_path / "drive").iterdir())


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
                                          ("two_best", "exactly one best"), ("no_rows", "no 'all' row")])
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
    commit_all(repo)
    assert queue(repo, tmp_path, "configs/d_mask.yaml", "configs/a_dfd.yaml", extra=("--smoke-test",)) == 0
    (folder,) = drive_runs(tmp_path)  # the eval-only config is skipped, the smoke run has no checkpoint to score
    record = json.loads((folder / "queue.json").read_text(encoding="utf-8"))
    assert record["smoke_test"] is True and record["best_checkpoint"] is None and record["word_acc_all"] == {}
    assert yaml.safe_load((folder / "resolved.yaml").read_text(encoding="utf-8"))["diagnostic"] == "fast_dev_run"
    # a smoke copy never counts as a finished real run
    assert run_queue.finished(tmp_path / "drive", {**{k: record[k] for k in ("config_path", "config_sha256", "commit")},
                                                   "smoke_test": False}) is None
