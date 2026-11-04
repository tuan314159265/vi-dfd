import csv
import json
import subprocess
from pathlib import Path

import pytest

from dfd.config import load_config, validate_config
from dfd.runtime import config_hash, prepare_run


ROOT = Path(__file__).resolve().parents[1]


def test_research_choices_stay_unresolved_and_seed_is_validated():
    config = load_config(ROOT / "configs/runs/E2/ti_dfd_s42.yaml")
    unresolved = validate_config(config)
    assert "training.epochs" in unresolved
    assert "model.teacher_checkpoint" in unresolved
    assert config["model"]["head"] == "factorized"
    config["seed"] = True
    with pytest.raises(ValueError, match="seed"):
        validate_config(config)


def test_config_hash_tracks_effective_seed():
    config = load_config(ROOT / "configs/runs/E2/ti_dfd_s42.yaml")
    first = config_hash(config)
    assert first == config_hash(dict(reversed(list(config.items()))))
    config["seed"] = 43
    assert first != config_hash(config)


@pytest.fixture
def clean_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("results/runs/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Scaffold test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "test fixture",
    ], check=True, capture_output=True)
    return repo


def test_prepared_run_csv_records_real_commit_and_no_fake_metrics(clean_repo):
    config = load_config(ROOT / "configs/runs/E2/ti_dfd_s42.yaml")
    first = prepare_run(config, clean_repo)
    config["seed"] = 43
    second = prepare_run(config, clean_repo)
    assert first != second
    expected_sha = subprocess.check_output(["git", "-C", str(clean_repo), "rev-parse", "HEAD"], text=True).strip()
    data = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert data["commit_hash"] == expected_sha
    assert data["seed"] == 42
    assert data["status"] == "prepared"
    with (first / "run.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["commit_hash"] == expected_sha
    assert row["config_hash"] == data["config_hash"]
    with (first / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    assert (first / "resolved.yaml").is_file()
    assert (first / "environment.txt").is_file()


def test_run_preparation_rejects_uncommitted_code(clean_repo):
    (clean_repo / "uncommitted.py").write_text("print('changed')\n", encoding="utf-8")
    config = load_config(ROOT / "configs/runs/E2/ti_dfd_s42.yaml")
    with pytest.raises(ValueError, match="Commit code/config changes"):
        prepare_run(config, clean_repo)
