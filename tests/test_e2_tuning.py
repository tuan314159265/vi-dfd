import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e2_tuning", ROOT / "scripts" / "e2_tuning.py")
e2_tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2_tuning)


@pytest.fixture
def policy(tmp_path):
    data = yaml.safe_load((ROOT / "configs/tuning.yaml").read_text(encoding="utf-8"))
    data["equal_budget_policy"] = "test_policy"
    path = tmp_path / "tuning.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_make_refuses_without_budget_policy(tmp_path):
    with pytest.raises(SystemExit, match="equal_budget_policy"):
        e2_tuning.make(ROOT / "configs/tuning.yaml", tmp_path / "out")


def test_make_writes_full_grids(policy, tmp_path):
    written = e2_tuning.make(policy, tmp_path / "out")
    names = {path.stem for path in written}
    assert len([name for name in names if name.startswith("ti_dfd")]) == 12
    assert len([name for name in names if name.startswith("ti_kd")]) == 6
    config = yaml.safe_load((tmp_path / "out" / "ti_dfd_tau4_lam0.5_wt1.yaml").read_text(encoding="utf-8"))
    assert config["seed"] == 42 and config["stage"] == "tune"
    assert config["distillation"]["temperature"] == 4.0
    assert config["distillation"]["lambda"] == 0.5
    assert config["distillation"]["component_weights"]["tone"] == 1.0
    assert config["evaluation"]["splits"] == ["validation"]


def write_run(runs, name, variant, tau, lam, wt, word_acc, ned=0.9, split="validation"):
    folder = runs / name
    folder.mkdir(parents=True)
    kd = {"temperature": tau, "lambda": lam, "component_weights": {"base": 1.0, "modifier": 1.0, "tone": wt}}
    config = {"experiment": "E2", "stage": "tune", "variant": variant, "distillation": kd}
    (folder / "resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["commit_hash", "split", "subset", "word_acc", "ned_score"])
        writer.writeheader()
        writer.writerow({"commit_hash": "a" * 40, "split": split, "subset": "all", "word_acc": word_acc, "ned_score": ned})
        writer.writerow({"commit_hash": "a" * 40, "split": "test", "subset": "all", "word_acc": 0.99, "ned_score": 0.99})


def test_select_uses_validation_and_breaks_ties(policy, tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "r1", "ti_dfd", 2.0, 1.0, 2.0, 0.80, ned=0.90)
    write_run(runs, "r2", "ti_dfd", 1.0, 0.5, 1.0, 0.80, ned=0.92)  # same accuracy, higher 1-NED
    write_run(runs, "r3", "ti_dfd", 4.0, 1.0, 2.0, 0.70)
    write_run(runs, "r4", "ti_kd", 1.0, 0.5, 1.0, 0.75)
    write_run(runs, "r5", "ti_kd", 2.0, 0.5, 1.0, 0.75)  # exact tie: earlier grid entry wins
    best = e2_tuning.select(policy, runs)
    assert best["ti_dfd"]["run_id"] == "r2"
    assert best["ti_kd"]["run_id"] == "r4"
