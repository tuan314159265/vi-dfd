import csv
import importlib.util
import json
import shutil
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


def test_make_refuses_without_budget_policy(policy, tmp_path):
    data = yaml.safe_load(policy.read_text(encoding="utf-8"))
    data["equal_budget_policy"] = None
    policy.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="equal_budget_policy"):
        e2_tuning.make(policy, tmp_path / "out")


def test_committed_policy_gives_equal_budgets():
    data = yaml.safe_load((ROOT / "configs/tuning.yaml").read_text(encoding="utf-8"))
    assert data["equal_budget_policy"]
    assert len(e2_tuning.grid_trials(data, "ti_dfd")) == len(e2_tuning.grid_trials(data, "ti_kd")) == 12
    assert data["standard_kd_grid"]["temperature"] == data["dfd_grid"]["temperature"]


def test_make_writes_full_grids(policy, tmp_path):
    written = e2_tuning.make(policy, tmp_path / "out")
    names = {path.stem for path in written}
    assert len([name for name in names if name.startswith("ti_dfd")]) == 12
    assert len([name for name in names if name.startswith("ti_kd")]) == 12
    config = yaml.safe_load((tmp_path / "out" / "ti_dfd_tau4_lam0.5_wt1.yaml").read_text(encoding="utf-8"))
    assert config["seed"] == 42 and config["stage"] == "tune"
    assert config["distillation"]["temperature"] == 4.0
    assert config["distillation"]["lambda"] == 0.5
    assert config["distillation"]["component_weights"]["tone"] == 1.0
    assert config["evaluation"]["splits"] == ["validation"]


def write_run(runs, name, variant, tau, lam, wt, word_acc, ned=0.9, split="validation", status="completed", diagnostic=None):
    folder = runs / name
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"run_id": name, "status": status}), encoding="utf-8")
    kd = {"temperature": tau, "lambda": lam, "component_weights": {"base": 1.0, "modifier": 1.0, "tone": wt}}
    config = {"experiment": "E2", "stage": "tune", "variant": variant, "seed": 42, "distillation": kd}
    if diagnostic:
        config["diagnostic"] = diagnostic
    (folder / "resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["commit_hash", "split", "subset", "word_acc", "ned_score"])
        writer.writeheader()
        writer.writerow({"commit_hash": "a" * 40, "split": split, "subset": "all", "word_acc": word_acc, "ned_score": ned})
        writer.writerow({"commit_hash": "a" * 40, "split": "test", "subset": "all", "word_acc": 0.99, "ned_score": 0.99})
    # PR4_FIX_COMPLETE_VALIDATION_FIXTURE
    # A normal synthetic completed run must include epoch evidence.
    run_dir = runs / name
    resolved = run_dir / "resolved.yaml"
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    config.setdefault("training", {})["epochs"] = 20
    resolved.write_text(yaml.safe_dump(config), encoding="utf-8")

    (run_dir / "validation.csv").write_text(
        "epoch,word_acc\n".replace("\\n", "\n")
        + "".join(f"{epoch},0.8\n" for epoch in range(1, 21)),
        encoding="utf-8",
    )



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


def write_full_grid(runs, policy_path, best_dfd=(4.0, 0.5, 2.0), best_kd=(1.0, 2.0)):
    data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    for values in e2_tuning.grid_trials(data, "ti_dfd"):
        key = (values["temperature"], values["lambda"], values["tone_weight"])
        write_run(runs, "dfd_" + e2_tuning.trial_name("ti_dfd", values), "ti_dfd", *map(float, key), 0.90 if key == best_dfd else 0.50)
    for values in e2_tuning.grid_trials(data, "ti_kd"):
        key = (values["temperature"], values["lambda"])
        write_run(runs, "kd_" + e2_tuning.trial_name("ti_kd", values), "ti_kd", *map(float, key), 1.0, 0.85 if key == best_kd else 0.50)


@pytest.fixture
def locked_setup(policy, tmp_path):
    runs = tmp_path / "runs"
    write_full_grid(runs, policy)
    configs = tmp_path / "runs_cfg"
    import shutil
    shutil.copytree(ROOT / "configs/runs", configs)
    return runs, configs, tmp_path / "locked.json"


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_lock_writes_selected_values_by_rule(policy, locked_setup):
    runs, configs, out = locked_setup
    e2_tuning.lock(policy, runs, configs, out)
    dfd = ("temperature", "lambda"), (4.0, 0.5)
    for path in (configs / "E2/ti_dfd_s43.yaml", configs / "E4/ti_dfd_s44_int8_dynamic.yaml", configs / "E5/ti_dfd_s42_fraction25.yaml",
                 configs / "E6/ti_dfd_s42_fp32.yaml", configs / "E7/ti_dfd_s42.yaml", configs / "E3/marginal_joint_kd_s42.yaml",
                 configs / "E3/tone_only_kd_s42.yaml", configs / "E3/no_validity_mask_s42.yaml"):
        kd = load(path)["distillation"]
        assert (kd["temperature"], kd["lambda"], kd["component_weights"]["tone"]) == (4.0, 0.5, 2.0), path
    for path in (configs / "E2/ti_kd_s44.yaml", configs / "E5/ti_kd_s42.yaml", configs / "E6/ti_kd_s42_int8_dynamic.yaml"):
        kd = load(path)["distillation"]
        assert (kd["temperature"], kd["lambda"]) == (1.0, 2.0), path
    no_tone = load(configs / "E3/no_tone_weight_s43.yaml")["distillation"]
    assert (no_tone["temperature"], no_tone["lambda"], no_tone["component_weights"]["tone"], no_tone["tone_confidence_weight"]) == (4.0, 0.5, 1.0, False)
    tone_only = load(configs / "E3/tone_only_kd_s42.yaml")["distillation"]["component_weights"]
    assert (tone_only["base"], tone_only["modifier"]) == (0.0, 0.0)
    joint = load(configs / "E3/joint_kd_s42.yaml")["distillation"]
    assert (joint["temperature"], joint["lambda"]) == (4.0, 0.5)
    assert load(configs / "E3/marginal_joint_kd_s42.yaml")["distillation"]["joint_lambda"] == 1.0
    sweep = [load(configs / f"E3/temperature_sweep_s42_tau{t}.yaml")["distillation"] for t in (1, 2, 4)]
    assert [kd["temperature"] for kd in sweep] == [1.0, 2.0, 4.0]
    assert all((kd["lambda"], kd["component_weights"]["tone"]) == (0.5, 2.0) for kd in sweep)
    record = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert record["dfd"]["temperature"] == 4.0 and record["kd"]["lambda"] == 2.0 and record["tuning_commit"] == "a" * 40


def test_lock_leaves_other_configs_byte_identical(policy, locked_setup):
    runs, configs, out = locked_setup
    untouched = [path for path in configs.rglob("*.yaml") if load(path)["variant"] in {"ti_ce", "ti_factorized", "teacher", "crnn", "vietocr"}]
    before = {path: path.read_bytes() for path in untouched}
    e2_tuning.lock(policy, runs, configs, out)
    assert untouched and all(path.read_bytes() == data for path, data in before.items())


def test_lock_keeps_comments_and_is_idempotent(policy, locked_setup):
    runs, configs, out = locked_setup
    first = (configs / "E2/ti_dfd_s43.yaml").read_text(encoding="utf-8").splitlines()[0]
    e2_tuning.lock(policy, runs, configs, out)
    assert (configs / "E2/ti_dfd_s43.yaml").read_text(encoding="utf-8").startswith(first)
    snapshot = {path: path.read_bytes() for path in configs.rglob("*.yaml")}
    assert e2_tuning.lock(policy, runs, configs, out) == []
    assert all(path.read_bytes() == data for path, data in snapshot.items())


def test_lock_dry_run_writes_nothing(policy, locked_setup):
    runs, configs, out = locked_setup
    snapshot = {path: path.read_bytes() for path in configs.rglob("*.yaml")}
    changes = e2_tuning.lock(policy, runs, configs, out, dry_run=True)
    assert changes and not out.exists()
    assert all(path.read_bytes() == data for path, data in snapshot.items())


def test_lock_refuses_incomplete_or_mixed_tuning(policy, locked_setup):
    runs, configs, out = locked_setup
    import shutil
    shutil.rmtree(runs / "kd_ti_kd_tau2_lam0.5")
    with pytest.raises(SystemExit, match="11 of 12|have no results"):
        e2_tuning.lock(policy, runs, configs, out)
    e2_tuning.lock(policy, runs, configs, out, dry_run=True, allow_incomplete=True)
    config_csv = runs / "dfd_ti_dfd_tau1_lam0.5_wt1" / "metrics.csv"
    config_csv.write_text(config_csv.read_text(encoding="utf-8").replace("a" * 40, "b" * 40), encoding="utf-8")
    with pytest.raises(SystemExit, match="different commits"):
        e2_tuning.lock(policy, runs, configs, out, allow_incomplete=True)


def test_committed_configs_are_covered_by_lock_rules():
    seen = {load(path)["variant"] for path in (ROOT / "configs/runs").rglob("*.yaml")}
    needs_kd = {variant for variant in seen if load(next(p for p in (ROOT / "configs/runs").rglob("*.yaml") if load(p)["variant"] == variant))["distillation"]["enabled"]}
    assert needs_kd <= set(e2_tuning.LOCK_RULES)


def test_select_ignores_smoke_and_unfinished_runs(policy, tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "good", "ti_dfd", 2.0, 1.0, 2.0, 0.80)
    write_run(runs, "smoke", "ti_dfd", 1.0, 0.5, 1.0, 0.99, diagnostic="fast_dev_run")
    write_run(runs, "crashed", "ti_dfd", 4.0, 0.5, 1.0, 0.99, status="failed")
    write_run(runs, "running", "ti_dfd", 4.0, 1.0, 1.0, 0.99, status="running")
    assert e2_tuning.select(policy, runs)["ti_dfd"]["run_id"] == "good"


def test_select_ignores_completed_runs_that_stopped_early(policy, tmp_path, capsys):
    """19/11: a Colab run said "completed" after 5.7 of 20 epochs; it must not fill its grid cell."""
    runs = tmp_path / "runs"
    for name, last in (("full", 20), ("short", 5)):
        write_run(runs, name, "ti_dfd", 2.0, 1.0, 2.0 if name == "full" else 1.0, 0.80 if name == "full" else 0.99)
        resolved = runs / name / "resolved.yaml"
        resolved.write_text(yaml.safe_dump({**yaml.safe_load(resolved.read_text(encoding="utf-8")),
                                            "training": {"epochs": 20}}), encoding="utf-8")
        (runs / name / "validation.csv").write_text("epoch,word_acc\n" + "".join(f"{e},0.8\n" for e in range(1, last + 1)),
                                                    encoding="utf-8")
    assert e2_tuning.select(policy, runs)["ti_dfd"]["run_id"] == "full"
    assert "Skipping short: trained 5 of 20 epochs" in capsys.readouterr().out


def test_select_ignores_completed_runs_missing_validation_csv(policy, tmp_path, capsys):
    """A completed run without epoch history must not fill a tuning grid cell."""
    runs = tmp_path / "runs"
    for name, score in (("full", 0.80), ("missing", 0.99)):
        write_run(runs, name, "ti_dfd", 2.0, 1.0, 2.0 if name == "full" else 1.0, score)
        resolved = runs / name / "resolved.yaml"
        config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        config["training"] = {"epochs": 20}
        resolved.write_text(yaml.safe_dump(config), encoding="utf-8")
        if name == "full":
            (runs / name / "validation.csv").write_text(
                "epoch,word_acc\n" + "".join(f"{epoch},0.8\n" for epoch in range(1, 21)),
                encoding="utf-8",
            )
        else:
            (runs / name / "validation.csv").unlink(missing_ok=True)
            reason = e2_tuning.run_problem(runs / name, config)
            assert "validation.csv is missing" in reason

    assert e2_tuning.select(policy, runs)["ti_dfd"]["run_id"] == "full"
    assert "Skipping missing: validation.csv is missing" in capsys.readouterr().out


def test_lock_refuses_duplicate_trials_until_excluded(policy, locked_setup):
    runs, configs, out = locked_setup
    shutil.copytree(runs / "dfd_ti_dfd_tau1_lam0.5_wt1", runs / "dfd_rerun")
    with pytest.raises(SystemExit, match="ti_dfd_tau1_lam0.5_wt1 has 2 runs"):
        e2_tuning.lock(policy, runs, configs, out)
    e2_tuning.lock(policy, runs, configs, out, exclude=["dfd_rerun"])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["excluded_runs"] == ["dfd_rerun"] and len(record["tuning_runs"]) == 24


def test_select_ignores_outside_grid_and_wrong_seed(policy, tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "valid", "ti_kd", 2., 1., 1., .8)
    write_run(runs, "outside", "ti_kd", 99., 99., 1., .99)
    write_run(runs, "wrong_seed", "ti_kd", 1., .5, 1., .98)
    path = runs / "wrong_seed" / "resolved.yaml"
    config = yaml.safe_load(path.read_text())
    config["seed"] = 43
    path.write_text(yaml.safe_dump(config))
    assert e2_tuning.select(policy, runs)["ti_kd"]["run_id"] == "valid"
