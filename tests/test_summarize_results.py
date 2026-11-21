import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("summarize_results", ROOT / "scripts" / "summarize_results.py")
summarize_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_results)

TUNING_COMMIT, MAIN_COMMIT = "t" * 40, "m" * 40
LOCKED = {"tuning_commit": TUNING_COMMIT, "dfd": {"temperature": 2.0, "lambda": 1.0, "tone_weight": 2.0},
          "kd": {"temperature": 2.0, "lambda": 0.5}}
REFERENCE = {"freeze_manifest_sha256": "F", "lmdb_data_mdb_sha256": {"train": "T", "val": "V", "test": "S", "unseen_test": "U"}}
TAGS = {TUNING_COMMIT: {"exp-v1"}, MAIN_COMMIT: {"exp-v2"}}
FIELDS = ["run_id", "commit_hash", "config_hash", "seed", "split", "subset", "precision", "num_words", "word_acc", "ned_score",
          "base_er", "modifier_er", "tone_er", "insertions", "deletions", "params"]


def kd_section(variant, tau, lam, wt):
    if variant in ("ti_ce", "ti_factorized"):
        return {"enabled": False, "temperature": 2.0, "lambda": 1.0, "component_weights": {"base": 1.0, "modifier": 1.0, "tone": 2.0}}
    return {"enabled": True, "temperature": tau, "lambda": lam, "component_weights": {"base": 1.0, "modifier": 1.0, "tone": wt}}


def make_run(runs, name, *, experiment="E2", stage="train", variant, seed, acc, tau=2.0, lam=1.0, wt=2.0, commit=MAIN_COMMIT,
             validity=True, freeze="F", test_lmdb="S", heavy=None, tone=0.05):
    folder = runs / name
    folder.mkdir(parents=True)
    config = {"experiment": experiment, "stage": stage, "variant": variant, "seed": seed,
              "distillation": kd_section(variant, tau, lam, wt), "composition": {"validity_mask": validity}}
    config["training"] = {"epochs": 20}
    (folder / "resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    status = "prepared" if variant == "no_validity_mask" else "completed"
    (folder / "metadata.json").write_text(json.dumps({"run_id": name, "commit_hash": commit, "status": status}), encoding="utf-8")
    if variant != "no_validity_mask":
        (folder / "data_identity.json").write_text(json.dumps({"freeze_sha256": freeze, "lmdb_sha256": {"train": "T", "val": "V"}}), encoding="utf-8")
    info = {"precision": "fp32", "validity_mask": validity, "splits": {"test": {"lmdb_sha256": test_lmdb}}}
    if not validity:
        info["splits"]["test"]["unmasked_decoding"] = {"positions": 1000, "invalid_triples": 12}
    (folder / "eval_info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = [("test", "all", acc, tone), ("test", "diacritic_heavy", heavy if heavy is not None else acc - 0.05, tone),
            ("unseen_test", "all", acc + 0.01, tone)]
    with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for split, subset, value, tone_er in rows:
            writer.writerow({"run_id": name, "commit_hash": commit, "seed": seed, "split": split, "subset": subset, "precision": "fp32",
                             "num_words": 100, "word_acc": value, "ned_score": 0.9, "base_er": 0.03, "modifier_er": 0.02, "tone_er": tone_er})
    # PR4_SYNTHETIC_VALIDATION_HISTORY
    if status == "completed":
        (folder / "validation.csv").write_text(
            "epoch,word_acc\n"
            + "".join(f"{epoch},0.8\n" for epoch in range(1, 21)),
            encoding="utf-8",
        )
    return folder


def tuned(variant):
    return {"stage": "tune", "commit": TUNING_COMMIT, "lam": 1.0 if variant == "ti_dfd" else 0.5}


def populate(runs, dfd=(0.82, 0.83, 0.84), kd=(0.80, 0.81, 0.82), fac=(0.79, 0.80, 0.81), ce=(0.78, 0.79, 0.80)):
    for seed, acc_dfd, acc_kd in zip((42, 43, 44), dfd, kd):
        for variant, acc in (("ti_dfd", acc_dfd), ("ti_kd", acc_kd)):
            extra = tuned(variant) if seed == 42 else {}
            make_run(runs, f"{variant}_{seed}", variant=variant, seed=seed, acc=acc, lam=extra.get("lam", 1.0 if variant == "ti_dfd" else 0.5),
                     **{k: v for k, v in extra.items() if k in ("stage", "commit")})
    for seed, (a, b) in enumerate(zip(fac, ce)):
        make_run(runs, f"fac_{seed}", variant="ti_factorized", seed=42 + seed, acc=a, commit=TUNING_COMMIT)
        make_run(runs, f"ce_{seed}", variant="ti_ce", seed=42 + seed, acc=b)


def run_summary(runs, **kwargs):
    return summarize_results.summarize(summarize_results.load_runs(runs), LOCKED, REFERENCE, tags_for=TAGS.get, **kwargs)


def test_paired_rule_marks_dfd_better(tmp_path):
    populate(tmp_path)
    result = run_summary(tmp_path)
    assert result["problems"] == []
    text = result["text"]
    assert "| ti_dfd vs ti_kd | +2.00 | +2.00 | +2.00 | +2.00 ± 0.00 | DFD better |" in text
    assert "| ti_dfd | 3 | 82.00 | 83.00 | 84.00 | 83.00 ± 1.00 |" in text


@pytest.mark.parametrize("kd, expected", [
    ((0.82, 0.83, 0.84), "inconclusive"),       # d = 0 for every seed: mean(d) = 0 is not above SD(d) = 0
    ((0.81, 0.83, 0.835), "inconclusive"),      # d = +1, 0, +0.5 points: one seed does not favour DFD
    ((0.819, 0.829, 0.79), "inconclusive"),     # all d > 0 but mean(d) 0.77 < SD(d) 1.7
    ((0.81, 0.81, 0.81), "DFD better"),         # d = 1, 2, 3 points: mean 2 > SD 1
])
def test_decision_three_cases(tmp_path, kd, expected):
    populate(tmp_path, kd=kd)
    line = next(row for row in run_summary(tmp_path)["text"].splitlines() if row.startswith("| ti_dfd vs ti_kd |"))
    assert line.rstrip().endswith(f"| {expected} |")


def test_incomplete_seeds_give_no_verdict_and_list_missing(tmp_path):
    populate(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "ti_kd_44")
    result = run_summary(tmp_path)
    assert "ti_kd seed 44: no run" in " ".join(result["missing"])
    line = next(row for row in result["text"].splitlines() if row.startswith("| ti_dfd vs ti_kd |"))
    assert line.rstrip().endswith("| incomplete |")
    assert result["problems"] == []


def test_run_with_unlocked_hyperparameters_is_a_problem_and_not_used(tmp_path):
    populate(tmp_path)
    make_run(tmp_path, "stale", variant="ti_dfd", seed=43, acc=0.99, tau=4.0)
    import shutil
    shutil.rmtree(tmp_path / "ti_dfd_43")
    result = run_summary(tmp_path)
    assert any("stale" in text and "locked hyperparameters" in text for text in result["problems"])
    assert "ti_dfd seed 43: no run" in " ".join(result["missing"])


def test_wrong_commit_tag_and_wrong_data_are_problems(tmp_path):
    populate(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "ti_dfd_43")
    make_run(tmp_path, "ti_dfd_43", variant="ti_dfd", seed=43, acc=0.83, commit=TUNING_COMMIT)   # exp-v1 only: not allowed for a main run
    shutil.rmtree(tmp_path / "ti_dfd_44")
    make_run(tmp_path, "ti_dfd_44", variant="ti_dfd", seed=44, acc=0.84, freeze="OTHER")
    shutil.rmtree(tmp_path / "ti_kd_43")
    make_run(tmp_path, "ti_kd_43", variant="ti_kd", seed=43, acc=0.81, lam=0.5, test_lmdb="X")
    problems = " ".join(run_summary(tmp_path)["problems"])
    assert "ti_dfd_43: commit" in problems and "expected one of ['exp-v2']" in problems
    assert "ti_dfd_44: freeze manifest differs" in problems
    assert "ti_kd_43: scored on a different 'test' LMDB" in problems


def test_ti_ce_and_factorized_may_use_exp_v1(tmp_path):
    populate(tmp_path)
    assert run_summary(tmp_path)["problems"] == []   # populate() puts ti_factorized on the exp-v1 commit


def test_tuning_run_must_come_from_the_locked_commit(tmp_path):
    populate(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "ti_dfd_42")
    make_run(tmp_path, "ti_dfd_42", variant="ti_dfd", seed=42, acc=0.82, stage="tune", commit="x" * 40)
    assert any("locked tuning commit" in text for text in run_summary(tmp_path)["problems"])


def test_duplicate_runs_need_exclude(tmp_path):
    populate(tmp_path)
    make_run(tmp_path, "ti_dfd_43_again", variant="ti_dfd", seed=43, acc=0.80)
    result = run_summary(tmp_path)
    assert any("2 scored runs" in text for text in result["problems"])
    assert run_summary(tmp_path, exclude=["ti_dfd_43_again"])["problems"] == []


def test_e3_rows_sweep_and_unmasked_note(tmp_path):
    populate(tmp_path)
    for seed, acc in zip((42, 43, 44), (0.80, 0.82, 0.83)):
        make_run(tmp_path, f"nt_{seed}", experiment="E3", variant="no_tone_weight", seed=seed, acc=acc, wt=1.0)
    make_run(tmp_path, "nvm", experiment="E3", variant="no_validity_mask", seed=42, acc=0.81, validity=False)
    for tau in (1.0, 4.0):
        make_run(tmp_path, f"sweep_{tau:g}", stage="tune", variant="ti_dfd", seed=42, acc=0.70 + tau / 100, tau=tau, commit=TUNING_COMMIT)
    result = run_summary(tmp_path)
    assert result["problems"] == []
    text = result["text"]
    assert "| − tone weighting | 3 | 81.67 ± 1.53 |" in text
    assert "invalid triples 12/1000 (1.20% of positions)" in text
    assert "| Temperature sweep τ=4 | 1 | 74.00 |" in text
    assert "| Temperature sweep τ=2 | 1 | 82.00 |" in text      # the selected DFD tuning run doubles as the τ=2 point
    assert text.count("indicative only") >= 4


def test_main_writes_files_and_exit_codes(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    populate(runs)
    (tmp_path / "locked.json").write_text(json.dumps(LOCKED), encoding="utf-8")
    (tmp_path / "ref.json").write_text(json.dumps(REFERENCE), encoding="utf-8")
    monkeypatch.setattr(summarize_results, "git_tags", TAGS.get)
    argv = ["x", "--runs", str(runs), "--out", str(tmp_path / "out"), "--locked", str(tmp_path / "locked.json"), "--reference", str(tmp_path / "ref.json")]
    monkeypatch.setattr(sys, "argv", argv)
    assert summarize_results.main() == 0      # E3 cells missing is fine without --strict
    monkeypatch.setattr(sys, "argv", argv + ["--strict"])
    assert summarize_results.main() == 1
    assert (tmp_path / "out" / "summary.md").exists()
    with (tmp_path / "out" / "runs_used.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 13  # the tau=2 sweep point reuses the selected DFD run


def test_smoke_and_unfinished_runs_are_never_counted(tmp_path):
    runs = tmp_path / "runs"
    populate(runs)
    smoke = make_run(runs, "zz_smoke", variant="ti_ce", seed=42, acc=0.99)
    config = yaml.safe_load((smoke / "resolved.yaml").read_text(encoding="utf-8"))
    config["diagnostic"] = "fast_dev_run"
    (smoke / "resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    crashed = make_run(runs, "zz_failed", variant="ti_kd", seed=43, acc=0.99)
    meta = json.loads((crashed / "metadata.json").read_text(encoding="utf-8"))
    (crashed / "metadata.json").write_text(json.dumps({**meta, "status": "interrupted"}), encoding="utf-8")
    skipped = []
    loaded = summarize_results.load_runs(runs, skipped)
    result = summarize_results.summarize(loaded, LOCKED, REFERENCE, tags_for=TAGS.get, skipped=skipped)
    assert not result["problems"]  # no "several runs for one cell"
    assert result["chosen"][("ti_ce", 42)]["id"] == "ce_0" and result["chosen"][("ti_kd", 43)]["id"] == "ti_kd_43"
    assert any("zz_smoke" in w and "diagnostic" in w for w in result["warnings"])
    assert any("zz_failed" in w and "interrupted" in w for w in result["warnings"])


def test_trained_run_left_in_prepared_state_is_not_counted(tmp_path):
    runs = tmp_path / "runs"
    folder = make_run(runs, "dfd", variant="ti_dfd", seed=43, acc=0.8)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    (folder / "metadata.json").write_text(json.dumps({**meta, "status": "prepared"}), encoding="utf-8")
    assert summarize_results.load_runs(runs) == []
    make_run(runs, "mask", experiment="E3", variant="no_validity_mask", seed=42, acc=0.8, validity=False)
    assert [run["id"] for run in summarize_results.load_runs(runs)] == ["mask"]


def test_missing_test_provenance_is_rejected(tmp_path):
    populate(tmp_path)
    path = tmp_path / "ti_dfd_43" / "eval_info.json"
    info = json.loads(path.read_text())
    info["splits"] = {}
    path.write_text(json.dumps(info))
    result = run_summary(tmp_path)
    assert any("does not cover split 'test'" in p for p in result["problems"])
    assert ("ti_dfd", 43) not in result["chosen"]
