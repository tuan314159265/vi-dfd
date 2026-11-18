"""Collect the E2 and E3 results into the report tables (Task C2).

    python scripts/summarize_results.py --runs results/runs --out results/summary

Reads every run folder (resolved.yaml, metadata.json, metrics.csv, data_identity.json, eval_info.json),
checks that each run used for a table cell is trustworthy, then applies decision 3 of docs/EXPERIMENTS.md:

    d_s = acc(DFD, seed s) - acc(other, seed s)           acc = word accuracy on test, subset all
    "DFD better" only if mean(d) > SD(d) and every d_s > 0   (SD with n - 1); otherwise "inconclusive".

The rule is a heuristic with n = 3 (about t > 1.73), not a significance test. Other metrics are descriptive.

A run is checked before it is used:
  * its hyperparameters equal configs/locked_hyperparameters.json (written by e2_tuning.py lock);
  * its commit carries the right tag (exp-v2 for main runs; ti_ce and ti_factorized may use exp-v1; tuning
    runs must come from the commit recorded at lock time);
  * data_identity.json and eval_info.json match configs/data_reference.json (freeze manifest and LMDB hashes).
Smoke tests (--smoke-test) and runs whose metadata status is not "completed" are never counted (listed under
Warnings). Problems make the exit code 1. Cells without a run are reported as missing (exit 1 only with --strict).
Several runs for one cell are a problem: pass --exclude <run_id> for the ones that must not count.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import e2_tuning  # noqa: E402


SEEDS = (42, 43, 44)
E2_VARIANTS = ("ti_ce", "ti_factorized", "ti_kd", "ti_dfd")
BASELINES = ("ti_kd", "ti_factorized", "ti_ce")
THREE_SEED_E3 = ("no_tone_weight", "joint_kd")
ONE_SEED_E3 = ("marginal_joint_kd", "tone_only_kd", "no_validity_mask")
PRIMARY = ("test", "all", "word_acc")
ANY_TAG = ("exp-v1", "exp-v2")
MAIN_TAG = ("exp-v2",)
SPLIT_DIR = {"validation": "val"}
TOL = 1e-9


# ----- loading -----

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


EVAL_ONLY = ("no_validity_mask",)  # made by `eval.py --config`, so their status stays "prepared"


def load_runs(runs_dir: Path, skipped: list | None = None) -> list[dict]:
    """Every run folder that may count. Smoke tests and unfinished runs go to `skipped` with the reason."""
    runs = []
    for folder in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if not (folder / "resolved.yaml").exists() or not (folder / "metrics.csv").exists():
            continue
        config = yaml.safe_load((folder / "resolved.yaml").read_text(encoding="utf-8"))
        problem = e2_tuning.run_problem(folder, config, allow_prepared=config.get("variant") in EVAL_ONLY)
        if problem:
            if skipped is not None:
                skipped.append(f"{folder.name}: {problem}")
            continue
        with (folder / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        runs.append({
            "id": folder.name, "config": config, "meta": read_json(folder / "metadata.json") or {}, "rows": rows,
            "data": read_json(folder / "data_identity.json"), "eval": read_json(folder / "eval_info.json"),
        })
    return runs


def metric(run: dict, split: str, subset: str, field: str):
    found = [row for row in run["rows"]
             if row["split"] == split and row["subset"] == subset and row.get("precision", "fp32") in ("fp32", "")]
    if len(found) != 1 or found[0].get(field, "") == "":
        return None
    return float(found[0][field])


def commit_of(run: dict) -> str:
    return run["meta"].get("commit_hash") or (run["rows"][0]["commit_hash"] if run["rows"] else "")


def git_tags(commit: str):
    """Tags pointing at `commit`, or None when the commit is unknown to the local repository."""
    try:
        out = subprocess.check_output(["git", "-C", str(ROOT), "tag", "--points-at", commit], text=True, stderr=subprocess.PIPE)
    except (OSError, subprocess.SubprocessError):
        return None
    return set(out.split())


# ----- which run fills which cell -----

def kd_matches(config: dict, expect: dict) -> bool:
    have = e2_tuning.current_values(config)
    return all(have[key] is not None and abs(have[key] - value) < TOL for key, value in expect.items())


def build_slots(locked: dict) -> list[dict]:
    slots = []

    def add(table, label, seed, match):
        slots.append({"table": table, "label": label, "seed": seed, "match": match})

    def same(experiment, variant, seed, expect, **extra):
        def match(run):
            config = run["config"]
            return (config["experiment"] == experiment and config["variant"] == variant and config["seed"] == seed
                    and kd_matches(config, expect)
                    and all((config.get("composition") or {}).get(key, True) == value for key, value in extra.items()))
        return match

    for variant in E2_VARIANTS:  # ti_dfd / ti_kd seed 42 is the winning tuning trial (stage tune), found by its values
        for seed in SEEDS:
            add("E2", variant, seed, same("E2", variant, seed, e2_tuning.expected_values(variant, locked)))
    for variant in (*THREE_SEED_E3, *ONE_SEED_E3):
        extra = {"validity_mask": False} if variant == "no_validity_mask" else {}
        for seed in (SEEDS if variant in THREE_SEED_E3 else (42,)):
            add("E3", variant, seed, same("E3", variant, seed, e2_tuning.expected_values(variant, locked), **extra))
    for tau in (1.0, 2.0, 4.0):  # sweep = the DFD tuning trials with the locked lambda and tone weight
        expect = {"temperature": tau, "lambda": float(locked["dfd"]["lambda"]), "tone_weight": float(locked["dfd"]["tone_weight"])}
        from_tuning, from_e3 = same("E2", "ti_dfd", 42, expect), same("E3", "temperature_sweep", 42, expect)
        add("E3", f"temperature_sweep_tau{tau:g}", 42, lambda run, a=from_tuning, b=from_e3: a(run) or b(run))
    return slots


def is_tuning(run: dict) -> bool:
    return run["config"].get("stage") == "tune"


# ----- trust checks -----

def check_run(run: dict, reference: dict, locked: dict, tags_for) -> tuple[list[str], list[str]]:
    problems, warnings = [], []
    name, config, commit = run["id"], run["config"], commit_of(run)
    if is_tuning(run):
        if locked.get("tuning_commit") and commit != locked["tuning_commit"]:
            problems.append(f"{name}: tuning run commit {commit[:12]} differs from the locked tuning commit {locked['tuning_commit'][:12]}")
    elif tags_for is not None:
        allowed = ANY_TAG if config["variant"] in ("ti_ce", "ti_factorized") else MAIN_TAG
        tags = tags_for(commit)
        if tags is None:
            problems.append(f"{name}: commit {commit[:12]} is not in the local repository (git fetch --tags)")
        elif not tags & set(allowed):
            problems.append(f"{name}: commit {commit[:12]} has tags {sorted(tags) or 'none'}, expected one of {list(allowed)}")

    needs_training_data = config["variant"] != "no_validity_mask"
    data = run["data"]
    if data is None:
        if needs_training_data:
            problems.append(f"{name}: data_identity.json is missing")
    else:
        if data.get("freeze_sha256") != reference["freeze_manifest_sha256"]:
            problems.append(f"{name}: freeze manifest differs from configs/data_reference.json")
        for split in ("train", "val"):
            if (data.get("lmdb_sha256") or {}).get(split) != reference["lmdb_data_mdb_sha256"][split]:
                problems.append(f"{name}: training data LMDB '{split}' differs from configs/data_reference.json")

    info = run["eval"]
    if info is None:
        problems.append(f"{name}: eval_info.json is missing")
        return problems, warnings
    for split, entry in (info.get("splits") or {}).items():
        expected = reference["lmdb_data_mdb_sha256"].get(SPLIT_DIR.get(split, split))
        if entry.get("lmdb_sha256") != expected:
            problems.append(f"{name}: scored on a different '{split}' LMDB than configs/data_reference.json")
    if "test" not in (info.get("splits") or {}):
        warnings.append(f"{name}: eval_info.json does not cover split 'test' (a later eval.py call overwrote it)")
    if info.get("precision") not in ("fp32", None):
        problems.append(f"{name}: scored at precision {info['precision']}")
    want_mask = config["variant"] != "no_validity_mask"
    if info.get("validity_mask", True) != want_mask:
        problems.append(f"{name}: scored with validity_mask={info.get('validity_mask')}, expected {want_mask}")
    return problems, warnings


# ----- statistics and formatting -----

def mean_sd(values: list[float]):
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def pct(value, digits=2, signed=False):
    if value is None:
        return "—"
    return f"{value * 100:+.{digits}f}" if signed else f"{value * 100:.{digits}f}"


def pm(values: list[float], digits=2, signed=False) -> str:
    mean, sd = mean_sd(values)
    if mean is None:
        return "—"
    return pct(mean, digits, signed) + (f" ± {sd * 100:.{digits}f}" if sd is not None else "")


def verdict(diffs: list, better: str = "DFD better") -> str:
    if len(diffs) != len(SEEDS) or any(d is None for d in diffs):
        return "incomplete"
    mean, sd = mean_sd(diffs)
    return better if mean > sd + TOL and all(d > 0 for d in diffs) else "inconclusive"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ----- the summary -----

def summarize(runs: list[dict], locked: dict, reference: dict, tags_for=git_tags, exclude=(), strict_tags=True,
              skipped=()) -> dict:
    problems, missing = [], []
    warnings = [f"not counted: {item}" for item in skipped]
    excluded = set(exclude)
    pool = [run for run in runs if run["id"] not in excluded]
    chosen, used = {}, []
    matched_ids = set()
    for slot in build_slots(locked):
        candidates = [run for run in pool if slot["match"](run)]
        matched_ids.update(run["id"] for run in candidates)
        scored = [run for run in candidates if metric(run, *PRIMARY) is not None]
        key = (slot["label"], slot["seed"])
        if len(scored) > 1:
            problems.append(f"{slot['label']} seed {slot['seed']}: {len(scored)} scored runs {[r['id'] for r in scored]}; pass --exclude for the ones that must not count")
        elif not scored:
            missing.append(f"{slot['label']} seed {slot['seed']}: " + ("not scored on test yet (" + candidates[0]["id"] + ")" if candidates else "no run"))
        else:
            run = scored[0]
            run_problems, run_warnings = check_run(run, reference, locked, tags_for if strict_tags else None)
            for text in run_problems:
                if text not in problems:
                    problems.append(text)
            for text in run_warnings:
                if text not in warnings:
                    warnings.append(text)
            chosen[key] = run
            used.append({"table": slot["table"], "label": slot["label"], "seed": slot["seed"], "run_id": run["id"],
                         "stage": run["config"].get("stage"), "commit_hash": commit_of(run)})
    slot_variants = {slot["label"] for slot in build_slots(locked)} | {"temperature_sweep"}
    for run in pool:  # main runs that exist but do not carry the locked hyperparameters
        config = run["config"]
        if (config["experiment"] in ("E2", "E3") and not is_tuning(run) and run["id"] not in matched_ids
                and (config["variant"] in slot_variants) and config["seed"] in SEEDS):
            problems.append(f"{run['id']}: {config['variant']} seed {config['seed']} does not carry the locked hyperparameters "
                            f"({e2_tuning.current_values(config)}); it was probably trained before lock")

    def value(label, seed, split, subset, field):
        run = chosen.get((label, seed))
        return None if run is None else metric(run, split, subset, field)

    def series(label, split, subset, field, seeds=SEEDS):
        return [value(label, seed, split, subset, field) for seed in seeds]

    def present(values):
        return [v for v in values if v is not None]

    def diffs(label, other, split, subset, field, seeds=SEEDS, sign=1):
        a, b = series(label, split, subset, field, seeds), series(other, split, subset, field, seeds)
        return [None if x is None or y is None else sign * (x - y) for x, y in zip(a, b)]

    text = ["# E2 and E3 results", ""]
    text += ["Word accuracy is in %, on `test`, subset `all`, FP32; ± is the sample SD over seeds (n − 1). "
             "d_s = DFD − other per seed, in percentage points.", ""]

    # E2
    rows = []
    for variant in E2_VARIANTS:
        values = series(variant, *PRIMARY[:2], "word_acc")
        rows.append([variant, str(len(present(values))), *(pct(v) for v in values), pm(present(values))])
    text += ["## E2: students (test, all)", "", table(["Model", "Seeds", "s42", "s43", "s44", "mean ± SD"], rows), ""]

    rows = []
    for base in BASELINES:
        d = diffs("ti_dfd", base, "test", "all", "word_acc")
        mean, sd = mean_sd(present(d))
        rows.append([f"ti_dfd vs {base}", *(pct(v, signed=True) for v in d),
                     pct(mean, signed=True) + (f" ± {sd * 100:.2f}" if sd is not None else "") if mean is not None else "—",
                     verdict(d)])
    text += ["## E2: paired differences, decision 3", "",
             table(["Pair", "d_42", "d_43", "d_44", "mean(d) ± SD(d)", "Verdict"], rows), "",
             "Verdict rule: mean(d) > SD(d) and all three d_s > 0. Heuristic, not a significance test. "
             "`ti_dfd vs ti_kd` answers RQ2.", ""]

    descriptive = (("diacritic_heavy word acc", "test", "diacritic_heavy", "word_acc", 1),
                   ("unseen_test word acc", "unseen_test", "all", "word_acc", 1),
                   ("test tone ER", "test", "all", "tone_er", 1), ("test base ER", "test", "all", "base_er", 1))
    rows = []
    for variant in E2_VARIANTS:
        rows.append([variant, *(pm(present(series(variant, split, subset, field))) for _, split, subset, field, _ in descriptive)])
    text += ["## E2: other metrics (descriptive only, %)", "", table(["Model", *(d[0] for d in descriptive)], rows), ""]
    rows = []
    for base in BASELINES:
        cells = []
        for _, split, subset, field, sign in descriptive:
            d = diffs("ti_dfd", base, split, subset, field, sign=sign)
            cells.append(" / ".join(pct(v, signed=True) for v in d))
        rows.append([f"ti_dfd − {base}", *cells])
    text += ["Per-seed differences DFD − other (s42 / s43 / s44, percentage points; for ER, negative means DFD errs less):", "",
             table(["Pair", *(d[0] for d in descriptive)], rows), ""]

    # E3
    rows = []

    def e3_row(label, shown, seeds, ref_label="ti_dfd", note=""):
        base_values = series(label, "test", "all", "word_acc", seeds)
        d = diffs(ref_label, label, "test", "all", "word_acc", seeds)
        three = len(seeds) == len(SEEDS)
        rows.append([shown, str(len(present(base_values))),
                     pm(present(base_values)),
                     pm(present(series(label, "test", "diacritic_heavy", "word_acc", seeds))),
                     pm(present(series(label, "test", "all", "tone_er", seeds))),
                     pm(present(series(label, "test", "all", "base_er", seeds))),
                     " / ".join(pct(v, signed=True) for v in d) if label != ref_label else "—",
                     ("—" if label == ref_label else verdict(d) if three else "indicative only"), note])

    e3_row("ti_dfd", "Full DFD", SEEDS, note="reference")
    for variant, shown in (("no_tone_weight", "− tone weighting"), ("joint_kd", "Joint KD on factorized head")):
        e3_row(variant, shown, SEEDS)
    note = ""
    run = chosen.get(("no_validity_mask", 42))
    stats = (((run or {}).get("eval") or {}).get("splits") or {}).get("test", {}).get("unmasked_decoding")
    if stats and stats.get("positions"):
        note = f"invalid triples {stats['invalid_triples']}/{stats['positions']} ({100 * stats['invalid_triples'] / stats['positions']:.2f}% of positions)"
    for variant, shown in (("no_validity_mask", "− validity mask"), ("marginal_joint_kd", "Marginal + joint KD"), ("tone_only_kd", "Tone-only KD")):
        e3_row(variant, shown, (42,), ref_label="ti_dfd", note=note if variant == "no_validity_mask" else "")
    for tau in (1, 2, 4):
        e3_row(f"temperature_sweep_tau{tau}", f"Temperature sweep τ={tau}", (42,))
    text += ["## E3: ablations (test, all; difference = Full DFD − variant)", "",
             table(["Variant", "Seeds", "Word acc all", "Word acc heavy", "Tone ER", "Base ER", "d (per seed)", "Verdict", "Note"], rows), "",
             "One-seed rows compare with Full DFD seed 42 only and are indicative. The sweep reuses the DFD tuning trials "
             "(locked λ and tone weight), so its τ=" + f"{locked['dfd']['temperature']:g}" + " row is the selected DFD run.", ""]

    text += ["## Notes", "",
             "- Tone ER and base ER come from `dfd/metrics.py` at the commit that scored each run. If `metrics.py` changed after "
             "some runs were scored (decision 4), rerun `eval.py --replace` on those runs before comparing them.",
             "- `rare_combinations` has only 6 / 24 / 33 samples (val / test / unseen); do not conclude from it.",
             "- The near-duplicate pair `im0894` (train) and `im1670` (unseen test) belongs in the limitations.", ""]
    if missing:
        text += ["## Missing cells", ""] + [f"- {item}" for item in missing] + [""]
    if warnings:
        text += ["## Warnings", ""] + [f"- {item}" for item in warnings] + [""]
    if problems:
        text += ["## PROBLEMS (fix before using any number)", ""] + [f"- {item}" for item in problems] + [""]
    return {"text": "\n".join(text), "problems": problems, "warnings": warnings, "missing": missing, "used": used,
            "chosen": chosen}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--out", type=Path, default=ROOT / "results/summary")
    parser.add_argument("--locked", type=Path, default=ROOT / "configs/locked_hyperparameters.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "configs/data_reference.json")
    parser.add_argument("--exclude", action="append", default=[], metavar="RUN_ID", help="run folder name that must not count")
    parser.add_argument("--skip-tag-check", action="store_true", help="do not require exp-v2 on main runs (development only)")
    parser.add_argument("--strict", action="store_true", help="exit 1 also when cells are missing")
    args = parser.parse_args()
    if not args.locked.exists():
        raise SystemExit(f"{args.locked} not found: run `python scripts/e2_tuning.py lock` first")
    locked = json.loads(args.locked.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    skipped = []
    runs = load_runs(args.runs, skipped)
    result = summarize(runs, locked, reference, tags_for=git_tags, exclude=args.exclude, strict_tags=not args.skip_tag_check,
                       skipped=skipped)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(result["text"] + "\n", encoding="utf-8")
    with (args.out / "runs_used.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table", "label", "seed", "run_id", "stage", "commit_hash"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["used"])
    print(result["text"])
    print(f"Wrote {args.out / 'summary.md'} and {args.out / 'runs_used.csv'}")
    return 1 if result["problems"] or (args.strict and result["missing"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
