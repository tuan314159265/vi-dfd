"""Evaluate a model on the frozen VinText splits and append rows to its run's metrics.csv.

PARSeq checkpoint (teacher, ti_ce, ti_kd, ti_factorized, ti_dfd, E3 variants):

    python eval.py --run-dir results/runs/<run_id> --checkpoint best.ckpt \
        --data-root <processed>/lmdb --subset-dir <processed>/splits/subsets

Any other model (VietOCR, CRNN, ONNX/INT8): write <dir>/<split>.csv with columns
source_id,prediction for every sample of the split, then:

    python eval.py --run-dir results/runs/<run_id> --predictions-dir <dir> ...

Without --run-dir, --config is required and a new run folder is prepared
(the Git tree must be clean, as for training). Splits and subsets come from the
config's `evaluation` section unless --splits/--subsets are given.

Outputs inside the run folder: metrics.csv rows (one per split x subset),
predictions/<split>.csv, tone_confusion/<split>_<subset>.json, eval_info.json.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dfd.config import load_config, validate_config  # noqa: E402
from dfd.runtime import METRIC_FIELDS, prepare_run  # noqa: E402
from dfd.vietchar import DecompositionTable  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="Run YAML; not needed with --run-dir (uses its resolved.yaml)")
    parser.add_argument("--run-dir", type=Path, help="Existing run folder from train.py --prepare-run")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="PARSeq Lightning checkpoint")
    source.add_argument("--predictions-dir", type=Path, help="Folder of <split>.csv with source_id,prediction")
    parser.add_argument("--data-root", type=Path, help="LMDB root with val/ test/ unseen_test/ (default: data.root)")
    parser.add_argument("--subset-dir", type=Path, help="Frozen subset CSV folder (default: data.subset_manifest)")
    parser.add_argument("--splits", nargs="+", help="Default: evaluation.splits of the config")
    parser.add_argument("--subsets", nargs="+", help="Default: all + evaluation.subsets of the config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replace", action="store_true", help="Replace existing rows for the same split/subset/precision")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.checkpoint:
        # train.py gets strhub on sys.path through dfd.teacher; eval.py must do it itself, before
        # dfd.parseq_kd is first imported, or DistilledPARSeq stays None (every Colab eval failed on 19/11).
        from dfd.parseq import activate_upstream

        activate_upstream()
    from dfd import evaluation  # imported late so --help works without optional deps

    try:
        if args.run_dir:
            run_dir = args.run_dir
            config = yaml.safe_load((run_dir / "resolved.yaml").read_text(encoding="utf-8"))
            validate_config(config)
        elif args.config:
            config = load_config(args.config)
            run_dir = prepare_run(config, ROOT)
        else:
            raise ValueError("Give --run-dir or --config")
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        settings = config.get("evaluation") or {}
        precision = settings.get("precision", "fp32")
        if args.checkpoint and precision != "fp32":
            raise ValueError(f"precision {precision!r}: score exported/quantized models with --predictions-dir")
        case_sensitive = bool(settings.get("case_sensitive", False))
        validity_mask = (config.get("composition") or {}).get("validity_mask", True) is not False
        splits = args.splits or settings.get("splits") or []
        subsets = args.subsets or list(dict.fromkeys(["all", *(settings.get("subsets") or [])]))
        data_root, subset_dir = args.data_root, args.subset_dir
        root = Path(config["data"]["root"]) if config["data"].get("root") else None
        if data_root is None and root is not None:
            # data.root is the processed root used for training (lmdb/, splits/); a bare LMDB root also works.
            data_root = root / "lmdb" if (root / "lmdb").is_dir() else root
        if subset_dir is None and config["data"].get("subset_manifest"):
            subset_dir = Path(config["data"]["subset_manifest"])
            if not subset_dir.is_absolute() and root is not None:
                subset_dir = root / subset_dir
        if data_root is None:
            raise ValueError("No LMDB root: pass --data-root or set data.root")
        if subset_dir is None and set(subsets) - {"all"}:
            raise ValueError("No subset folder: pass --subset-dir or set data.subset_manifest")
        if not splits:
            raise ValueError("No splits to evaluate")

        table = DecompositionTable.from_characters()
        invalid = evaluation.metrics.INVALID_CHARACTER
        system = params = None
        if args.checkpoint:
            system = evaluation.load_parseq_checkpoint(str(args.checkpoint), table, args.device)
            params = sum(parameter.numel() for parameter in system.parameters())

        metrics_path = run_dir / "metrics.csv"
        rows = existing_rows(metrics_path)
        info = {
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(args.checkpoint or args.predictions_dir),
            "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
            "vocabulary_fingerprint": table.fingerprint(),
            "case_sensitive": case_sensitive, "precision": precision, "validity_mask": validity_mask, "splits": {},
        }
        for split in splits:
            lmdb_path = data_root / evaluation.split_dir(split)
            samples = evaluation.read_lmdb(lmdb_path)
            ids = [sample.source_id for sample in samples]
            members = evaluation.read_subsets(subset_dir, split, subsets, ids)
            if system is not None:
                decode_stats = {}
                raw = evaluation.predict_parseq(system, samples, args.batch_size, args.device,
                                                validity_mask=validity_mask, stats=decode_stats)
            else:
                given = evaluation.read_predictions(args.predictions_dir / f"{split}.csv")
                missing = set(ids) - set(given)
                if missing:
                    raise ValueError(f"{split}: {len(missing)} samples have no prediction, e.g. {sorted(missing)[:3]}")
                extra = set(given) - set(ids)
                if extra:
                    raise ValueError(f"{split}: {len(extra)} predictions are not in the LMDB, e.g. {sorted(extra)[:3]}")
                raw = [given[source_id] for source_id in ids]
            records = [
                (sample.source_id, evaluation.canonical_label(sample.label, table, case_sensitive),
                 evaluation.canonical_prediction(prediction, case_sensitive))
                for sample, prediction in zip(samples, raw)
            ]
            new_rows, confusion = evaluation.score(records, members)

            if any(row["split"] == split and row["subset"] in {r["subset"] for r in new_rows} and row["precision"] == precision for row in rows):
                if not args.replace:
                    raise ValueError(f"metrics.csv already has {split} rows at {precision}; pass --replace to overwrite")
                rows = [row for row in rows if not (row["split"] == split and row["precision"] == precision)]
            for row in new_rows:
                rows.append({
                    field: "" for field in METRIC_FIELDS
                } | {
                    "run_id": metadata["run_id"], "commit_hash": metadata["commit_hash"],
                    "config_hash": metadata["config_hash"], "seed": metadata["seed"], "split": split,
                    "precision": precision, "params": params if params is not None else "",
                } | {key: (f"{value:.6f}" if isinstance(value, float) else value) for key, value in row.items()})

            (run_dir / "predictions").mkdir(exist_ok=True)
            with (run_dir / "predictions" / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["source_id", "label", "prediction", "raw_prediction", "correct"])
                for (source_id, label, prediction), raw_prediction in zip(records, raw):
                    writer.writerow([source_id, label, prediction, raw_prediction, int(label == prediction and invalid not in label)])
            (run_dir / "tone_confusion").mkdir(exist_ok=True)
            for name, matrix in confusion.items():
                (run_dir / "tone_confusion" / f"{split}_{name}.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
            info["splits"][split] = {
                "lmdb": str(lmdb_path), "lmdb_sha256": evaluation.lmdb_fingerprint(lmdb_path),
                "num_samples": len(samples),
                "labels_with_unknown_characters": sum(invalid in record[1] for record in records),
            }
            if system is not None and not validity_mask:
                info["splits"][split]["unmasked_decoding"] = decode_stats
            for row in new_rows:
                print(f"{split:<12} {row['subset']:<18} n={row['num_words']:<6} word_acc={row['word_acc']:.4f} "
                      f"1-NED={row['ned_score']:.4f} tone_er={row['tone_er']:.4f}")
        write_rows(metrics_path, rows)
        (run_dir / "eval_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {metrics_path}")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"eval.py: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
