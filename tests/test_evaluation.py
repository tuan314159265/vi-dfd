import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest
import yaml

lmdb = pytest.importorskip("lmdb")
pytest.importorskip("rapidfuzz")
Image = pytest.importorskip("PIL.Image")

from dfd import evaluation  # noqa: E402
from dfd.vietchar import DecompositionTable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_cli", ROOT / "eval.py")
eval_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_cli)

TABLE = DecompositionTable.from_characters()
LONG = "a" * 30
SAMPLES = [("1:0", "Việt"), ("1:1", "ĐẸP"), ("2:0", "25°C"), ("3:0", LONG), ("3:1", "hoà")]


def png(seed: int) -> bytes:
    image = Image.new("RGB", (100, 30), color=(seed * 40 % 255, 120, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "lmdb"
    (root / "val").mkdir(parents=True)
    env = lmdb.open(str(root / "val"), map_size=1 << 24)
    with env.begin(write=True) as txn:
        for index, (source_id, label) in enumerate(SAMPLES, start=1):
            suffix = f"{index:09d}".encode()
            txn.put(b"image-" + suffix, png(index))
            txn.put(b"label-" + suffix, label.encode("utf-8"))
            txn.put(b"source-id-" + suffix, source_id.encode("utf-8"))
        txn.put(b"num-samples", str(len(SAMPLES)).encode())
    env.close()
    subsets = tmp_path / "subsets"
    subsets.mkdir()
    for name, members in {"diacritic_heavy": [("1", "0"), ("1", "1")], "small_crops": [], "rare_combinations": [("3", "1")]}.items():
        with (subsets / f"val_{name}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["final_split", "image_id", "line_number", "transcript"])
            writer.writerows(["val", image_id, line, ""] for image_id, line in members)
    return root, subsets


@pytest.fixture
def run_dir(tmp_path):
    config = yaml.safe_load((ROOT / "configs/runs/E2/ti_dfd_s42.yaml").read_text(encoding="utf-8"))
    config["evaluation"]["splits"] = ["validation"]
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "resolved.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"run_id": "r1", "commit_hash": "c" * 40, "config_hash": "h", "seed": 42}))
    return folder


def write_predictions(folder: Path, predictions: dict[str, str]) -> Path:
    folder.mkdir(exist_ok=True)
    with (folder / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "prediction"])
        writer.writerows(predictions.items())
    return folder


def read_metrics(run_dir: Path) -> dict[str, dict]:
    with (run_dir / "metrics.csv").open(encoding="utf-8") as handle:
        return {row["subset"]: row for row in csv.DictReader(handle)}


def test_canonical_label_rules():
    assert evaluation.canonical_label("Việt Nam", TABLE, case_sensitive=False) == "việtnam"
    assert evaluation.canonical_label("25°C", TABLE, case_sensitive=True) == "25|C"
    assert evaluation.canonical_label("ĐẸP", TABLE, case_sensitive=False) == "đẹp"
    assert evaluation.canonical_label("hoà", TABLE, case_sensitive=True) == "hoà"  # NFC, tone position kept


def test_read_lmdb_keeps_every_sample(data):
    samples = evaluation.read_lmdb(data[0] / "val")
    assert [(sample.source_id, sample.label) for sample in samples] == SAMPLES


def test_subset_ids_must_exist(data):
    root, subsets = data
    with (subsets / "val_small_crops.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["val", "99", "0", ""])
    with pytest.raises(ValueError, match="not in the validation LMDB"):
        evaluation.read_subsets(subsets, "validation", ["small_crops"], [s for s, _ in SAMPLES])


def test_predictions_mode_end_to_end(data, run_dir, tmp_path):
    root, subsets = data
    predictions = write_predictions(tmp_path / "preds", {
        "1:0": "việt",   # case-insensitive match
        "1:1": "ĐEP",    # tone error on Ẹ
        "2:0": "25°C",   # label has a character outside the table: always wrong
        "3:0": LONG,     # long labels are kept and scored like any other
        "3:1": "hòa",    # tone mark position differs from the label: wrong
    })
    args = ["--run-dir", str(run_dir), "--predictions-dir", str(predictions), "--data-root", str(root), "--subset-dir", str(subsets)]
    assert eval_cli.main(args) == 0
    rows = read_metrics(run_dir)
    assert set(rows) == {"all", "diacritic_heavy", "rare_combinations"}  # empty small_crops writes no row
    assert rows["all"]["num_words"] == "5"
    assert float(rows["all"]["word_acc"]) == pytest.approx(2 / 5)
    assert float(rows["diacritic_heavy"]["word_acc"]) == pytest.approx(1 / 2)
    assert rows["all"]["split"] == "validation" and rows["all"]["commit_hash"] == "c" * 40
    assert float(rows["all"]["tone_er"]) > 0
    info = json.loads((run_dir / "eval_info.json").read_text(encoding="utf-8"))
    assert info["splits"]["validation"]["labels_with_unknown_characters"] == 1
    matrix = json.loads((run_dir / "tone_confusion" / "validation_all.json").read_text())
    assert matrix["dot"]["level"] == 1
    with (run_dir / "predictions" / "validation.csv").open(encoding="utf-8") as handle:
        correct = {row["source_id"]: row["correct"] for row in csv.DictReader(handle)}
    assert correct == {"1:0": "1", "1:1": "0", "2:0": "0", "3:0": "1", "3:1": "0"}

    # Rows are not silently overwritten.
    assert eval_cli.main(args) == 2
    assert eval_cli.main(args + ["--replace"]) == 0
    assert len(list(csv.DictReader((run_dir / "metrics.csv").open(encoding="utf-8")))) == 3


def test_missing_predictions_are_rejected(data, run_dir, tmp_path):
    root, subsets = data
    predictions = write_predictions(tmp_path / "preds", {"1:0": "việt"})
    args = ["--run-dir", str(run_dir), "--predictions-dir", str(predictions), "--data-root", str(root), "--subset-dir", str(subsets)]
    assert eval_cli.main(args) == 2
    assert not (run_dir / "metrics.csv").exists()


def test_checkpoint_mode_with_factorized_student(data, run_dir, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("strhub.models.parseq.system")
    from dfd.parseq_kd import DistilledPARSeq

    torch.manual_seed(0)
    charset = "".join(TABLE.characters[1:])
    system = DistilledPARSeq(
        charset_train=charset, charset_test=charset, max_label_length=25, batch_size=4, lr=1e-3,
        warmup_pct=0.1, weight_decay=0.0, img_size=[32, 128], patch_size=[4, 8], embed_dim=32,
        enc_num_heads=2, enc_mlp_ratio=2, enc_depth=1, dec_num_heads=2, dec_mlp_ratio=2, dec_depth=1,
        perm_num=6, perm_forward=True, perm_mirrored=True, decode_ar=True, refine_iters=1, dropout=0.1,
    )
    system.use_factorized_head(TABLE)
    checkpoint = tmp_path / "student.ckpt"
    torch.save({"hyper_parameters": dict(system.hparams), "state_dict": system.state_dict()}, checkpoint)

    loaded = evaluation.load_parseq_checkpoint(str(checkpoint), TABLE)
    samples = evaluation.read_lmdb(data[0] / "val")
    expected = evaluation.predict_parseq(system.eval(), samples, batch_size=2)
    assert evaluation.predict_parseq(loaded, samples, batch_size=3) == expected

    root, subsets = data
    args = ["--run-dir", str(run_dir), "--checkpoint", str(checkpoint), "--data-root", str(root), "--subset-dir", str(subsets)]
    assert eval_cli.main(args) == 0
    rows = read_metrics(run_dir)
    assert int(rows["all"]["params"]) == sum(parameter.numel() for parameter in system.parameters())
    info = json.loads((run_dir / "eval_info.json").read_text(encoding="utf-8"))
    assert info["checkpoint_sha256"] and info["vocabulary_fingerprint"] == TABLE.fingerprint()
