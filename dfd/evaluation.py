"""Shared evaluation pipeline: LMDB split -> predictions -> metrics rows.

Every model is scored by the same code on the same denominators:

- Every LMDB sample is kept. Unlike PARSeq's LmdbDataset, nothing is filtered:
  labels longer than the model's maximum length and labels with characters
  outside the 229-class table stay in, and are counted as errors.
- Labels and predictions: NFC, spaces removed, lowercased when the protocol
  is case-insensitive. A label character outside the table is replaced by the
  metrics module's invalid marker, so the whole crop counts as wrong.
- Subsets are joined by `source-id` (`image_id:line_number`).

Metrics come from `dfd.metrics` (Task 10). Models other than PARSeq (VietOCR,
CRNN, ONNX/INT8 exports) are scored from a predictions CSV with the same code.
"""

import csv
import hashlib
import io
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dfd import metrics
from dfd.vietchar import DecompositionTable


SPLIT_DIRS = {"validation": "val", "test": "test", "unseen_test": "unseen_test"}
SUBSETS = ("all", "diacritic_heavy", "small_crops", "rare_combinations")


@dataclass(frozen=True)
class Sample:
    source_id: str
    label: str
    image: bytes


def split_dir(split: str) -> str:
    if split not in SPLIT_DIRS:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(SPLIT_DIRS)}")
    return SPLIT_DIRS[split]


def read_lmdb(path: Path) -> list[Sample]:
    """Read every sample of a split LMDB written by notebook 04, in index order."""
    import lmdb

    if not (Path(path) / "data.mdb").is_file():
        raise FileNotFoundError(f"No LMDB at {path}")
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)
    try:
        with env.begin() as txn:
            count = int(txn.get(b"num-samples").decode("ascii"))
            samples = []
            for index in range(1, count + 1):
                suffix = f"{index:09d}".encode("ascii")
                source_id = txn.get(b"source-id-" + suffix)
                if source_id is None:
                    raise ValueError(f"{path}: sample {index} has no source-id; rebuild the LMDB with notebook 04")
                samples.append(Sample(
                    source_id.decode("utf-8"),
                    unicodedata.normalize("NFC", txn.get(b"label-" + suffix).decode("utf-8")),
                    bytes(txn.get(b"image-" + suffix)),
                ))
    finally:
        env.close()
    ids = [sample.source_id for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate source-id values")
    return samples


def lmdb_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with (Path(path) / "data.mdb").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(row: dict) -> str:
    line = str(row["line_number"]).strip()
    if line.endswith(".0") and line[:-2].isdigit():
        line = line[:-2]
    return f"{int(float(row['image_id']))}:{line}"


def read_subsets(subset_dir: Path, split: str, names: list[str], source_ids: list[str]) -> dict[str, set[str]]:
    """Map each subset name to its source-ids; `all` is the whole split."""
    known = set(source_ids)
    result = {}
    for name in names:
        if name == "all":
            result[name] = known
            continue
        path = Path(subset_dir) / f"{split_dir(split)}_{name}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"image_id", "line_number"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} lacks columns {sorted(missing)} needed to build source-id")
            members = {_source_id(row) for row in reader}
        unknown = members - known
        if unknown:
            raise ValueError(f"{path}: {len(unknown)} source-ids are not in the {split} LMDB, e.g. {sorted(unknown)[:3]}")
        result[name] = members
    return result


def canonical_label(text: str, table: DecompositionTable, case_sensitive: bool) -> str:
    known = set(table.characters[1:])
    text = unicodedata.normalize("NFC", text).replace(" ", "")
    text = "".join(character if character in known else metrics.INVALID_CHARACTER for character in text)
    return text if case_sensitive else text.lower()


def canonical_prediction(text: str, case_sensitive: bool) -> str:
    text = unicodedata.normalize("NFC", text).replace(" ", "")
    return text if case_sensitive else text.lower()


def insertions_deletions(predictions: list[str], labels: list[str]) -> tuple[int, int]:
    """Total inserted and deleted characters over crops without the invalid marker."""
    inserted = deleted = 0
    for prediction, label in zip(predictions, labels):
        if metrics.contains_invalid_character(prediction) or metrics.contains_invalid_character(label):
            continue
        for operation, _, _ in metrics.levenshtein_alignment(prediction, label):
            inserted += operation == "insert"
            deleted += operation == "delete"
    return inserted, deleted


def score(records: list[tuple[str, str, str]], subsets: dict[str, set[str]]) -> tuple[list[dict], dict]:
    """Score (source_id, canonical_label, canonical_prediction) records per subset.

    Returns metric rows (dfd.runtime.METRIC_FIELDS names) and tone confusion matrices.
    """
    rows, confusion = [], {}
    for name, members in subsets.items():
        chosen = [record for record in records if record[0] in members]
        if not chosen:
            print(f"Subset {name!r} is empty; no row written")
            continue
        labels = [record[1] for record in chosen]
        predictions = [record[2] for record in chosen]
        rates = metrics.component_error_rates(predictions, labels)
        inserted, deleted = insertions_deletions(predictions, labels)
        rows.append({
            "subset": name, "num_words": len(chosen),
            "word_acc": metrics.word_accuracy(predictions, labels),
            "ned_score": metrics.one_minus_ned(predictions, labels),
            "base_er": rates["base_er"], "modifier_er": rates["modifier_er"], "tone_er": rates["tone_er"],
            "insertions": inserted, "deletions": deleted,
        })
        confusion[name] = metrics.tone_confusion_matrix(predictions, labels)
    return rows, confusion


def load_parseq_checkpoint(path: str, table: DecompositionTable, device: str = "cpu"):
    """Load a PARSeq Lightning checkpoint with either head; returns the system in eval mode."""
    import torch

    from dfd.heads import FactorizedHead
    from dfd.parseq_kd import DistilledPARSeq

    if DistilledPARSeq is None:
        raise ImportError("strhub (PARSeq) is required to evaluate a checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = dict(checkpoint["hyper_parameters"])
    if hparams.get("charset_test") != "".join(table.characters[1:]):
        raise ValueError("Checkpoint charset_test is not the 228-character table order")
    system = DistilledPARSeq(**hparams)
    state = checkpoint["state_dict"]
    factorized_size = sum(FactorizedHead(1, table).sizes)
    if state["model.head.weight"].shape[0] == factorized_size:
        system.use_factorized_head(table)
    system.load_state_dict(state, strict=True)
    return system.eval().to(device)


def predict_parseq(system, samples: list[Sample], batch_size: int = 256, device: str = "cpu",
                   validity_mask: bool = True, stats: dict | None = None) -> list[str]:
    """Greedy PARSeq decoding (the model's own AR + refinement settings), no length filtering.

    `validity_mask=False` is the E3 `no_validity_mask` ablation and needs a factorized head.
    Decoding runs exactly as for full DFD (same AR contexts, same refinement), and at every
    output position up to and including EOS the independent argmax triple
    (base, modifier, tone) is also taken from the final component logits. Where that triple
    is a valid character it equals the masked prediction (the masked argmax is the maximum of
    the same product restricted to valid triples), so the output is unchanged. Where it is not
    a character, the position becomes `metrics.INVALID_CHARACTER`, so the crop counts as wrong.
    The word-accuracy difference to full DFD is therefore exactly the effect of the mask.
    `stats` (if given) receives the number of positions and of invalid triples.
    """
    import torch
    from PIL import Image
    from strhub.data.module import SceneTextDataModule

    head = system.model.head
    captured = []
    hook = None
    if not validity_mask:
        from dfd.heads import FactorizedHead
        if not isinstance(head, FactorizedHead):
            raise ValueError("validity_mask: false needs a checkpoint with the factorized head")
        sizes = head.sizes
        lookup = torch.full(sizes, -1, dtype=torch.long)
        lookup[head.base_ids.cpu(), head.modifier_ids.cpu(), head.tone_ids.cpu()] = torch.arange(len(head.base_ids))
        hook = head.register_forward_hook(lambda module, inputs, output: captured.append(inputs[0]))
    transform = SceneTextDataModule.get_transform(tuple(system.hparams.img_size))
    predictions = []
    positions = invalid = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                images = torch.stack([transform(Image.open(io.BytesIO(sample.image)).convert("RGB")) for sample in batch])
                captured.clear()
                logits = system(images.to(device))
                decoded, _ = system.tokenizer.decode(logits.softmax(-1))
                if validity_mask:
                    predictions.extend(decoded)
                    continue
                # Last head call = refinement pass over the whole sequence; without refinement,
                # the AR loop calls the head once per position.
                hidden = captured[-1] if captured[-1].shape[1] == logits.shape[1] else torch.cat(captured, dim=1)
                parts = head.component_logits(hidden)
                classes = lookup[parts.base.argmax(-1).cpu(), parts.modifier.argmax(-1).cpu(), parts.tone.argmax(-1).cpu()]
                for row, text in enumerate(decoded):
                    length = min(len(text) + 1, classes.shape[1])  # characters plus the EOS position
                    bad = (classes[row, :length] < 0).tolist()
                    positions += length
                    invalid += sum(bad)
                    chars = [metrics.INVALID_CHARACTER if bad[i] else char for i, char in enumerate(text)]
                    if length > len(text) and bad[len(text)]:
                        chars.append(metrics.INVALID_CHARACTER)
                    predictions.append("".join(chars))
    finally:
        if hook is not None:
            hook.remove()
    if stats is not None and not validity_mask:
        stats.update(positions=positions, invalid_triples=invalid)
    return predictions


def read_predictions(path: Path) -> dict[str, str]:
    """Read a `source_id,prediction` CSV produced by another model's inference."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"source_id", "prediction"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path} needs columns source_id,prediction")
        result = {}
        for row in reader:
            if row["source_id"] in result:
                raise ValueError(f"{path}: duplicate source_id {row['source_id']}")
            result[row["source_id"]] = row["prediction"]
    return result
