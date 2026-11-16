"""Train the PARSeq-Ti students of E2/E3: CE, factorized head, standard KD and DFD.

Students reuse the teacher pipeline (`dfd/teacher.py`): the same LMDB loading and
augmentation, budget, full-denominator validation metric and checkpointing. They differ
from each other only in the output head and the loss, which come from the run YAML.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dfd.parseq import PRETRAINED_TINY_URL, initialize_teacher
from dfd.teacher import VietnamesePARSeq, datasets, fit_parseq
from dfd.runtime import seed_everything
from dfd.distill import DistillationObjective
from dfd.parseq_kd import DistilledPARSeq, check_compatible, load_frozen_teacher
from dfd.vietchar import DecompositionTable
from dfd import evaluation

TINY = dict(embed_dim=192, enc_num_heads=3, dec_num_heads=6)


class StudentPARSeq(VietnamesePARSeq, DistilledPARSeq):
    """Teacher-style validation/checkpoint hooks on top of the KD training step.

    MRO: StudentPARSeq -> VietnamesePARSeq -> DistilledPARSeq -> PARSeq, so
    `VietnamesePARSeq.training_step` calls `DistilledPARSeq.training_step`.
    """


def build_student(config, table=None, pretrained=True, size=TINY):
    table = table or DecompositionTable.from_characters()
    charset = "".join(table.characters[1:])
    training = config["training"]
    model = StudentPARSeq(
        charset_train=charset, charset_test=charset, max_label_length=25,
        batch_size=training["batch_size"], lr=training["lr"],
        warmup_pct=training.get("warmup_pct", 0.075), weight_decay=training.get("weight_decay", 0.0),
        img_size=(32, 128), patch_size=(4, 8), enc_mlp_ratio=4, enc_depth=12,
        dec_mlp_ratio=4, dec_depth=1, perm_num=6, perm_forward=True, perm_mirrored=True,
        decode_ar=True, refine_iters=1, dropout=0.1, **size,
    )
    model.rng = np.random.default_rng(config["seed"])
    if pretrained:
        initialization = initialize_teacher(model, table, config["model"].get("english_checkpoint"), url=PRETRAINED_TINY_URL)
    else:
        initialization = {"pretrained": False}
    if config["model"]["head"] == "factorized":
        model.use_factorized_head(table)  # replaces the freshly reset 229-class head
        initialization["head"] = "factorized"
    return model, initialization


def attach_teacher(model, config, table):
    objective = DistillationObjective.from_config(config["distillation"])
    teacher = None
    if objective.enabled:
        teacher = load_frozen_teacher(config["model"]["teacher_checkpoint"], table)
        check_compatible(model, teacher, 25)
    model.configure_distillation(objective, teacher, table)
    return objective, teacher


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def student_data_identity(config, teacher_checkpoint=None):
    """Frozen-data identity without `preparation.json`, so Tiến's original export also works.

    Records the freeze manifest hash and LMDB file hashes. When a teacher is used, its own
    recorded data identity is copied next to them so a different data build is visible.
    """
    root = Path(config["data"]["root"])
    table = DecompositionTable.from_characters()
    vocab_path = root / config["data"]["vocab_path"] if config["data"].get("vocab_path") else None
    if vocab_path and vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        if vocab["fingerprint"] != table.fingerprint() or tuple(vocab["characters"]) != table.characters:
            raise ValueError("Frozen vocabulary differs from the model class order")
    freeze_path = root / config["data"]["split_manifest"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for path, expected in freeze["manifest_sha256"].items():
        if _sha256(root / "splits" / path) != expected:
            raise ValueError(f"Frozen split/subset changed: {path}")
    identity = {
        "freeze_sha256": _sha256(freeze_path),
        "vocabulary_fingerprint": table.fingerprint(),
        "word_crop_counts": freeze["word_crop_counts"],
        "lmdb_sha256": {split: evaluation.lmdb_fingerprint(root / "lmdb" / split) for split in ("train", "val")},
    }
    if (root / "preparation.json").exists():
        identity["preparation"] = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    if teacher_checkpoint:
        saved = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        teacher_identity = saved.get("dfd_data_identity") or {}
        identity["teacher_checkpoint_sha256"] = _sha256(teacher_checkpoint)
        identity["teacher_data_identity"] = teacher_identity
        identity["teacher_same_freeze_manifest"] = teacher_identity.get("freeze_sha256") == identity["freeze_sha256"]
    return identity


def run_student(config, repo, resume=None, smoke=False):
    if config["model"]["backbone"] != "parseq_tiny" or config.get("stage", "train") not in ("train", "tune"):
        raise ValueError("This entry point trains PARSeq-Ti students only (stage train or tune)")
    if float(config["data"].get("training_fraction", 1.0)) != 1.0:
        raise ValueError("training_fraction < 1 (E5) needs an agreed subsampling rule; not implemented")
    if (config.get("composition") or {}).get("validity_mask") is False:
        raise ValueError("no_validity_mask trains exactly like full DFD (see docs/PROGRESS_VINH.md); "
                         "reuse the ti_dfd checkpoint of the same seed and change only the decoding")
    seed_everything(config["seed"])
    table = DecompositionTable.from_characters()
    teacher_checkpoint = config["model"].get("teacher_checkpoint") if config["distillation"].get("enabled") else None
    identity = student_data_identity(config, teacher_checkpoint)
    if teacher_checkpoint and not identity["teacher_same_freeze_manifest"]:
        print("WARNING: the teacher was trained on a different freeze_manifest.json; "
              "confirm the split assignments match before reporting results", flush=True)
    train, validation = datasets(config)
    expected = identity["word_crop_counts"]
    if train.audit["total"] != expected["train"] or len(validation) != expected["val"]:
        raise ValueError("LMDB counts differ from the frozen data preparation")
    model, initialization = build_student(config, table)
    objective, teacher = attach_teacher(model, config, table)
    initialization["distillation"] = {"enabled": objective.enabled, "mode": config["distillation"].get("mode")}
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["dfd_config"] != config or saved["dfd_data_identity"] != identity:
            raise ValueError("Resume requires the exact original configuration and frozen data identity")
    return fit_parseq(config, repo, model, initialization, identity, train, validation, resume, smoke)
