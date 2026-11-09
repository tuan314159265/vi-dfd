"""Fine-tune the official PARSeq-small with a 229-class Vietnamese head.

Permutation training and AdamW/one-cycle are inherited directly from pinned
upstream PARSeq (Darwin Bautista et al.; see THIRD_PARTY.md). Validation is
overridden to preserve unsupported/long labels and avoid ASCII transliteration.
"""

import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from dfd.parseq import PARSEQ_COMMIT, activate_upstream, initialize_teacher
from dfd.runtime import config_hash, prepare_run, seed_everything
from dfd.teacher_data import VietnameseLmdb, is_heavy
from dfd.vietchar import DecompositionTable

activate_upstream()
from strhub.models.parseq.system import PARSeq as UpstreamPARSeq  # noqa: E402


class VietnamesePARSeq(UpstreamPARSeq):
    def on_validation_epoch_start(self):
        self.val_stats = {"words": 0, "correct": 0, "heavy_words": 0, "heavy_correct": 0, "ned": 0.0}

    def validation_step(self, batch, batch_idx):
        images, labels, source_ids = batch
        predictions, _ = self.tokenizer.decode(self(images).softmax(-1))
        stats = self.val_stats
        for prediction, label in zip(predictions, labels):
            prediction, label = prediction.casefold(), label.casefold()
            correct = prediction == label
            heavy = is_heavy(label)
            stats["words"] += 1
            stats["correct"] += correct
            stats["heavy_words"] += heavy
            stats["heavy_correct"] += heavy and correct
            stats["ned"] += 1 - edit_distance(prediction, label) / max(len(prediction), len(label), 1)

    def on_validation_epoch_end(self):
        stats = self.val_stats
        if not stats["words"]:
            raise ValueError("Validation has no samples")
        self.log("validation_word_acc", stats["correct"] / stats["words"], prog_bar=True)
        self.log("validation_ned", stats["ned"] / stats["words"])
        if stats["heavy_words"]:
            self.log("validation_heavy_acc", stats["heavy_correct"] / stats["heavy_words"], prog_bar=True)
        if self.trainer.sanity_checking or not getattr(self, "artifact_dir", None):
            return
        path = self.artifact_dir / "validation.csv"
        row = {
            "epoch": self.current_epoch + 1, "global_step": self.global_step,
            "run_id": self.run_identity["run_id"], "commit_hash": self.run_identity["commit_hash"],
            "config_hash": self.run_identity["config_hash"], "seed": self.run_config["seed"],
            "num_words": stats["words"], "word_acc": stats["correct"] / stats["words"],
            "heavy_words": stats["heavy_words"],
            "heavy_word_acc": stats["heavy_correct"] / max(stats["heavy_words"], 1),
            "ned_score": stats["ned"] / stats["words"],
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 3),
        }
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nEpoch {row['epoch']}: validation={row['word_acc']:.4%}, heavy={row['heavy_word_acc']:.4%}, N={stats['words']}", flush=True)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log("train_ce", loss, on_step=False, on_epoch=True, batch_size=len(batch[1]))
        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint["permutation_rng_state"] = self.rng.bit_generator.state
        checkpoint["dfd_config"] = self.run_config
        checkpoint["dfd_initialization"] = self.initialization
        checkpoint["dfd_data_identity"] = self.data_identity
        checkpoint["dfd_run_identity"] = self.run_identity
        checkpoint["vocabulary_fingerprint"] = DecompositionTable.from_characters().fingerprint()
        checkpoint["parseq_commit"] = PARSEQ_COMMIT

    def on_load_checkpoint(self, checkpoint):
        self.rng.bit_generator.state = checkpoint["permutation_rng_state"]


def build_teacher(config, pretrained=True):
    table = DecompositionTable.from_characters()
    charset = "".join(table.characters[1:])
    training = config["training"]
    model = VietnamesePARSeq(
        charset_train=charset, charset_test=charset, max_label_length=25,
        batch_size=training["batch_size"], lr=training["lr"],
        warmup_pct=training.get("warmup_pct", 0.075), weight_decay=training.get("weight_decay", 0.0),
        img_size=(32, 128), patch_size=(4, 8), embed_dim=384,
        enc_num_heads=6, enc_mlp_ratio=4, enc_depth=12,
        dec_num_heads=12, dec_mlp_ratio=4, dec_depth=1,
        perm_num=6, perm_forward=True, perm_mirrored=True,
        decode_ar=True, refine_iters=1, dropout=0.1,
    )
    model.rng = np.random.default_rng(config["seed"])
    initialization = initialize_teacher(model, table, config["model"].get("english_checkpoint")) if pretrained else {"pretrained": False}
    return model, initialization


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def datasets(config):
    root = Path(config["data"]["root"])
    table = DecompositionTable.from_characters()
    train = VietnameseLmdb(root / "lmdb/train", table.characters[1:], training=True)
    validation = VietnameseLmdb(root / "lmdb/val", table.characters[1:])
    return train, validation


def data_identity(config):
    root = Path(config["data"]["root"])
    table = DecompositionTable.from_characters()
    vocab = json.loads((root / config["data"]["vocab_path"]).read_text(encoding="utf-8"))
    if vocab["fingerprint"] != table.fingerprint() or tuple(vocab["characters"]) != table.characters:
        raise ValueError("Frozen vocabulary differs from the model class order")
    freeze_path = root / config["data"]["split_manifest"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for path, expected in freeze["manifest_sha256"].items():
        actual = hashlib.sha256((root / "splits" / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen split/subset changed: {path}")
    preparation = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    return {"freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            "vocabulary_fingerprint": table.fingerprint(), "preparation": preparation}


def probe_batches(config, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this outside the device-restricted sandbox")
    seed_everything(config["seed"])
    model, initialization = build_teacher(config)
    model.cuda().train()
    model.log = lambda *args, **kwargs: None  # Standalone memory probe, not a Lightning training run.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    reports = []
    for batch_size in (16, 32, 64):
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            images = torch.randn(batch_size, 3, 32, 128, device="cuda")
            labels = ["ắ" * 25] * batch_size
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model.training_step((images, labels), 0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            reports.append({"batch_size": batch_size, "success": True, "peak_memory_mib": peak / 2**20,
                            "memory_fraction": peak / total, "seconds": time.perf_counter() - started,
                            "finite_loss": math.isfinite(loss.item())})
            del images, loss
        except torch.cuda.OutOfMemoryError:
            reports.append({"batch_size": batch_size, "success": False, "reason": "CUDA OOM"})
            break
        print(reports[-1], flush=True)
    candidates = [row["batch_size"] for row in reports if row["success"] and row["finite_loss"] and row["memory_fraction"] < 0.85]
    if not candidates:
        raise RuntimeError("No candidate meets the 15% VRAM headroom requirement")
    selected = max(candidates)
    payload = {"gpu": torch.cuda.get_device_name(0), "initialization": initialization,
               "reports": reports, "selected_micro_batch": selected, "effective_batch": 64,
               "accumulate_grad_batches": 64 // selected,
               "note": "Worst-length synthetic inputs; optimizer step included; not an accuracy experiment"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_teacher(config, repo, resume=None, smoke=False):
    if config["variant"] != "teacher" or config["model"]["backbone"] != "parseq_small":
        raise ValueError("This entry point currently supports only the E1 PARSeq-S teacher")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run outside the sandbox or use a GPU notebook")
    seed_everything(config["seed"])
    identity = data_identity(config)
    train, validation = datasets(config)
    expected = identity["preparation"]["word_crop_counts"]
    if train.audit["total"] != expected["train"] or len(validation) != expected["val"]:
        raise ValueError("LMDB counts differ from the frozen data preparation")
    model, initialization = build_teacher(config)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["dfd_config"] != config or saved["dfd_data_identity"] != identity:
            raise ValueError("Resume requires the exact original configuration and frozen data identity")
        if saved["parseq_commit"] != PARSEQ_COMMIT:
            raise ValueError("Resume uses a different upstream revision")
    folder = prepare_run(config, repo)
    model.run_config = config
    model.initialization = initialization
    model.data_identity = identity
    model.artifact_dir = folder
    model.run_identity = json.loads((folder / "metadata.json").read_text())
    model.started_at = time.perf_counter()
    for name, payload in (("initialization.json", initialization), ("data_identity.json", identity),
                          ("dataset_audit.json", {"train": train.audit, "validation": validation.audit})):
        (folder / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    workers = config["training"].get("num_workers", 4)
    generator = torch.Generator().manual_seed(config["seed"])
    kwargs = {"num_workers": workers, "pin_memory": True, "worker_init_fn": seed_worker, "persistent_workers": workers > 0}
    train_loader = DataLoader(train, batch_size=config["training"]["batch_size"], shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(validation, batch_size=config["training"]["batch_size"], shuffle=False, **kwargs)
    best = ModelCheckpoint(dirpath=folder / "checkpoints", filename="best-{epoch:02d}-{validation_word_acc:.4f}",
                           monitor="validation_word_acc", mode="max", save_top_k=1, save_last=True)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=config["training"]["epochs"],
        precision="16-mixed", accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        gradient_clip_val=20.0, callbacks=[best], logger=CSVLogger(str(folder), name="lightning"),
        deterministic=True, num_sanity_val_steps=2, log_every_n_steps=25,
        fast_dev_run=smoke, enable_progress_bar=False,
    )
    summary = {"status": "running", "smoke_test": smoke, "train_audit": train.audit,
               "validation_audit": validation.audit, "config_hash": config_hash(config)}
    (folder / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Run directory:", folder, flush=True)
    print("Dataset audit:", {"train": train.audit, "validation": validation.audit}, flush=True)
    try:
        metadata = json.loads((folder / "metadata.json").read_text())
        metadata["status"] = "running"
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        trainer.fit(model, train_loader, val_loader, ckpt_path=str(resume) if resume else None)
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    else:
        summary.update(status="smoke_completed" if smoke else "completed", best_checkpoint=best.best_model_path,
                       best_validation_word_acc=float(best.best_model_score) if best.best_model_score is not None else None,
                       global_steps=trainer.global_step, elapsed_seconds=time.perf_counter() - model.started_at)
    finally:
        (folder / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
        metadata = json.loads((folder / "metadata.json").read_text())
        metadata["status"] = summary["status"]
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (folder / "run.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata), lineterminator="\n")
            writer.writeheader()
            writer.writerow(metadata)
    print(json.dumps(summary, indent=2), flush=True)
    return folder
