"""Pinned upstream PARSeq integration and English-to-Vietnamese initialization."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path


PARSEQ_COMMIT = "1902db043c029a7e03a3818c616c06600af574be"
PRETRAINED_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt"
PRETRAINED_TINY_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt"


def activate_upstream(root=None):
    root = Path(root or os.environ.get("PARSEQ_ROOT", Path.home() / ".cache/vi-dfd/parseq")).expanduser().resolve()
    if not (root / "strhub/models/parseq/system.py").is_file():
        raise FileNotFoundError(f"Clone https://github.com/baudm/parseq into {root} and checkout {PARSEQ_COMMIT}")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != PARSEQ_COMMIT:
        raise ValueError(f"PARSeq must be pinned to {PARSEQ_COMMIT}; found {revision}")
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--name-only"], text=True).strip():
        raise ValueError("Upstream PARSeq has uncommitted source changes")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def initialize_teacher(system, table, checkpoint=None, url=PRETRAINED_URL):
    """Load every compatible English weight, retain shared input embeddings, reset head.

    Official `parseq` is the 384-wide small model, called PARSeq-S in our proposal;
    students pass `url=PRETRAINED_TINY_URL` for the 192-wide `parseq-tiny`.
    `head.*` is intentionally new. English context embeddings including EOS/BOS/PAD
    are copied by token identity instead of discarding all embedding knowledge.
    """
    import torch
    from strhub.data.utils import Tokenizer
    from dfd.vietchar import ASCII_CHARSET

    if checkpoint is None:
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True, weights_only=True)
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    else:
        checkpoint = Path(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    original = Tokenizer(ASCII_CHARSET)
    width = system.model.head.in_features
    if state["head.weight"].shape != (95, width) or state["text_embed.embedding.weight"].shape != (97, width):
        raise ValueError(f"Expected official 94-character English PARSeq weights of width {width}")
    filtered = {key: value for key, value in state.items() if not key.startswith("head.")}
    embedding = system.model.text_embed.embedding.weight.detach().clone()
    copied = 0
    for token, old_id in original._stoi.items():
        if token in system.tokenizer._stoi:
            embedding[system.tokenizer._stoi[token]] = state["text_embed.embedding.weight"][old_id]
            copied += 1
    filtered["text_embed.embedding.weight"] = embedding
    loaded = system.model.load_state_dict(filtered, strict=False)
    if set(loaded.missing_keys) != {"head.weight", "head.bias"} or loaded.unexpected_keys:
        raise ValueError(f"Unexpected pretrained mismatch: {loaded}")
    if system.model.head.out_features != len(table) or len(system.tokenizer) != len(table) + 2:
        raise ValueError("Tokenizer/head must agree with the 229-class vocabulary plus BOS/PAD")
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    return {"url": url, "checkpoint_sha256": digest, "copied_context_embeddings": copied,
            "loaded_model_keys": len(filtered), "reset_keys": loaded.missing_keys,
            "output_classes": system.model.head.out_features, "tokenizer_tokens": len(system.tokenizer)}
