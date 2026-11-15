"""Factorized PARSeq-Ti output head with valid-character composition.

The factorized-head concept is inherited from VietTDR. Encoder, decoder and
composed-character context embeddings stay unchanged; see proposal note 05.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dfd.vietchar import COMPONENTS, MODIFIERS, TONES, DecompositionTable


class ComponentLogits(NamedTuple):
    base: Tensor
    modifier: Tensor
    tone: Tensor


class FactorizedHead(nn.Linear):
    """Drop-in replacement for PARSeq's `model.head`.

    The base, modifier and tone classifiers are consecutive output blocks of one
    linear layer, each with its own softmax. It subclasses `nn.Linear` because
    upstream PARSeq finds its device from `head.parameters(recurse=False)`, which
    is empty for a wrapper around three child modules.
    """

    def __init__(self, hidden_size: int, table: DecompositionTable):
        sizes = (len(table.bases), len(MODIFIERS), len(TONES))
        super().__init__(hidden_size, sum(sizes))
        self.sizes = sizes
        # Same initialization as upstream's init_weights, which has already run when the head is replaced.
        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)
        for name in COMPONENTS:
            self.register_buffer(f"{name}_ids", torch.tensor(table.component_ids(name), dtype=torch.long))

    def component_logits(self, hidden: Tensor) -> ComponentLogits:
        """Return unnormalized scores for each component; used by CE and KD."""
        return ComponentLogits(*super().forward(hidden).split(self.sizes, dim=-1))

    def compose(self, logits: ComponentLogits) -> Tensor:
        """Return normalized log-probabilities over the valid characters only."""
        log_probabilities = (
            logits.base.log_softmax(-1)[..., self.base_ids]
            + logits.modifier.log_softmax(-1)[..., self.modifier_ids]
            + logits.tone.log_softmax(-1)[..., self.tone_ids]
        )
        return log_probabilities - log_probabilities.logsumexp(-1, keepdim=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.compose(self.component_logits(hidden))

    def component_loss(self, logits: ComponentLogits, targets: Tensor, pad_id: int) -> dict[str, Tensor]:
        """Cross-entropy per component over non-padding positions, plus their sum as `total`."""
        valid = targets != pad_id
        ids = targets[valid]
        losses = {
            name: F.cross_entropy(getattr(logits, name)[valid], getattr(self, f"{name}_ids")[ids])
            for name in COMPONENTS
        }
        losses["total"] = sum(losses.values())
        return losses
