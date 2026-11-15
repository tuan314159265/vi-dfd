"""PARSeq training step with a frozen teacher on shared permutations (proposal note 06).

Teacher and student receive the same augmented images, the same sampled
permutations, the same ground-truth context and the same attention/padding
masks, so the KL compares the same position under the same context. The
teacher runs without gradients in eval mode.

The loop mirrors upstream `strhub.models.parseq.system.PARSeq.training_step`
at commit 1902db0: every pass is weighted by its number of valid targets, and
EOS targets are dropped after the canonical and reverse permutations.

Usage with upstream PARSeq (requires `strhub` on the path):

    student = DistilledPARSeq(**student_hparams)          # same args as strhub PARSeq
    student.use_factorized_head(table)                    # ti_factorized / ti_dfd and E3
    student.configure_distillation(objective, teacher)    # teacher from load_frozen_teacher
    trainer.fit(student, datamodule)
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from dfd.distill import DistillationObjective, distillation_loss, marginal_matrices
from dfd.heads import FactorizedHead
from dfd.vietchar import DecompositionTable


def freeze(system: nn.Module) -> nn.Module:
    """Put a teacher in eval mode and stop its gradients."""
    system.eval()
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    return system


def check_compatible(student, teacher, max_label_length: int) -> None:
    """Teacher and student must share token ids and cover the target length."""
    if list(student.tokenizer._itos) != list(teacher.tokenizer._itos):
        raise ValueError("Teacher and student tokenizers differ; both must use the 229-class table order")
    if teacher.model.max_label_length < max_label_length:
        raise ValueError("Teacher max_label_length is shorter than the student's targets")
    if isinstance(teacher.model.head, FactorizedHead):
        raise ValueError("The teacher must keep its ordinary 229-way head")


def shared_permutation_loss(
    student,
    teacher,
    images: Tensor,
    labels: list[str],
    objective: DistillationObjective,
    matrices: dict[str, Tensor] | None = None,
    permutations: Callable[[Tensor], Tensor] | None = None,
) -> dict[str, Tensor]:
    """Return the PARSeq permutation-language-modeling loss with optional KD.

    `student` and `teacher` are strhub PARSeq systems (or objects exposing
    `model`, `tokenizer`, `pad_id`, `eos_id`, `gen_tgt_perms`, `generate_attn_masks`).
    `teacher` may be None when KD is disabled (ti_ce, ti_factorized).
    """
    if objective.enabled and teacher is None:
        raise ValueError("KD is enabled but no teacher was attached")
    head = student.model.head
    factorized = isinstance(head, FactorizedHead)
    if objective.needs_factorized_student and not factorized:
        raise ValueError(f"mode {objective.mode!r} needs the factorized student head")

    tgt = student.tokenizer.encode(labels, images.device)
    tgt_perms = (permutations or student.gen_tgt_perms)(tgt)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_padding_mask = (tgt_in == student.pad_id) | (tgt_in == student.eos_id)

    memory = student.model.encode(images)
    if objective.enabled:
        check_compatible(student, teacher, tgt.shape[1] - 2)
        with torch.no_grad():
            teacher_memory = teacher.model.encode(images)

    totals: dict[str, Tensor] = {}
    numel = 0
    n = int((tgt_out != student.pad_id).sum())
    for index, perm in enumerate(tgt_perms):
        tgt_mask, query_mask = student.generate_attn_masks(perm)
        hidden = student.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        output = head.component_logits(hidden) if factorized else head(hidden)
        teacher_logits = None
        if objective.enabled:
            with torch.no_grad():
                teacher_hidden = teacher.model.decode(
                    tgt_in, teacher_memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask
                )
                teacher_logits = teacher.model.head(teacher_hidden)
        parts = distillation_loss(output, teacher_logits, tgt_out, student.pad_id, objective, matrices, head if factorized else None)
        for name, value in parts.items():
            totals[name] = totals.get(name, 0) + n * value
        numel += n
        # Same as upstream: after the canonical and reverse orderings, drop EOS targets.
        if index == 1:
            tgt_out = torch.where(tgt_out == student.eos_id, student.pad_id, tgt_out)
            n = int((tgt_out != student.pad_id).sum())
    return {name: value / numel for name, value in totals.items()}


def load_frozen_teacher(checkpoint: str, table: DecompositionTable, **kwargs):
    """Load a fine-tuned PARSeq-S Lightning checkpoint and freeze it.

    Checks that its charset is exactly the 228 table characters in class order.
    Uses the PARSeq class directly, so the file name need not contain "parseq".
    """
    if PARSeq is None:
        raise ImportError("strhub (PARSeq) is required to load the teacher")
    teacher = freeze(PARSeq.load_from_checkpoint(checkpoint, map_location="cpu", **kwargs))
    expected = "".join(table.characters[1:])
    charset = teacher.hparams.get("charset_train")
    if charset != expected:
        raise ValueError("Teacher charset_train is not the 228-character table order; see docs/PROGRESS_VINH.md")
    return teacher


try:
    from strhub.models.parseq.system import PARSeq
except ImportError:  # strhub is optional for the pure loss code and its tests.
    PARSeq = None
else:

    class DistilledPARSeq(PARSeq):
        """Upstream PARSeq whose training step supports the factorized head and KD.

        The teacher is kept outside the module tree: it is not saved in the
        student checkpoint, not given to the optimizer and not switched to train
        mode by Lightning. It is moved to the student's device on first use.
        """

        _objective = DistillationObjective(enabled=False)
        _teacher = None

        def use_factorized_head(self, table: DecompositionTable) -> None:
            if len(table) != self.model.head.out_features:
                raise ValueError("Table size differs from the head's output size")
            self.model.head = FactorizedHead(self.model.head.in_features, table)

        def configure_distillation(self, objective: DistillationObjective, teacher=None, table: DecompositionTable | None = None) -> None:
            if objective.enabled and teacher is None:
                raise ValueError("KD is enabled but no teacher was given")
            self._objective = objective
            object.__setattr__(self, "_teacher", freeze(teacher) if teacher is not None else None)
            table = table or DecompositionTable.from_characters()
            self._matrices = marginal_matrices(table) if objective.enabled else None

        def training_step(self, batch, batch_idx):
            images, labels = batch
            teacher = self._teacher
            matrices = self._matrices if self._objective.enabled else None
            if teacher is not None:
                if next(teacher.parameters()).device != images.device:
                    teacher.to(images.device)
                teacher.eval()
                matrices = {name: matrix.to(images.device) for name, matrix in matrices.items()}
                self._matrices = matrices
            parts = shared_permutation_loss(self, teacher, images, labels, self._objective, matrices)
            for name, value in parts.items():
                self.log(name if name == "loss" else f"train/{name}", value)
            return parts["loss"]
