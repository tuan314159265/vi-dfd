"""Integration tests against upstream PARSeq (skipped when strhub is not importable)."""

import copy
import warnings

import pytest

try:
    import torch
except (ImportError, OSError) as error:
    pytest.skip(f"PyTorch is unavailable: {error}", allow_module_level=True)

strhub_system = pytest.importorskip("strhub.models.parseq.system", reason="PARSeq (strhub) is not on the path")

from dfd.distill import DistillationObjective, marginal_matrices  # noqa: E402
from dfd.heads import FactorizedHead  # noqa: E402
from dfd.parseq_kd import DistilledPARSeq, freeze, shared_permutation_loss  # noqa: E402
from dfd.vietchar import DecompositionTable  # noqa: E402


TABLE = DecompositionTable.from_characters()
CHARSET = "".join(TABLE.characters[1:])
MATRICES = marginal_matrices(TABLE)
LABELS = ["Việt", "ĐẸP", "phở", "25k", "quốc"]
DFD = DistillationObjective(
    enabled=True, mode="marginal", temperature=2.0, weight=1.0,
    component_weights={"base": 1.0, "modifier": 1.0, "tone": 2.0}, tone_confidence_weight=True,
)
JOINT = DistillationObjective(enabled=True, mode="joint", temperature=2.0, weight=1.0)


def build(cls=DistilledPARSeq, embed_dim=32, seed=0):
    torch.manual_seed(seed)
    return cls(
        charset_train=CHARSET, charset_test=CHARSET, max_label_length=25, batch_size=len(LABELS),
        lr=1e-3, warmup_pct=0.1, weight_decay=0.0, img_size=[32, 128], patch_size=[4, 8],
        embed_dim=embed_dim, enc_num_heads=2, enc_mlp_ratio=2, enc_depth=1,
        dec_num_heads=2, dec_mlp_ratio=2, dec_depth=1, perm_num=6, perm_forward=True,
        perm_mirrored=True, decode_ar=True, refine_iters=1, dropout=0.1,
    )


def images(seed=1):
    torch.manual_seed(seed)
    return torch.randn(len(LABELS), 3, 32, 128)


def fixed_perms(system):
    tgt = system.tokenizer.encode(LABELS, system.device)
    perms = system.gen_tgt_perms(tgt)
    return lambda _: perms


def test_tokenizer_matches_table_order():
    system = build()
    assert len(system.tokenizer) == 231
    assert system.tokenizer._itos[0] == "[E]" and system.eos_id == 0  # upstream spells EOS "[E]"
    assert system.tokenizer._itos[1 : len(TABLE)] == TABLE.characters[1:]
    assert system.pad_id == len(TABLE) + 1
    assert system.tokenizer.encode(["Việt"])[0, 1:5].tolist() == [TABLE.characters.index(c) for c in "Việt"]


def test_ce_only_step_matches_upstream_training_step():
    student = build(cls=strhub_system.PARSeq).eval()  # eval: no dropout, so both passes are identical
    perms = fixed_perms(student)
    student.gen_tgt_perms = perms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # self.log without a Trainer
        upstream = student.training_step((images(), LABELS), 0)
    ours = shared_permutation_loss(student, None, images(), LABELS, DistillationObjective(enabled=False), permutations=perms)
    assert ours["loss"].item() == pytest.approx(upstream.item(), rel=1e-5)


def test_identical_teacher_gives_zero_joint_kd_on_every_permutation():
    """Teacher = student weights; KD is 0 only if both see the same permutation and context."""
    student = build().eval()
    teacher = freeze(copy.deepcopy(student))
    parts = shared_permutation_loss(student, teacher, images(), LABELS, JOINT, MATRICES, permutations=fixed_perms(student))
    assert parts["kd"].item() < 1e-6
    assert parts["ce"].item() > 0
    # Control: a different teacher produces a clearly nonzero KD.
    other = freeze(build(seed=5).eval())
    control = shared_permutation_loss(student, other, images(), LABELS, JOINT, MATRICES, permutations=fixed_perms(student))
    assert control["kd"].item() > 1e-3


def test_dfd_step_with_larger_teacher_backpropagates_to_student_only():
    student = build()
    student.use_factorized_head(TABLE)
    teacher = freeze(build(embed_dim=48, seed=3))  # PARSeq-S is wider than Ti; only logits are compared
    parts = shared_permutation_loss(student, teacher, images(), LABELS, DFD, MATRICES)
    for name in ("loss", "ce", "ce_base", "ce_modifier", "ce_tone", "kd", "kl_base", "kl_modifier", "kl_tone", "alpha"):
        assert torch.isfinite(parts[name]), name
    assert 0 <= parts["alpha"].item() <= 1
    assert parts["loss"].item() == pytest.approx(parts["ce"].item() + 4 * parts["kd"].item(), rel=1e-5)
    parts["loss"].backward()
    assert student.model.head.weight.grad.abs().sum() > 0
    assert student.model.encoder.patch_embed.proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_marginal_mode_needs_factorized_head():
    student, teacher = build(), freeze(build(seed=3))
    with pytest.raises(ValueError, match="factorized student head"):
        shared_permutation_loss(student, teacher, images(), LABELS, DFD, MATRICES)


def test_teacher_tokenizer_must_match():
    student = build()
    student.use_factorized_head(TABLE)
    teacher = freeze(build(seed=3))
    teacher.tokenizer._itos = teacher.tokenizer._itos[::-1]
    with pytest.raises(ValueError, match="tokenizers differ"):
        shared_permutation_loss(student, teacher, images(), LABELS, DFD, MATRICES)


def test_teacher_is_hidden_from_checkpoint_and_optimizer():
    student = build()
    student.use_factorized_head(TABLE)
    teacher = build(embed_dim=48, seed=3)
    student.configure_distillation(DFD, teacher, TABLE)
    assert not any(key.startswith("_teacher") for key in student.state_dict())
    student_ids = {id(parameter) for parameter in student.parameters()}
    assert not student_ids & {id(parameter) for parameter in teacher.parameters()}
    student.train()
    assert not teacher.training
    assert isinstance(student.model.head, FactorizedHead)


def test_kd_without_teacher_is_rejected():
    student = build()
    student.use_factorized_head(TABLE)
    with pytest.raises(ValueError, match="no teacher"):
        student.configure_distillation(DFD, None)


def test_lightning_fit_runs_dfd_steps():
    pl = pytest.importorskip("pytorch_lightning")
    from torch.utils.data import DataLoader, Dataset

    class Batches(Dataset):
        def __len__(self):
            return 2 * len(LABELS)

        def __getitem__(self, index):
            torch.manual_seed(index)
            return torch.randn(3, 32, 128), LABELS[index % len(LABELS)]

    student = build()
    student.use_factorized_head(TABLE)
    teacher = build(embed_dim=48, seed=3)
    student.configure_distillation(DFD, teacher, TABLE)
    before = student.model.head.weight.detach().clone()
    teacher_before = copy.deepcopy(teacher.state_dict())
    trainer = pl.Trainer(
        max_steps=2, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(student, DataLoader(Batches(), batch_size=len(LABELS), num_workers=0))
    assert trainer.global_step == 2
    assert not torch.equal(before, student.model.head.weight)
    assert all(torch.equal(value, teacher.state_dict()[key]) for key, value in teacher_before.items())
    for name in ("loss", "train/kd", "train/kl_tone", "train/alpha", "train/ce_tone"):
        assert name in trainer.callback_metrics, name
