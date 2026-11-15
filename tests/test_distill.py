import math

import pytest

try:
    import torch
except (ImportError, OSError) as error:  # A broken local install raises OSError on Windows.
    pytest.skip(f"PyTorch is unavailable: {error}", allow_module_level=True)

from dfd.distill import (  # noqa: E402
    DistillationObjective,
    distillation_loss,
    kd_position_terms,
    kl_divergence,
    marginal_matrices,
    marginalize,
    tone_confidence,
)
from dfd.heads import ComponentLogits, FactorizedHead  # noqa: E402
from dfd.vietchar import TONES, DecompositionTable  # noqa: E402


TABLE = DecompositionTable.from_characters()
MATRICES = marginal_matrices(TABLE)
PAD_ID = len(TABLE) + 1
HIDDEN = 32


def objective(**overrides):
    settings = dict(
        enabled=True, mode="marginal", temperature=2.0, weight=1.0,
        component_weights={"base": 1.0, "modifier": 1.0, "tone": 2.0}, tone_confidence_weight=True,
    )
    settings.update(overrides)
    return DistillationObjective(**settings)


@pytest.fixture
def head():
    torch.manual_seed(0)
    return FactorizedHead(HIDDEN, TABLE)


def test_matrices_are_one_hot_rows():
    for name, size in (("base", 95), ("modifier", 5), ("tone", 6)):
        matrix = MATRICES[name]
        assert matrix.shape == (229, size)
        assert torch.equal(matrix.sum(1), torch.ones(229))


def test_marginals_sum_character_probabilities():
    probabilities = torch.randn(2, 3, 229).softmax(-1)
    marginals = marginalize(probabilities, MATRICES)
    for value in marginals.values():
        assert torch.allclose(value.sum(-1), torch.ones(2, 3), atol=1e-6)
    level = [index for index, parts in enumerate(TABLE.parts) if parts.tone == "level"]
    assert torch.allclose(marginals["tone"][..., TONES.index("level")], probabilities[..., level].sum(-1), atol=1e-6)
    a_bases = [index for index, parts in enumerate(TABLE.parts) if parts.base == "a"]
    assert len(a_bases) == 18  # a, ă, â, each with six tones
    assert torch.allclose(marginals["base"][..., TABLE.bases.index("a")], probabilities[..., a_bases].sum(-1), atol=1e-6)


def test_tone_confidence_bounds():
    uniform = torch.full((6,), 1 / 6)
    one_hot = torch.tensor([1.0, 0, 0, 0, 0, 0])
    assert tone_confidence(uniform).item() == pytest.approx(0, abs=1e-6)
    assert tone_confidence(one_hot).item() == pytest.approx(1)
    two_way = torch.tensor([0.5, 0.5, 0, 0, 0, 0])
    assert tone_confidence(two_way).item() == pytest.approx(1 - math.log(2) / math.log(6))


def test_kl_is_zero_for_identical_and_handles_zero_targets():
    target = torch.tensor([0.7, 0.3, 0.0])
    assert kl_divergence(target, target.clamp_min(1e-30).log()).item() == pytest.approx(0, abs=1e-6)
    prediction = torch.tensor([0.2, 0.3, 0.5])
    expected = 0.7 * math.log(0.7 / 0.2)
    assert kl_divergence(target, prediction.log()).item() == pytest.approx(expected, rel=1e-5)


def test_marginal_kl_matches_manual_formula(head):
    torch.manual_seed(1)
    student = head.component_logits(torch.randn(1, 2, HIDDEN))
    teacher_logits = torch.randn(1, 2, 229) * 3
    config = objective()
    terms = kd_position_terms(student, teacher_logits, config, MATRICES)
    teacher = (teacher_logits / 2).softmax(-1)
    marginals = {name: teacher @ MATRICES[name] for name in MATRICES}
    expected = 0
    for name, weight in config.component_weights.items():
        log_student = (getattr(student, name) / 2).log_softmax(-1)
        expected = expected + weight * (marginals[name] * (marginals[name].log() - log_student)).sum(-1)
    alpha = 1 + (marginals["tone"] * marginals["tone"].log()).sum(-1) / math.log(6)
    assert torch.allclose(terms["kd"], alpha * expected, atol=1e-5)
    assert torch.allclose(terms["alpha"], alpha, atol=1e-6)


def test_marginal_kd_vanishes_when_student_matches_teacher_marginals():
    torch.manual_seed(2)
    teacher_logits = torch.randn(1, 4, 229)
    tau = 2.0
    marginals = marginalize((teacher_logits / tau).softmax(-1), MATRICES)
    # Student logits whose tempered softmax equals the teacher marginals.
    student = ComponentLogits(*(tau * marginals[name].log() for name in ("base", "modifier", "tone")))
    terms = kd_position_terms(student, teacher_logits, objective(temperature=tau), MATRICES)
    assert terms["kd"].abs().max().item() < 1e-5


def test_joint_kd_is_standard_hinton_kl():
    torch.manual_seed(3)
    student = torch.randn(2, 3, 229)
    teacher_logits = torch.randn(2, 3, 229)
    config = objective(mode="joint", tone_confidence_weight=False, temperature=4.0)
    terms = kd_position_terms(student, teacher_logits, config, MATRICES)
    expected = torch.nn.functional.kl_div(
        (student / 4).log_softmax(-1), (teacher_logits / 4).softmax(-1), reduction="none"
    ).sum(-1)
    assert torch.allclose(terms["kd"], expected, atol=1e-5)
    assert torch.equal(terms["alpha"], torch.ones(2, 3))


def test_joint_kd_on_factorized_student_uses_composed_distribution(head):
    torch.manual_seed(4)
    student = head.component_logits(torch.randn(1, 2, HIDDEN))
    teacher_logits = torch.randn(1, 2, 229)
    config = objective(mode="joint", tone_confidence_weight=False, temperature=2.0)
    terms = kd_position_terms(student, teacher_logits, config, MATRICES, head)
    composed = head.compose(ComponentLogits(*(part / 2 for part in student)))
    teacher = (teacher_logits / 2).softmax(-1)
    assert torch.allclose(terms["kd"], (teacher * (teacher.log() - composed)).sum(-1), atol=1e-5)
    with pytest.raises(ValueError, match="needs its head"):
        kd_position_terms(student, teacher_logits, config, MATRICES)


def test_marginal_and_joint_adds_weighted_joint_term(head):
    torch.manual_seed(5)
    student = head.component_logits(torch.randn(1, 2, HIDDEN))
    teacher_logits = torch.randn(1, 2, 229)
    marginal = kd_position_terms(student, teacher_logits, objective(tone_confidence_weight=False), MATRICES, head)
    both = kd_position_terms(
        student, teacher_logits, objective(mode="marginal_and_joint", joint_weight=0.5, tone_confidence_weight=False), MATRICES, head
    )
    assert torch.allclose(both["kd"], marginal["kd"] + 0.5 * both["kl_joint"], atol=1e-6)


def test_tone_only_weights_ignore_base_and_modifier(head):
    torch.manual_seed(6)
    student = head.component_logits(torch.randn(1, 3, HIDDEN))
    teacher_logits = torch.randn(1, 3, 229)
    config = objective(component_weights={"base": 0.0, "modifier": 0.0, "tone": 2.0}, tone_confidence_weight=False)
    terms = kd_position_terms(student, teacher_logits, config, MATRICES)
    assert torch.allclose(terms["kd"], 2 * terms["kl_tone"])


def test_loss_masks_padding_and_scales_by_lambda_tau_squared(head):
    torch.manual_seed(7)
    student = head.component_logits(torch.randn(2, 4, HIDDEN))
    teacher_logits = torch.randn(2, 4, 229)
    targets = torch.tensor([[5, 40, 0, PAD_ID], [7, 0, PAD_ID, PAD_ID]])
    config = objective(temperature=2.0, weight=0.5)
    parts = distillation_loss(student, teacher_logits, targets, PAD_ID, config, MATRICES, head)
    terms = kd_position_terms(student, teacher_logits, config, MATRICES)
    valid = targets != PAD_ID
    assert parts["kd"].item() == pytest.approx(terms["kd"][valid].mean().item(), rel=1e-5)
    assert parts["loss"].item() == pytest.approx(parts["ce"].item() + 0.5 * 4 * parts["kd"].item(), rel=1e-5)
    # Changing the teacher at padding positions must not change the loss.
    teacher_logits[valid.logical_not()] = torch.randn(int((~valid).sum()), 229) * 10
    again = distillation_loss(student, teacher_logits, targets, PAD_ID, config, MATRICES, head)
    assert again["loss"].item() == pytest.approx(parts["loss"].item(), rel=1e-6)


def test_gradients_reach_student_only(head):
    hidden = torch.randn(1, 3, HIDDEN)
    teacher_logits = torch.randn(1, 3, 229, requires_grad=True)
    targets = torch.tensor([[10, 0, PAD_ID]])
    parts = distillation_loss(head.component_logits(hidden), teacher_logits, targets, PAD_ID, objective(), MATRICES, head)
    parts["loss"].backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert teacher_logits.grad is None


def test_disabled_kd_is_component_ce(head):
    student = head.component_logits(torch.randn(1, 3, HIDDEN))
    targets = torch.tensor([[10, 0, PAD_ID]])
    parts = distillation_loss(student, None, targets, PAD_ID, DistillationObjective(enabled=False), head=head)
    assert parts["loss"].item() == pytest.approx(head.component_loss(student, targets, PAD_ID)["total"].item())
    assert "kd" not in parts


def test_joint_student_ce_is_229_way():
    student = torch.randn(1, 3, 229)
    targets = torch.tensor([[10, 0, PAD_ID]])
    parts = distillation_loss(student, None, targets, PAD_ID, DistillationObjective(enabled=False))
    expected = torch.nn.functional.cross_entropy(student[0, :2], targets[0, :2])
    assert parts["loss"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_extreme_logits_stay_finite(head):
    student = ComponentLogits(*(torch.randn(1, 2, size) * 1e4 for size in head.sizes))
    teacher_logits = torch.randn(1, 2, 229) * 1e4
    parts = distillation_loss(student, teacher_logits, torch.tensor([[3, 0]]), PAD_ID, objective(), MATRICES, head)
    assert torch.isfinite(parts["loss"])


def test_half_precision_inputs_are_computed_in_float32(head):
    student = ComponentLogits(*(torch.randn(1, 2, size).half() for size in head.sizes))
    terms = kd_position_terms(student, torch.randn(1, 2, 229).half(), objective(), MATRICES)
    assert terms["kd"].dtype == torch.float32


def test_marginal_mode_rejects_joint_student():
    with pytest.raises(ValueError, match="factorized student"):
        kd_position_terms(torch.randn(1, 2, 229), torch.randn(1, 2, 229), objective(), MATRICES)


@pytest.mark.parametrize("overrides, message", [
    ({"mode": "logits"}, "mode"),
    ({"temperature": 0}, "temperature"),
    ({"weight": -1.0}, "lambda"),
    ({"component_weights": {"base": 1.0, "tone": 1.0}}, "component_weights"),
    ({"component_weights": {"base": 0.0, "modifier": 0.0, "tone": 0.0}}, "positive"),
    ({"mode": "marginal_and_joint", "joint_weight": None}, "joint_lambda"),
])
def test_objective_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        objective(**overrides)


def test_objective_from_every_kd_run_config():
    from pathlib import Path

    from dfd.config import load_config

    root = Path(__file__).resolve().parents[1] / "configs" / "runs"
    seen = set()
    for path in sorted(root.rglob("*.yaml")):
        section = load_config(path)["distillation"]
        if section.get("mode") == "marginal_and_joint" and section.get("joint_lambda") is None:
            with pytest.raises(ValueError, match="joint_lambda"):
                DistillationObjective.from_config(section)
            continue
        parsed = DistillationObjective.from_config(section)
        seen.add((parsed.enabled, parsed.mode if parsed.enabled else None))
    assert {(True, "marginal"), (True, "joint"), (False, None)} <= seen
