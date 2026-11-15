"""Diacritic-factorized distillation (DFD) and standard KD losses.

Follows proposal note 06. The marginal-distillation principle comes from
Structural KD; it is not claimed as new here.

Teacher marginals:  p_T^x = softmax(z_T / tau) @ M_x,  x in {base, modifier, tone}.
Student marginals:  p_S^x(tau) = softmax(z_S^x / tau) from the factorized head.
KD term per position i:
    marginal: alpha_i * sum_x w_x * KL(p_T^x || p_S^x(tau))
    joint:    alpha_i * KL(p_T || p_S(tau)) over the 229 classes
alpha_i = 1 - H(p_T^tone) / log 6 when tone-confidence weighting is on, else 1.
Objective = CE + lambda * tau**2 * mean_i(KD_i).

Reduction: every term is a mean over valid (non-padding) target positions, the
same reduction as PARSeq's own CE, so CE and KD stay on one scale. alpha_i
attenuates the KD term; it is not renormalized by sum(alpha).

This module is pure torch. Running teacher and student on the same image,
permutation and context is done in `dfd.parseq_kd`.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from dfd.heads import ComponentLogits, FactorizedHead
from dfd.vietchar import COMPONENTS, TONES, DecompositionTable


MODES = ("marginal", "joint", "marginal_and_joint")


@dataclass(frozen=True)
class DistillationObjective:
    """KD settings parsed from a run YAML's `distillation` section."""

    enabled: bool
    mode: str = "marginal"
    temperature: float = 1.0
    weight: float = 1.0  # lambda in the proposal
    component_weights: dict[str, float] = field(default_factory=lambda: dict.fromkeys(COMPONENTS, 1.0))
    tone_confidence_weight: bool = False
    joint_weight: float | None = None  # weight of the joint KL in marginal_and_joint mode

    def __post_init__(self):
        if not self.enabled:
            return
        if self.mode not in MODES:
            raise ValueError(f"distillation.mode must be one of {MODES}, got {self.mode!r}")
        for name, value in (("temperature", self.temperature), ("lambda", self.weight)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"distillation.{name} must be positive")
        if self.mode != "joint":
            if set(self.component_weights) != set(COMPONENTS):
                raise ValueError(f"distillation.component_weights needs exactly {COMPONENTS}")
            if any(value < 0 for value in self.component_weights.values()):
                raise ValueError("distillation.component_weights must be nonnegative")
            if not any(value > 0 for value in self.component_weights.values()):
                raise ValueError("At least one component weight must be positive")
        if self.mode == "marginal_and_joint" and (self.joint_weight is None or self.joint_weight <= 0):
            raise ValueError("distillation.joint_lambda must be set (positive) for marginal_and_joint; it is still undecided")

    @classmethod
    def from_config(cls, section: dict) -> "DistillationObjective":
        if not section.get("enabled"):
            return cls(enabled=False)
        return cls(
            enabled=True,
            mode=section.get("mode", "marginal"),
            temperature=section["temperature"],
            weight=section["lambda"],
            component_weights={name: float(value) for name, value in (section.get("component_weights") or {}).items()}
            or dict.fromkeys(COMPONENTS, 1.0),
            tone_confidence_weight=bool(section.get("tone_confidence_weight", False)),
            joint_weight=section.get("joint_lambda"),
        )

    @property
    def needs_factorized_student(self) -> bool:
        return self.enabled and self.mode != "joint"


def marginal_matrices(table: DecompositionTable, device=None, dtype=torch.float32) -> dict[str, Tensor]:
    """Fixed binary M_x of shape (229, K_x) for each component."""
    return {name: torch.tensor(table.matrix(name), dtype=dtype, device=device) for name in COMPONENTS}


def marginalize(probabilities: Tensor, matrices: dict[str, Tensor]) -> dict[str, Tensor]:
    """Sum character probabilities [..., 229] into component marginals [..., K_x]."""
    return {name: probabilities @ matrix.to(probabilities) for name, matrix in matrices.items()}


def tone_confidence(tone_marginal: Tensor) -> Tensor:
    """alpha = 1 - H(p_tone) / log 6, in [0, 1]; 1 for a one-hot teacher, 0 for uniform."""
    entropy = -torch.special.xlogy(tone_marginal, tone_marginal).sum(-1)
    return (1 - entropy / math.log(len(TONES))).clamp(0, 1)


def kl_divergence(target: Tensor, log_prediction: Tensor) -> Tensor:
    """KL(target || prediction) over the last axis; zero-probability targets contribute 0."""
    return (torch.special.xlogy(target, target) - target * log_prediction).sum(-1)


def student_joint_log_probabilities(student: Tensor | ComponentLogits, temperature: float, head: FactorizedHead | None) -> Tensor:
    """log p_S(c | tau) over the 229 classes, for a joint or a factorized student."""
    if isinstance(student, ComponentLogits):
        if head is None:
            raise ValueError("A factorized student needs its head to compose the joint distribution")
        return head.compose(ComponentLogits(*(part / temperature for part in student)))
    return (student / temperature).log_softmax(-1)


def kd_position_terms(
    student: Tensor | ComponentLogits,
    teacher_logits: Tensor,
    objective: DistillationObjective,
    matrices: dict[str, Tensor],
    head: FactorizedHead | None = None,
) -> dict[str, Tensor]:
    """Unreduced KD terms per position, shape [N, L] each.

    `student` is either joint logits [N, L, 229] or the factorized head's
    `ComponentLogits`. `teacher_logits` are the teacher head outputs [N, L, 229]
    (raw logits or log-probabilities; the softmax is the same). Returns the
    weighted `kd` term, plus `kl_<component>`, `kl_joint` and `alpha` for logging.
    """
    tau = objective.temperature
    teacher = (teacher_logits.detach().float() / tau).softmax(-1)
    teacher_marginals = marginalize(teacher, matrices)
    terms: dict[str, Tensor] = {}
    kd = torch.zeros(teacher.shape[:-1], device=teacher.device)

    if objective.mode in ("marginal", "marginal_and_joint"):
        if not isinstance(student, ComponentLogits):
            raise ValueError(f"mode {objective.mode!r} needs a factorized student")
        for name in COMPONENTS:
            log_student = (getattr(student, name).float() / tau).log_softmax(-1)
            terms[f"kl_{name}"] = kl_divergence(teacher_marginals[name], log_student)
            kd = kd + objective.component_weights[name] * terms[f"kl_{name}"]

    if objective.mode in ("joint", "marginal_and_joint"):
        if isinstance(student, ComponentLogits):
            student = ComponentLogits(*(part.float() for part in student))
        else:
            student = student.float()
        terms["kl_joint"] = kl_divergence(teacher, student_joint_log_probabilities(student, tau, head))
        kd = kd + (objective.joint_weight if objective.mode == "marginal_and_joint" else 1.0) * terms["kl_joint"]

    alpha = tone_confidence(teacher_marginals["tone"]) if objective.tone_confidence_weight else torch.ones_like(kd)
    terms["alpha"] = alpha
    terms["kd"] = alpha * kd
    return terms


def ce_loss(student: Tensor | ComponentLogits, targets: Tensor, pad_id: int, head: FactorizedHead | None = None) -> dict[str, Tensor]:
    """Ground-truth CE: per-component CE (summed) for a factorized student, 229-way CE otherwise."""
    if isinstance(student, ComponentLogits):
        if head is None:
            raise ValueError("A factorized student needs its head for component targets")
        return head.component_loss(student, targets, pad_id)
    loss = F.cross_entropy(student.flatten(end_dim=-2), targets.flatten(), ignore_index=pad_id)
    return {"char": loss, "total": loss}


def distillation_loss(
    student: Tensor | ComponentLogits,
    teacher_logits: Tensor | None,
    targets: Tensor,
    pad_id: int,
    objective: DistillationObjective,
    matrices: dict[str, Tensor] | None = None,
    head: FactorizedHead | None = None,
) -> dict[str, Tensor]:
    """Full objective for one decoding pass: CE + lambda * tau^2 * mean over valid positions of KD.

    Returns `loss` plus detached-friendly parts: `ce`, `ce_<component>`, `kd`
    (before lambda * tau^2), `kl_<...>` and `alpha` averaged over valid positions.
    """
    ce = ce_loss(student, targets, pad_id, head)
    out = {"ce": ce["total"], **{f"ce_{name}": value for name, value in ce.items() if name != "total"}}
    out["loss"] = ce["total"]
    if not objective.enabled:
        return out
    if teacher_logits is None or matrices is None:
        raise ValueError("KD is enabled but teacher logits or marginal matrices are missing")
    valid = targets != pad_id
    count = valid.sum().clamp(min=1)
    for name, value in kd_position_terms(student, teacher_logits, objective, matrices, head).items():
        out[name] = (value * valid).sum() / count
    out["loss"] = out["loss"] + objective.weight * objective.temperature**2 * out["kd"]
    return out
