"""PARSeq-Ti student wiring (needs PyTorch and upstream PARSeq at PARSEQ_ROOT)."""

import copy
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("strhub.models.parseq.system", reason="PARSeq (strhub) is not on the path")
try:
    from dfd import student  # imports dfd.teacher, which checks the pinned PARSeq checkout
except (FileNotFoundError, ValueError, OSError) as error:
    pytest.skip(f"Pinned PARSeq checkout unavailable: {error}", allow_module_level=True)

from strhub.models.parseq.system import PARSeq  # noqa: E402

from dfd import evaluation  # noqa: E402
from dfd.heads import FactorizedHead  # noqa: E402
from dfd.parseq import initialize_teacher  # noqa: E402
from dfd.vietchar import ASCII_CHARSET, DecompositionTable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = DecompositionTable.from_characters()
NARROW = dict(embed_dim=32, enc_num_heads=2, dec_num_heads=2)


def config(name):
    return yaml.safe_load((ROOT / "configs/runs" / name).read_text(encoding="utf-8"))


def plain(charset, embed_dim=32):
    return PARSeq(charset_train=charset, charset_test=charset, max_label_length=25, batch_size=2, lr=1e-3,
                  warmup_pct=0.1, weight_decay=0.0, img_size=[32, 128], patch_size=[4, 8], embed_dim=embed_dim,
                  enc_num_heads=2, enc_mlp_ratio=2, enc_depth=1, dec_num_heads=2, dec_mlp_ratio=2, dec_depth=1,
                  perm_num=6, perm_forward=True, perm_mirrored=True, decode_ar=True, refine_iters=1, dropout=0.1)


def test_mro_runs_kd_step_inside_teacher_style_hooks():
    names = [cls.__name__ for cls in student.StudentPARSeq.__mro__[:4]]
    assert names == ["StudentPARSeq", "VietnamesePARSeq", "DistilledPARSeq", "PARSeq"]


@pytest.mark.parametrize("name, factorized", [("E2/ti_ce_s42.yaml", False), ("E2/ti_factorized_s42.yaml", True),
                                             ("E2/ti_kd_s42.yaml", False), ("E2/ti_dfd_s42.yaml", True)])
def test_e2_variants_build_and_train_one_step(name, factorized, monkeypatch):
    torch.manual_seed(0)
    cfg = config(name)
    model, info = student.build_student(cfg, TABLE, pretrained=False, size=NARROW)
    assert isinstance(model.model.head, FactorizedHead) == factorized
    teacher = plain("".join(TABLE.characters[1:]))
    monkeypatch.setattr(student, "load_frozen_teacher", lambda path, table: teacher)
    objective, used = student.attach_teacher(model, cfg, TABLE)
    assert objective.enabled == cfg["distillation"]["enabled"] and (used is teacher) == objective.enabled
    logged = {}
    model.log = lambda key, value, **kwargs: logged.setdefault(key, float(value))
    before = copy.deepcopy(teacher.state_dict())
    loss = model.training_step((torch.randn(2, 3, 32, 128), ["phở", "Việt"]), 0)
    loss.backward()
    assert torch.isfinite(loss) and "train_ce" in logged
    assert ("train/kd" in logged) == objective.enabled
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())


def test_student_checkpoint_round_trips_through_eval(tmp_path):
    torch.manual_seed(0)
    model, _ = student.build_student(config("E2/ti_dfd_s42.yaml"), TABLE, pretrained=False, size=NARROW)
    path = tmp_path / "student.ckpt"
    torch.save({"hyper_parameters": dict(model.hparams), "state_dict": model.state_dict()}, path)
    loaded = evaluation.load_parseq_checkpoint(str(path), TABLE)
    images = torch.randn(3, 3, 32, 128)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(loaded(images), model(images), atol=1e-6)


def test_english_initialization_accepts_tiny_width(tmp_path):
    english = plain(ASCII_CHARSET, embed_dim=32)
    path = tmp_path / "english.pt"
    torch.save(english.model.state_dict(), path)
    target = plain("".join(TABLE.characters[1:]), embed_dim=32)
    info = initialize_teacher(target, TABLE, path, url="file://english.pt")
    assert info["reset_keys"] == ["head.weight", "head.bias"] and info["output_classes"] == 229
    assert info["copied_context_embeddings"] == 97
    with pytest.raises(ValueError, match="width 32"):
        torch.save({**english.model.state_dict(), "head.weight": torch.zeros(95, 64)}, path)
        initialize_teacher(target, TABLE, path)


@pytest.mark.parametrize("edit, message", [
    (lambda c: c["data"].update(training_fraction=0.25), "training_fraction"),
    (lambda c: c.setdefault("composition", {}).update(validity_mask=False), "no_validity_mask"),
    (lambda c: c["model"].update(backbone="parseq_small"), "PARSeq-Ti"),
])
def test_run_student_refuses_unsupported_runs(edit, message):
    cfg = config("E2/ti_dfd_s42.yaml")
    edit(cfg)
    with pytest.raises(ValueError, match=message):
        student.run_student(cfg, ROOT)
