import math

import pytest

try:
    import torch
except (ImportError, OSError) as error:  # A broken local install raises OSError on Windows.
    pytest.skip(f"PyTorch is unavailable: {error}", allow_module_level=True)

from dfd.heads import ComponentLogits, FactorizedHead  # noqa: E402
from dfd.vietchar import EOS, MODIFIERS, TONES, DecompositionTable  # noqa: E402


TABLE = DecompositionTable.from_characters()
PAD_ID = len(TABLE) + 1  # PARSeq order: EOS, characters, BOS, PAD.
HIDDEN = 192


@pytest.fixture
def head():
    torch.manual_seed(42)
    return FactorizedHead(HIDDEN, TABLE)


def component_scores(*chosen, value=30.0):
    """Build logits for a batch of one word, strongly predicting each (base, modifier, tone)."""
    logits = [torch.full((1, len(chosen), len(names)), -value) for names in (TABLE.bases, MODIFIERS, TONES)]
    for position, triple in enumerate(chosen):
        for tensor, names, name in zip(logits, (TABLE.bases, MODIFIERS, TONES), triple):
            tensor[0, position, names.index(name)] = value
    return ComponentLogits(*logits)


def test_shapes_match_upstream_head_contract(head):
    hidden = torch.randn(2, 7, HIDDEN)
    logits = head.component_logits(hidden)
    assert [part.shape for part in logits] == [(2, 7, 95), (2, 7, 5), (2, 7, 6)]
    assert head(hidden).shape == (2, 7, 229)


def test_parameter_count_matches_proposal_estimate(head):
    assert sum(parameter.numel() for parameter in head.parameters()) == (HIDDEN + 1) * 106


def test_upstream_device_lookup_finds_direct_parameters(head):
    assert next(head.parameters(recurse=False)).device == torch.device("cpu")


def test_composition_equals_normalized_product_over_valid_triples(head):
    logits = head.component_logits(torch.randn(1, 1, HIDDEN))
    probabilities = [part.softmax(-1)[0, 0] for part in logits]
    product = torch.stack([
        probabilities[0][b] * probabilities[1][m] * probabilities[2][t]
        for b, m, t in zip(head.base_ids, head.modifier_ids, head.tone_ids)
    ])
    expected = product / product.sum()
    assert torch.allclose(head.compose(logits).exp()[0, 0], expected, atol=1e-6)
    assert head.compose(logits).exp().sum(-1).item() == pytest.approx(1)


def test_mask_rejects_invalid_top_component_triple(head):
    logits = component_scores(("q", "none", "level"), value=0.0)
    for tensor in logits:
        tensor.fill_(-30.0)
    logits.base[0, 0, TABLE.bases.index("q")] = 10.0
    logits.base[0, 0, TABLE.bases.index("a")] = 8.0
    logits.modifier[0, 0, MODIFIERS.index("none")] = 10.0
    logits.tone[0, 0, TONES.index("tilde")] = 3.0
    logits.tone[0, 0, TONES.index("level")] = 0.0
    # q + tilde is the per-component top-1 but does not exist; ã is the best valid character.
    assert TABLE.characters[head.compose(logits).argmax(-1).item()] == "ã"


def test_composition_is_numerically_stable_for_extreme_logits(head):
    logits = ComponentLogits(*(torch.randn(1, 3, size) * 1e4 for size in head.sizes))
    log_probabilities = head.compose(logits)
    assert torch.isfinite(log_probabilities).all()
    assert torch.allclose(log_probabilities.exp().sum(-1), torch.ones(1, 3))


def test_greedy_decode_stops_at_eos(head):
    triples = [("D", "bar", "level"), ("e", "circumflex", "dot"), ("p", "none", "level"), (EOS, "none", "level"), ("a", "none", "level")]
    predicted = head.compose(component_scores(*triples)).argmax(-1)[0].tolist()
    assert predicted[3] == 0
    assert "".join(TABLE.characters[index] for index in predicted[:predicted.index(0)]) == "Đệp"


def test_loss_is_zero_for_perfect_and_log_k_for_uniform_scores(head):
    targets = torch.tensor([[TABLE.characters.index("ắ"), 0]])
    perfect = head.component_loss(component_scores(("a", "breve", "acute"), (EOS, "none", "level")), targets, PAD_ID)
    assert perfect["total"].item() == pytest.approx(0, abs=1e-6)
    uniform = ComponentLogits(*(torch.zeros(1, 2, size) for size in head.sizes))
    losses = head.component_loss(uniform, targets, PAD_ID)
    assert losses["base"].item() == pytest.approx(math.log(95))
    assert losses["modifier"].item() == pytest.approx(math.log(5))
    assert losses["tone"].item() == pytest.approx(math.log(6))
    assert losses["total"].item() == pytest.approx(math.log(95 * 5 * 6))


def test_loss_ignores_padding_and_reaches_every_classifier(head):
    hidden = torch.randn(1, 3, HIDDEN)
    targets = torch.tensor([[TABLE.characters.index("ệ"), 0, PAD_ID]])
    logits = head.component_logits(hidden)
    changed = ComponentLogits(*(part.clone() for part in logits))
    for part in changed:
        part[0, 2] += 100.0
    assert head.component_loss(changed, targets, PAD_ID)["total"].item() == pytest.approx(head.component_loss(logits, targets, PAD_ID)["total"].item())
    head.component_loss(logits, targets, PAD_ID)["total"].backward()
    assert all(block.abs().sum() > 0 for block in head.weight.grad.split(head.sizes, dim=0))


def test_drop_in_on_upstream_parseq():
    model_module = pytest.importorskip("strhub.models.parseq.model", exc_type=ImportError)
    tokenizer_module = pytest.importorskip("strhub.data.utils", exc_type=ImportError)
    tokenizer = tokenizer_module.Tokenizer("".join(TABLE.characters[1:]))
    assert tokenizer.eos_id == 0 and tokenizer.pad_id == PAD_ID
    assert all(tokenizer._stoi[character] == index for index, character in enumerate(TABLE.characters) if index)

    torch.manual_seed(42)
    model = model_module.PARSeq(
        num_tokens=len(tokenizer), max_label_length=25, img_size=(32, 128), patch_size=(4, 8), embed_dim=HIDDEN,
        enc_num_heads=3, enc_mlp_ratio=4, enc_depth=12, dec_num_heads=6, dec_mlp_ratio=4, dec_depth=1,
        decode_ar=True, refine_iters=1, dropout=0.1,
    )
    model.head = FactorizedHead(model.head.in_features, TABLE)

    images = torch.randn(2, 3, 32, 128)
    model.eval()
    with torch.no_grad():
        output = model(tokenizer, images, max_length=25)
    assert output.shape == (2, 26, 229)
    labels, _ = tokenizer.decode(output.softmax(-1))
    assert len(labels) == 2

    model.train()
    target = tokenizer.encode(["Việt", "đẹp"])
    target_in, target_out = target[:, :-1], target[:, 1:]
    hidden = model.decode(target_in, model.encode(images), tgt_padding_mask=(target_in == tokenizer.pad_id) | (target_in == tokenizer.eos_id))
    logits = model.head.component_logits(hidden)
    assert logits.base.shape == (2, target_out.shape[1], 95)
    model.head.component_loss(logits, target_out, tokenizer.pad_id)["total"].backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert next(p for n, p in model.named_parameters() if n.startswith("encoder")).grad is not None
