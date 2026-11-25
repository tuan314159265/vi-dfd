import csv
import torch
import pytest
from dfd.baselines import CRNN, ctc_decode, validation_accuracy, write_predictions


def test_ctc_repeated_characters_and_blank():
    assert ctc_decode([1, 1, 0, 1, 2, 2, 0], 'ab') == 'aab'
    assert ctc_decode([0, 0], 'ab') == ''


def test_crnn_backward_and_repeat_alignment():
    torch.set_num_threads(2)
    model = CRNN('aấb')
    images = torch.randn(2, 3, 32, 128)
    # All 25 identical letters require 49 frames: must not silently become zero loss.
    loss = model.loss(images, ['a' * 25, 'ấb'])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    model.eval()
    assert len(model.predict(images)) == 2


def test_validation_retains_unsupported_and_long_labels():
    rows = [('1', 'Việt', 'việt'), ('2', '25°C', '25C'), ('3', 'a' * 30, 'a' * 25)]
    assert validation_accuracy(rows) == pytest.approx(1 / 3)


def test_prediction_csv_keeps_source_id_and_unicode(tmp_path):
    path = tmp_path / 'validation.csv'
    write_predictions(path, [('2:5', 'ấ', 'ầ'), ('3:7', 'a', '')])
    assert list(csv.DictReader(path.open(encoding='utf-8'))) == [
        {'source_id': '2:5', 'prediction': 'ầ'}, {'source_id': '3:7', 'prediction': ''}]


def test_vietocr_weights_reject_wrong_checksum(tmp_path):
    from dfd.baselines import verify_vietocr_weights
    path = tmp_path / 'weights.pth'
    path.write_bytes(b'corrupt weights')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        verify_vietocr_weights(path)


def test_vietocr_weights_accept_matching_checksum(tmp_path, monkeypatch):
    import hashlib
    from dfd import baselines
    path = tmp_path / 'weights.pth'
    content = b'test fixture weights'
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(baselines, 'VIETOCR_WEIGHTS_SHA256', expected)
    assert baselines.verify_vietocr_weights(path) == expected
