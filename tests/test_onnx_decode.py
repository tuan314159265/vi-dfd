from bench.predict_onnx import decode_ids


def test_decoding_stops_at_eos_and_keeps_vietnamese():
    assert decode_ids([1, 2, 0, 3], ['[E]', 'ấ', 'đ', 'a']) == 'ấđ'
    assert decode_ids([0, 1], ['[E]', 'a']) == ''
    assert decode_ids([1, 1], ['[E]', 'a']) == 'aa'
