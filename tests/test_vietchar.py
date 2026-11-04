import unicodedata

import pytest

from dfd.vietchar import Components, build_table, compose, decompose


@pytest.mark.parametrize(("character", "expected"), [
    ("a", Components("a")),
    ("ắ", Components("a", "breve", "acute")),
    ("ệ", Components("e", "circumflex", "dot")),
    ("ờ", Components("o", "horn", "grave")),
    ("ữ", Components("u", "horn", "tilde")),
    ("đ", Components("d", "bar")),
    ("Đ", Components("D", "bar")),
    ("7", Components("7")),
    ("[EOS]", Components("[EOS]")),
])
def test_proposal_examples(character, expected):
    assert decompose(character) == expected
    assert compose(expected) == character


def test_vietnamese_vowels_round_trip_all_cases_and_tones():
    vowels = "aăâeêioôơuưy"
    marks = ("", "\u0301", "\u0300", "\u0309", "\u0303", "\u0323")
    characters = [unicodedata.normalize("NFC", vowel + mark) for vowel in vowels + vowels.upper() for mark in marks]
    table = build_table(characters + ["đ", "Đ", "7", "[EOS]"])
    assert len(table) == 148
    assert all(compose(parts) == character for character, parts in table.items())


def test_reject_invalid_triple_and_noncanonical_vocabulary():
    with pytest.raises(ValueError):
        compose(Components("q", "none", "tilde"))
    with pytest.raises(ValueError):
        compose(Components("y", "circumflex", "acute"))
    with pytest.raises(ValueError):
        build_table(["ắ", "a\u0306\u0301"])
    with pytest.raises(ValueError):
        build_table(["a\u0306\u0301"])
