import unicodedata

import pytest

from dfd.vietchar import ASCII_CHARSET, EOS, MODIFIERS, TONES, Components, DecompositionTable, build_table, compose, decompose


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


TABLE = DecompositionTable.from_characters()


def test_vocabulary_has_229_classes_with_eos_first():
    assert len(ASCII_CHARSET) == len(set(ASCII_CHARSET)) == 94
    assert len(TABLE) == 229
    assert TABLE.characters[0] == EOS
    assert TABLE.characters[1:95] == tuple(ASCII_CHARSET)
    assert all(unicodedata.is_normalized("NFC", character) for character in TABLE.characters)
    assert sum(not character.isascii() for character in TABLE.characters) == 134


def test_vocabulary_contains_every_vietnamese_letter_form():
    lowercase = (
        "a\u00e0\u00e1\u1ea3\u00e3\u1ea1\u0103\u1eb1\u1eaf\u1eb3\u1eb5\u1eb7\u00e2\u1ea7\u1ea5\u1ea9\u1eab\u1eadbcd\u0111e\u00e8\u00e9\u1ebb\u1ebd\u1eb9\u00ea\u1ec1\u1ebf\u1ec3\u1ec5\u1ec7fghi\u00ec\u00ed\u1ec9\u0129\u1ecbjklmno\u00f2\u00f3\u1ecf\u00f5\u1ecd\u00f4\u1ed3\u1ed1\u1ed5\u1ed7\u1ed9\u01a1\u1edd\u1edb\u1edf\u1ee1\u1ee3"
        "pqrstu\u00f9\u00fa\u1ee7\u0169\u1ee5\u01b0\u1eeb\u1ee9\u1eed\u1eef\u1ef1vwxy\u1ef3\u00fd\u1ef7\u1ef9\u1ef5z"
    )
    assert len(lowercase) == 93
    assert set(lowercase) | set(lowercase.upper()) <= set(TABLE.characters)


def test_component_vocabularies_match_proposal_sizes():
    assert TABLE.bases == (EOS, *ASCII_CHARSET)
    assert [len(TABLE.classes(name)) for name in ("base", "modifier", "tone")] == [95, 5, 6]


def test_every_class_round_trips_and_triples_are_unique():
    assert all(compose(parts) == character for character, parts in zip(TABLE.characters, TABLE.parts))
    assert len(set(TABLE.parts)) == 229


@pytest.mark.parametrize(("character", "base", "modifier", "tone"), [
    ("\u1eab", "a", "circumflex", "tilde"),
    ("\u1eb4", "A", "breve", "tilde"),
    ("\u1ead", "a", "circumflex", "dot"),
    ("\u1ede", "O", "horn", "hook"),
    ("\u1ef1", "u", "horn", "dot"),
    ("\u1ef5", "y", "none", "dot"),
    ("\u1ef6", "Y", "none", "hook"),
    ("\u1ebd", "e", "none", "tilde"),
    ("\u0111", "d", "bar", "level"),
    ("\u0110", "D", "bar", "level"),
])
def test_hand_verified_entries(character, base, modifier, tone):
    assert TABLE.parts[TABLE.characters.index(character)] == Components(base, modifier, tone)


@pytest.mark.parametrize("component", ["base", "modifier", "tone"])
def test_matrices_map_each_class_to_exactly_one_component(component):
    matrix = TABLE.matrix(component)
    assert len(matrix) == 229
    assert {len(row) for row in matrix} == {len(TABLE.classes(component))}
    assert {sum(row) for row in matrix} == {1}
    assert all(set(row) <= {0, 1} for row in matrix)
    assert [row.index(1) for row in matrix] == TABLE.component_ids(component)


def test_marginalizing_sums_probability_by_component():
    probabilities = [0.0] * len(TABLE)
    for character, probability in (("\u1eaf", 0.5), ("\u1ea5", 0.25), ("a", 0.15), (EOS, 0.1)):
        probabilities[TABLE.characters.index(character)] = probability
    marginal = {
        component: [sum(p * row[k] for p, row in zip(probabilities, TABLE.matrix(component))) for k in range(len(TABLE.classes(component)))]
        for component in ("base", "modifier", "tone")
    }
    assert marginal["tone"] == pytest.approx([0.25, 0.75, 0, 0, 0, 0])
    assert marginal["modifier"] == pytest.approx([0.25, 0.25, 0.5, 0, 0])
    assert marginal["base"][TABLE.bases.index("a")] == pytest.approx(0.9)
    assert marginal["base"][TABLE.bases.index(EOS)] == pytest.approx(0.1)


def test_modifier_and_tone_orders_are_fixed():
    assert MODIFIERS == ("none", "circumflex", "breve", "horn", "bar")
    assert TONES == ("level", "acute", "grave", "hook", "tilde", "dot")


def test_coverage_reports_out_of_vocabulary_and_unused():
    missing, unused = TABLE.coverage(["a", "a\u0306\u0301", "\u20ac", " "])
    assert missing == [" ", "\u20ac"]
    assert "a" not in unused and "\u1eaf" not in unused and "b" in unused


def test_custom_vocabulary_rejects_explicit_eos():
    with pytest.raises(ValueError):
        DecompositionTable.from_characters(["a", EOS])


def test_fingerprint_pins_class_order():
    assert TABLE.fingerprint() == "5958403f4ca3a84c80d007a9af14c1f9cfd9ffa7dd0fd3cd812c65140911360d"
