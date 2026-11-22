import pytest
from dfd.metrics import (
    INVALID_CHARACTER, normalize_text, word_accuracy, normalized_edit_similarity,
    one_minus_ned, levenshtein_alignment, component_error_counts,
    component_error_rates, tone_confusion_matrix, evaluate_predictions,
)


def test_normalize_nfc_lowercase_and_spaces():
    assert normalize_text(" A\u0306\u0301 B ") == "ắb"
    assert normalize_text("ĐƯỜNG") == "đường"
    assert normalize_text("A|B") == "a|b"


def test_invalid_marker_changed_to_replacement_char():
    assert INVALID_CHARACTER == "\ufffd"
    assert word_accuracy(["A|B", "a\ufffdb"], ["a|b", "a\ufffdb"]) == 0.5
    assert normalized_edit_similarity("a\ufffdb", "a\ufffdb") == 0.0


def test_case_insensitive_accuracy_ned():
    assert word_accuracy(["HỌCSINH", "CÀ PHÊ"], ["học sinh", "càphê"]) == 1.0
    assert one_minus_ned(["MẶT", "CÁ"], ["mắt", "cá"]) == pytest.approx(5 / 6)


def test_tone_substitution_with_fixed_gt_denominator():
    r = component_error_counts("MẶT", "MẮT")
    assert r == dict(base_errors=0, modifier_errors=0, tone_errors=1,
                     insertions=0, deletions=0, total=3)
    rates = component_error_rates(["MẶT"], ["MẮT"])
    assert rates["tone_er"] == pytest.approx(1 / 3)
    assert rates["insertion_rate"] == 0.0
    assert tone_confusion_matrix(["MẶT"], ["MẮT"])["acute"]["dot"] == 1


def test_modifier_and_base_substitutions():
    assert component_error_counts("a", "ă")["modifier_errors"] == 1
    assert component_error_counts("b", "a")["base_errors"] == 1
    assert component_error_counts("a", "ă")["tone_errors"] == 0


def test_insertion_only_separate_not_component():
    r = component_error_counts("HOC", "HO")
    assert r["total"] == 2
    assert (r["base_errors"], r["modifier_errors"], r["tone_errors"]) == (0, 0, 0)
    assert (r["insertions"], r["deletions"]) == (1, 0)
    rates = component_error_rates(["HOC"], ["HO"])
    assert rates["insertion_rate"] == pytest.approx(0.5)
    assert rates["deletion_rate"] == 0.0


def test_deletion_only_separate_not_component():
    r = component_error_counts("HO", "HOC")
    assert r["total"] == 3
    assert (r["base_errors"], r["modifier_errors"], r["tone_errors"]) == (0, 0, 0)
    assert (r["insertions"], r["deletions"]) == (0, 1)
    assert component_error_rates(["HO"], ["HOC"])["deletion_rate"] == pytest.approx(1 / 3)


def test_invalid_replacement_penalizes_components_not_entire_crop():
    r = component_error_counts("a\ufffdc", "abc")
    assert (r["base_errors"], r["modifier_errors"], r["tone_errors"]) == (1, 1, 1)
    assert r["total"] == 3
    assert component_error_rates(["a\ufffdc"], ["abc"])["base_er"] == pytest.approx(1 / 3)
    assert word_accuracy(["a\ufffdc"], ["abc"]) == 0
    assert normalized_edit_similarity("a\ufffdc", "abc") == 0
    assert tone_confusion_matrix(["a\ufffdc"], ["abc"])["level"]["level"] == 2


def test_invalid_equal_is_still_wrong():
    r = component_error_counts("a\ufffdc", "a\ufffdc")
    assert (r["base_errors"], r["modifier_errors"], r["tone_errors"]) == (1, 1, 1)
    assert word_accuracy(["a\ufffdc"], ["a\ufffdc"]) == 0


def test_dataset_micro_denominator_ignores_prediction_length():
    # GT length: 3 + 2 + 3 = 8. 1 tone replacement, 1 insertion, 1 deletion.
    result = evaluate_predictions(["MẶT", "HOC", "HO"], ["MẮT", "HO", "HOC"])
    er = result["component_error_rates"]
    assert er["tone_er"] == pytest.approx(1 / 8)
    assert er["base_er"] == 0
    assert er["modifier_er"] == 0
    assert er["insertion_rate"] == pytest.approx(1 / 8)
    assert er["deletion_rate"] == pytest.approx(1 / 8)
    assert result["word_accuracy"] == 0


def test_alignment_tracks_insert_delete():
    assert ("insert", None, "c") in levenshtein_alignment("HOC", "HO")
    assert ("delete", "c", None) in levenshtein_alignment("HO", "HOC")


def test_oov_degree_substitution_still_raises():
    with pytest.raises(ValueError):
        component_error_counts("°", "A")


def test_empty_gt_denominator_is_explicit_error():
    with pytest.raises(ValueError, match="zero characters"):
        component_error_rates(["abc"], [""])


def test_mismatched_lists_and_empty_dataset():
    with pytest.raises(ValueError):
        evaluate_predictions(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        evaluate_predictions([], [])


def test_evaluate_accepts_legacy_third_argument():
    out = evaluate_predictions(["A"], ["a"], vocabulary=None)
    assert out["word_accuracy"] == 1
    assert out["component_error_rates"]["tone_er"] == 0



# ============================================================
# ADDITIONAL TESTS — Vietnamese OCR Metrics
# ============================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", ""),
        ("   ", ""),
        ("A B C", "abc"),
        ("  XIN   CHÀO  ", "xinchào"),
        ("ĐIỆN THOẠI", "điệnthoại"),
        ("A|B", "a|b"),
        ("a\u0306\u0301", "ắ"),
        ("e\u0302\u0301", "ế"),
        ("o\u031b\u0300", "ờ"),
        ("u\u031b\u0323", "ự"),
    ],
)
def test_normalize_text_multiple_cases(text, expected):
    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    "prediction,ground_truth",
    [
        ("HỌC SINH", "họcsinh"),
        ("ĐƯỜNG", "đường"),
        ("CÀ PHÊ", "càphê"),
        ("a\u0306\u0301", "Ắ"),
        ("VIỆT NAM", "việtnam"),
    ],
)
def test_word_accuracy_equivalent_normalized_forms(
    prediction, ground_truth
):
    assert word_accuracy(
        [prediction], [ground_truth]
    ) == 1.0


@pytest.mark.parametrize(
    "prediction,ground_truth,expected",
    [
        ("abc", "abc", 1.0),
        ("abc", "abd", 2 / 3),
        ("abc", "ab", 2 / 3),
        ("ab", "abc", 2 / 3),
        ("a", "b", 0.0),
        ("", "", 1.0),
        ("", "abc", 0.0),
    ],
)
def test_normalized_edit_similarity_boundaries(
    prediction, ground_truth, expected
):
    assert normalized_edit_similarity(
        prediction, ground_truth
    ) == pytest.approx(expected)


def test_normalized_edit_similarity_symmetric():
    pairs = [
        ("abc", "abd"),
        ("HỌC", "HỌ"),
        ("VIỆT", "VIẾT"),
        ("xin", "xinchào"),
    ]

    for prediction, ground_truth in pairs:
        assert normalized_edit_similarity(
            prediction, ground_truth
        ) == pytest.approx(
            normalized_edit_similarity(
                ground_truth, prediction
            )
        )


def test_one_minus_ned_micro_average_by_word():
    predictions = ["abc", "ab", "a"]
    ground_truths = ["abc", "abc", "b"]

    # Similarities: 1, 2/3, 0
    expected = (1 + 2 / 3 + 0) / 3

    assert one_minus_ned(
        predictions, ground_truths
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "prediction,ground_truth,error_field",
    [
        ("a", "b", "base_errors"),
        ("a", "ă", "modifier_errors"),
        ("a", "á", "tone_errors"),
        ("mặt", "mắt", "tone_errors"),
        ("đ", "d", "modifier_errors"),
    ],
)
def test_component_error_types(
    prediction, ground_truth, error_field
):
    result = component_error_counts(
        prediction, ground_truth
    )

    assert result[error_field] == 1
    assert result["insertions"] == 0
    assert result["deletions"] == 0


def test_component_errors_zero_for_normalized_equivalence():
    result = component_error_counts(
        "ĐƯỜNG", "đường"
    )

    assert result["base_errors"] == 0
    assert result["modifier_errors"] == 0
    assert result["tone_errors"] == 0
    assert result["insertions"] == 0
    assert result["deletions"] == 0
    assert result["total"] == len("đường")


def test_component_rates_micro_average_not_macro_average():
    predictions = ["á", "bbbbbbbbb"]
    ground_truths = ["a", "bbbbbbbbb"]

    # One tone error over 10 ground-truth characters.
    rates = component_error_rates(
        predictions, ground_truths
    )

    assert rates["tone_er"] == pytest.approx(1 / 10)
    assert rates["base_er"] == 0
    assert rates["modifier_er"] == 0


def test_multiple_insertions_do_not_increase_component_errors():
    result = component_error_counts(
        "abcXYZ", "abc"
    )

    assert result["total"] == 3
    assert result["insertions"] == 3
    assert result["deletions"] == 0

    assert result["base_errors"] == 0
    assert result["modifier_errors"] == 0
    assert result["tone_errors"] == 0

    rates = component_error_rates(
        ["abcXYZ"], ["abc"]
    )
    assert rates["insertion_rate"] == pytest.approx(1.0)


def test_multiple_deletions_do_not_increase_component_errors():
    result = component_error_counts(
        "abc", "abcXYZ"
    )

    assert result["total"] == 6
    assert result["deletions"] == 3
    assert result["insertions"] == 0

    assert result["base_errors"] == 0
    assert result["modifier_errors"] == 0
    assert result["tone_errors"] == 0

    rates = component_error_rates(
        ["abc"], ["abcXYZ"]
    )
    assert rates["deletion_rate"] == pytest.approx(0.5)


def test_tone_confusion_matrix_counts_correct_pairs():
    predictions = ["á", "à", "a"]
    ground_truths = ["á", "á", "a"]

    matrix = tone_confusion_matrix(
        predictions, ground_truths
    )

    assert matrix["acute"]["acute"] == 1
    assert matrix["acute"]["grave"] == 1
    assert matrix["level"]["level"] == 1


def test_tone_confusion_ignores_insertion_deletion():
    matrix = tone_confusion_matrix(
        ["abcX"], ["abc"]
    )

    assert matrix["level"]["level"] == 3

    total = sum(
        count
        for row in matrix.values()
        for count in row.values()
    )

    assert total == 3


def test_invalid_marker_does_not_count_as_correct_tone():
    matrix = tone_confusion_matrix(
        ["a\ufffdc"], ["abc"]
    )

    # Only the two valid aligned characters contribute.
    assert matrix["level"]["level"] == 2

    total = sum(
        count
        for row in matrix.values()
        for count in row.values()
    )

    assert total == 2


def test_evaluate_predictions_matches_individual_metrics():
    predictions = ["MẶT", "HỌC", "abcX"]
    ground_truths = ["MẮT", "học", "abc"]

    result = evaluate_predictions(
        predictions, ground_truths
    )

    assert result["word_accuracy"] == pytest.approx(
        word_accuracy(predictions, ground_truths)
    )

    assert result["one_minus_ned"] == pytest.approx(
        one_minus_ned(predictions, ground_truths)
    )

    expected_rates = component_error_rates(
        predictions, ground_truths
    )

    for metric in [
        "base_er",
        "modifier_er",
        "tone_er",
        "insertion_rate",
        "deletion_rate",
    ]:
        assert result["component_error_rates"][
            metric
        ] == pytest.approx(expected_rates[metric])


def test_identical_valid_predictions_have_zero_errors():
    predictions = [
        "học",
        "sinh",
        "việt",
        "nam",
    ]

    result = evaluate_predictions(
        predictions, predictions
    )

    assert result["word_accuracy"] == 1.0
    assert result["one_minus_ned"] == 1.0

    for value in result["component_error_rates"].values():
        assert value == pytest.approx(0.0)
