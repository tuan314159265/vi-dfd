
import pytest
from dfd.metrics import normalize_text
from dfd.metrics import word_accuracy
from dfd.metrics import normalized_edit_similarity, one_minus_ned
from dfd.metrics import levenshtein_alignment
from dfd.metrics import (
    component_error_counts,
    component_error_rates,
)
from dfd.metrics import tone_confusion_matrix
from dfd.metrics import evaluate_predictions


# TESTS FOR WORD ACCURACY
def test_word_accuracy():
    ground_truths = [
        "CHẤT",
        "LƯỢNG",
        "HỌC SINH",
        "MẮT",
        "A|B",
    ]

    predictions = [
        "CHẤT",
        "LƯƠNG",
        "HỌCSINH",
        "MẶT",
        "A|B",
    ]

    result = word_accuracy(predictions, ground_truths)

    assert result == pytest.approx(0.4)


def test_word_accuracy_nfc():
    # "á" represented as a + combining acute accent
    predictions = ["a\u0301"]
    ground_truths = ["á"]

    assert word_accuracy(predictions, ground_truths) == 1.0


def test_word_accuracy_empty():
    with pytest.raises(ValueError):
        word_accuracy([], [])


def test_word_accuracy_mismatched_lengths():
    with pytest.raises(ValueError):
        word_accuracy(["A"], ["A", "B"])


# TESTS FOR NORMALIZED EDIT SIMILARITY
def test_ned_exact_match():
    assert normalized_edit_similarity("CHẤT", "CHẤT") == 1.0


def test_ned_one_substitution():
    result = normalized_edit_similarity("MẶT", "MẮT")
    assert result == pytest.approx(2 / 3)


def test_ned_whitespace_normalization():
    assert normalized_edit_similarity("HỌCSINH", "HỌC SINH") == 1.0


def test_ned_invalid_character():
    assert normalized_edit_similarity("A|B", "A|B") == 0.0


def test_ned_empty_strings():
    assert normalized_edit_similarity("", "") == 1.0
    assert normalized_edit_similarity("", "ABC") == 0.0


def test_dataset_one_minus_ned():
    predictions = ["CHẤT", "MẶT", "HỌCSINH"]
    ground_truths = ["CHẤT", "MẮT", "HỌC SINH"]

    result = one_minus_ned(predictions, ground_truths)

    assert result == pytest.approx(8 / 9)


def test_ned_mismatched_lengths():
    with pytest.raises(ValueError):
        one_minus_ned(["A"], ["A", "B"])
        
        
        
# TESTS FOR LEVENSHTEIN ALIGNMENT

def test_alignment_replace():
    result = levenshtein_alignment("MẶT", "MẮT")

    assert result == [
        ("equal", "M", "M"),
        ("replace", "Ắ", "Ặ"),
        ("equal", "T", "T"),
    ]


def test_alignment_delete():
    result = levenshtein_alignment("HO", "HOC")

    assert result == [
        ("equal", "H", "H"),
        ("equal", "O", "O"),
        ("delete", "C", None),
    ]


def test_alignment_insert():
    result = levenshtein_alignment("HOC", "HO")

    assert result == [
        ("equal", "H", "H"),
        ("equal", "O", "O"),
        ("insert", None, "C"),
    ]


def test_alignment_invalid_character():
    with pytest.raises(ValueError):
        levenshtein_alignment("A|B", "AB")


# TESTS FOR COMPONENT ERROR COUNTS AND RATES

def test_component_tone_error():
    result = component_error_counts("MẶT", "MẮT")

    assert result["base_errors"] == 0
    assert result["modifier_errors"] == 0
    assert result["tone_errors"] == 1
    assert result["total"] == 3


def test_component_modifier_error():
    result = component_error_counts("a", "ă")

    assert result["base_errors"] == 0
    assert result["modifier_errors"] == 1
    assert result["tone_errors"] == 0


def test_component_base_error():
    result = component_error_counts("B", "A")

    assert result["base_errors"] == 1
    assert result["modifier_errors"] == 0
    assert result["tone_errors"] == 0


def test_component_delete():
    result = component_error_counts("HO", "HOC")

    assert result["base_errors"] == 1
    assert result["modifier_errors"] == 1
    assert result["tone_errors"] == 1


def test_component_insert():
    result = component_error_counts("HOC", "HO")

    assert result["base_errors"] == 1
    assert result["modifier_errors"] == 1
    assert result["tone_errors"] == 1


def test_component_error_rates():
    predictions = ["MẶT", "HOC", "CA"]
    ground_truths = ["MẮT", "HỌC", "CÁ"]

    result = component_error_rates(
        predictions, ground_truths
    )

    assert result["base_er"] == pytest.approx(0.0)
    assert result["modifier_er"] == pytest.approx(0.0)
    assert result["tone_er"] == pytest.approx(3 / 8)


# TESTS FOR TONE CONFUSION MATRIX
def test_tone_confusion_matrix():
    predictions = ["mà", "mã", "má", "mà"]
    ground_truths = ["má", "mả", "mạ", "mà"]

    matrix = tone_confusion_matrix(
        predictions,
        ground_truths,
    )

    assert matrix["acute"]["grave"] == 1
    assert matrix["hook"]["tilde"] == 1
    assert matrix["dot"]["acute"] == 1
    assert matrix["grave"]["grave"] == 1


def test_tone_confusion_modifier_mismatch():
    matrix = tone_confusion_matrix(
        predictions=["â"],
        ground_truths=["ắ"],
    )

    assert sum(
        sum(row.values())
        for row in matrix.values()
    ) == 0


def test_tone_confusion_invalid_marker():
    matrix = tone_confusion_matrix(
        predictions=["A|B"],
        ground_truths=["A|B"],
    )

    assert sum(
        sum(row.values())
        for row in matrix.values()
    ) == 0


def test_tone_confusion_nfc():
    matrix = tone_confusion_matrix(
        predictions=["a\u0300"],
        ground_truths=["á"],
    )

    assert matrix["acute"]["grave"] == 1
    
    
# TESTS FOR EDGE CASES 
def test_unicode_nfc_nfd_consistency():
    """NFC and NFD should produce identical evaluation results."""

    nfc = "Ắ"
    nfd = "A\u0306\u0301"

    assert normalize_text(nfc) == normalize_text(nfd)
    assert word_accuracy([nfd], [nfc]) == 1.0
    assert normalized_edit_similarity(nfd, nfc) == 1.0

    counts = component_error_counts(nfd, nfc)

    assert counts["base_errors"] == 0
    assert counts["modifier_errors"] == 0
    assert counts["tone_errors"] == 0


def test_spaces_consistency():
    """ASCII spaces are ignored across metrics."""

    pred = "HỌCSINH"
    gt = "HỌC SINH"

    assert word_accuracy([pred], [gt]) == 1.0
    assert one_minus_ned([pred], [gt]) == 1.0

    rates = component_error_rates([pred], [gt])

    assert rates["base_er"] == 0.0
    assert rates["modifier_er"] == 0.0
    assert rates["tone_er"] == 0.0


def test_single_tone_error_across_metrics():
    """One incorrect tone, but correct base and modifier."""

    pred = "MẶT"
    gt = "MẮT"

    assert word_accuracy([pred], [gt]) == 0.0
    assert normalized_edit_similarity(pred, gt) == pytest.approx(2 / 3)

    rates = component_error_rates([pred], [gt])

    assert rates["base_er"] == 0.0
    assert rates["modifier_er"] == 0.0
    assert rates["tone_er"] == pytest.approx(1 / 3)

    matrix = tone_confusion_matrix([pred], [gt])

    assert matrix["acute"]["dot"] == 1


def test_insertion_deletion_consistency():
    """Insertions and deletions are counted as component errors."""

    insert = component_error_counts("HOC", "HO")
    delete = component_error_counts("HO", "HOC")

    for key in ("base_errors", "modifier_errors", "tone_errors"):
        assert insert[key] == 1
        assert delete[key] == 1

    assert any(
        op == "insert"
        for op, _, _ in levenshtein_alignment("HOC", "HO")
    )

    assert any(
        op == "delete"
        for op, _, _ in levenshtein_alignment("HO", "HOC")
    )


def test_invalid_marker_consistency():
    """A crop containing | must never count as correct."""

    pred = "A|B"
    gt = "A|B"

    assert word_accuracy([pred], [gt]) == 0.0
    assert normalized_edit_similarity(pred, gt) == 0.0

    rates = component_error_rates([pred], [gt])

    assert rates["base_er"] == 1.0
    assert rates["modifier_er"] == 1.0
    assert rates["tone_er"] == 1.0

    matrix = tone_confusion_matrix([pred], [gt])

    assert sum(sum(row.values()) for row in matrix.values()) == 0


def test_empty_prediction():
    """An empty prediction should count as missing characters."""

    pred = ""
    gt = "MẮT"

    assert word_accuracy([pred], [gt]) == 0.0
    assert normalized_edit_similarity(pred, gt) == 0.0

    rates = component_error_rates([pred], [gt])

    assert rates["base_er"] == 1.0
    assert rates["modifier_er"] == 1.0
    assert rates["tone_er"] == 1.0


def test_unsupported_character():
    """Unsupported characters should not be silently discarded."""

    with pytest.raises(ValueError):
        component_error_counts("°", "A")
        
        
# TESTS FOR EVALUATE_PREDICTIONS
def test_evaluate_perfect_predictions():
    """TEST 1: All predictions are perfectly correct."""

    gt = ["MẮT", "VIỆT NAM", "ĐƯỜNG"]
    pred = ["MẮT", "VIỆTNAM", "ĐƯỜNG"]

    result = evaluate_predictions(pred, gt)
    rates = result["component_error_rates"]
    matrix = result["tone_confusion_matrix"]

    assert result["word_accuracy"] == 1.0
    assert result["one_minus_ned"] == 1.0

    assert rates["base_er"] == 0.0
    assert rates["modifier_er"] == 0.0
    assert rates["tone_er"] == 0.0

    # All counted tone predictions must be correct.
    for gt_tone, row in matrix.items():
        for pred_tone, count in row.items():
            if gt_tone != pred_tone:
                assert count == 0


def test_evaluate_tone_only_error():
    """TEST 2: Wrong tone, correct base and modifier."""

    result = evaluate_predictions(["MẶT"], ["MẮT"])

    rates = result["component_error_rates"]
    matrix = result["tone_confusion_matrix"]

    assert result["word_accuracy"] == 0.0
    assert result["one_minus_ned"] == pytest.approx(2 / 3)

    assert rates["base_er"] == 0.0
    assert rates["modifier_er"] == 0.0
    assert rates["tone_er"] == pytest.approx(1 / 3)

    assert matrix["acute"]["dot"] == 1


def test_evaluate_modifier_only_error():
    """TEST 3: Wrong modifier, correct base and tone."""

    result = evaluate_predictions(["a"], ["ă"])

    rates = result["component_error_rates"]
    matrix = result["tone_confusion_matrix"]

    assert result["word_accuracy"] == 0.0
    assert result["one_minus_ned"] == 0.0

    assert rates["base_er"] == 0.0
    assert rates["modifier_er"] == 1.0
    assert rates["tone_er"] == 0.0

    # Base/Modifier mismatch: exclude this pair
    # from tone confusion analysis.
    assert sum(
        sum(row.values())
        for row in matrix.values()
    ) == 0


def test_evaluate_base_only_error():
    """TEST 4: Wrong base, correct modifier and tone."""

    result = evaluate_predictions(["B"], ["A"])

    rates = result["component_error_rates"]

    assert result["word_accuracy"] == 0.0
    assert result["one_minus_ned"] == 0.0

    assert rates["base_er"] == 1.0
    assert rates["modifier_er"] == 0.0
    assert rates["tone_er"] == 0.0

    assert sum(
        sum(row.values())
        for row in result["tone_confusion_matrix"].values()
    ) == 0


def test_evaluate_insertion_error():
    """TEST 5: Prediction contains one extra character."""

    result = evaluate_predictions(["HOC"], ["HO"])

    rates = result["component_error_rates"]
    matrix = result["tone_confusion_matrix"]

    assert result["word_accuracy"] == 0.0
    assert result["one_minus_ned"] == pytest.approx(2 / 3)

    assert rates["base_er"] == pytest.approx(1 / 3)
    assert rates["modifier_er"] == pytest.approx(1 / 3)
    assert rates["tone_er"] == pytest.approx(1 / 3)

    # H and O are correct; inserted C is excluded.
    assert matrix["level"]["level"] == 2


def test_evaluate_deletion_error():
    """TEST 6: Prediction is missing one character."""

    result = evaluate_predictions(["HO"], ["HOC"])

    rates = result["component_error_rates"]
    matrix = result["tone_confusion_matrix"]

    assert result["word_accuracy"] == 0.0
    assert result["one_minus_ned"] == pytest.approx(2 / 3)

    assert rates["base_er"] == pytest.approx(1 / 3)
    assert rates["modifier_er"] == pytest.approx(1 / 3)
    assert rates["tone_er"] == pytest.approx(1 / 3)

    assert matrix["level"]["level"] == 2


def test_evaluate_unicode_and_spaces():
    """TEST 7: NFC, NFD and spaces must be consistent."""

    gt = ["Ắ", "HỌC SINH", "CÀ PHÊ"]

    pred = [
        "A\u0306\u0301",
        "HỌCSINH",
        "CÀPHÊ",
    ]

    result = evaluate_predictions(pred, gt)
    rates = result["component_error_rates"]

    assert result["word_accuracy"] == 1.0
    assert result["one_minus_ned"] == 1.0

    assert rates["base_er"] == 0.0
    assert rates["modifier_er"] == 0.0
    assert rates["tone_er"] == 0.0


def test_evaluate_invalid_marker():
    """TEST 8: A crop containing | must be penalized."""

    gt = ["A|B"]
    pred = ["A|B"]

    result = evaluate_predictions(pred, gt)
    rates = result["component_error_rates"]
    matrix = result["tone_confusion_matrix"]

    assert result["word_accuracy"] == 0.0
    assert result["one_minus_ned"] == 0.0

    assert rates["base_er"] == 1.0
    assert rates["modifier_er"] == 1.0
    assert rates["tone_er"] == 1.0

    # Invalid samples do not enter the matrix.
    assert sum(
        sum(row.values())
        for row in matrix.values()
    ) == 0


def test_evaluate_unsupported_character():
    """TEST 9: Unsupported character during replacement."""

    with pytest.raises(ValueError):
        evaluate_predictions(
            predictions=["°"],
            ground_truths=["A"],
        )


def test_evaluate_mixed_dataset():
    """TEST 10: Integration test with multiple error types."""

    gt = [
        "MẮT",       # Tone error
        "HỌC",       # Tone error
        "CÁ",        # Tone error
        "HỌC SINH",  # Correct after normalization
        "A|B",       # Invalid marker
    ]

    pred = [
        "MẶT",
        "HOC",
        "CA",
        "HỌCSINH",
        "A|B",
    ]

    result = evaluate_predictions(pred, gt)

    rates = result["component_error_rates"]
    matrix = result["tone_confusion_matrix"]

    # Exactly 1 out of 5 word crops is correct.
    assert result["word_accuracy"] == pytest.approx(1 / 5)

    # (2/3 + 2/3 + 1/2 + 1 + 0) / 5
    assert result["one_minus_ned"] == pytest.approx(17 / 30)

    # Total aligned positions:
    # 3 + 3 + 2 + 7 + 3 = 18
    #
    # Invalid crop contributes 3 errors to each
    # component. Three other crops have tone errors.

    assert rates["base_er"] == pytest.approx(3 / 18)
    assert rates["modifier_er"] == pytest.approx(3 / 18)
    assert rates["tone_er"] == pytest.approx(6 / 18)

    # Check specific tone confusions.
    assert matrix["acute"]["dot"] == 1
    assert matrix["dot"]["level"] == 1
    assert matrix["acute"]["level"] == 1

    # Invalid crop excluded from confusion matrix.
    assert sum(
        sum(row.values())
        for row in matrix.values()
    ) == 15
