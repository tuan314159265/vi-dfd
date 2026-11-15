"""Recognition metric interfaces (implementation pending).

NFC-normalized, diacritic-sensitive and case-insensitive main scoring.
Report word accuracy (full and diacritic-heavy), 1-NED, aligned component ER,
insertions/deletions separately, and a tone confusion matrix. Freeze alignment
tie-breaking, denominators and heavy-word definitions before experiments.
"""
import unicodedata
from rapidfuzz.distance import Levenshtein
from dfd.vietchar import decompose
from dfd.vietchar import TONES


def evaluate_predictions(predictions, labels, vocabulary):
    raise NotImplementedError("Implement shared Levenshtein alignment and component metrics before baseline training")


INVALID_CHARACTER = "|"


def normalize_text(text: str) -> str:
    """
    Normalize Vietnamese text before evaluation.

    - Unicode NFC
    - Remove ASCII spaces
    - Preserve Vietnamese diacritics
    """
    if not isinstance(text, str):
        raise TypeError("Expected a string")

    text = unicodedata.normalize("NFC", text)

    return text.replace(" ", "")


def contains_invalid_character(text: str) -> bool:
    """Check whether a transcript contains the invalid marker."""
    return INVALID_CHARACTER in text



def word_accuracy(
    predictions: list[str],
    ground_truths: list[str],
) -> float:
    """
    Compute word-level accuracy.

    Rules:
    - Normalize Unicode NFC.
    - Remove ASCII spaces.
    - Any pair containing '|' is considered incorrect.
    - Return a value between 0.0 and 1.0.
    """

    if len(predictions) != len(ground_truths):
        raise ValueError(
            "Predictions and ground truths must have the same length"
        )

    if not ground_truths:
        raise ValueError("Cannot evaluate an empty dataset")

    correct = 0

    for pred, gt in zip(predictions, ground_truths):
        pred = normalize_text(pred)
        gt = normalize_text(gt)

        # Invalid marker: always count as incorrect
        if (
            contains_invalid_character(pred)
            or contains_invalid_character(gt)
        ):
            continue

        if pred == gt:
            correct += 1

    return correct / len(ground_truths)



def normalized_edit_similarity(
    prediction: str,
    ground_truth: str,
) -> float:
    """
    Compute 1-NED for a single prediction/ground-truth pair.

    Returns a value between 0.0 and 1.0.
    """

    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)

    # Invalid marker: treat the sample as incorrect.
    if (
        contains_invalid_character(pred)
        or contains_invalid_character(gt)
    ):
        return 0.0

    # Two empty strings are considered identical.
    if not pred and not gt:
        return 1.0

    edit_distance = Levenshtein.distance(pred, gt)

    max_length = max(len(pred), len(gt))

    similarity = 1.0 - edit_distance / max_length

    return similarity


def one_minus_ned(
    predictions: list[str],
    ground_truths: list[str],
) -> float:
    """
    Compute dataset-level 1-NED.

    Each word crop contributes equally.
    """

    if len(predictions) != len(ground_truths):
        raise ValueError(
            "Predictions and ground truths must have the same length"
        )

    if not ground_truths:
        raise ValueError("Cannot evaluate an empty dataset")

    total_similarity = 0.0

    for pred, gt in zip(predictions, ground_truths):
        total_similarity += normalized_edit_similarity(pred, gt)

    return total_similarity / len(ground_truths)



def levenshtein_alignment(
    prediction: str,
    ground_truth: str,
) -> list[tuple[str, str | None, str | None]]:
    """
    Align characters between ground truth and prediction.

    Returns:
        List of (operation, gt_char, pred_char)

    Operations:
        equal   : characters match
        replace : characters differ
        insert  : extra character in prediction
        delete  : missing character in prediction
    """

    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)

    if (
        contains_invalid_character(pred)
        or contains_invalid_character(gt)
    ):
        raise ValueError(
            "Cannot align transcripts containing invalid marker '|'"
        )

    operations = Levenshtein.opcodes(gt, pred)

    alignment = []

    for op, gt_start, gt_end, pred_start, pred_end in operations:

        if op in ("equal", "replace"):
            for gt_char, pred_char in zip(
                gt[gt_start:gt_end],
                pred[pred_start:pred_end],
            ):
                alignment.append((op, gt_char, pred_char))

        elif op == "insert":
            for pred_char in pred[pred_start:pred_end]:
                alignment.append(("insert", None, pred_char))

        elif op == "delete":
            for gt_char in gt[gt_start:gt_end]:
                alignment.append(("delete", gt_char, None))

    return alignment



def component_error_counts(
    prediction: str,
    ground_truth: str,
) -> dict[str, int]:
    """
    Count Base, Modifier and Tone errors for one word crop.
    
    Operation	Base Error	Modifier Error	Tone Error
    Equal	    0	        0	            0
    Replace	    Compare     Compare         compare
    Insert	    1	        1	            1
    Delete	    1	        1	            1
    """

    pred = normalize_text(prediction)
    gt = normalize_text(ground_truth)

    # Number of aligned character positions used as denominator
    total = max(len(pred), len(gt), 1)

    # Team policy: a crop containing | is always incorrect.
    if (
        contains_invalid_character(pred)
        or contains_invalid_character(gt)
    ):
        return {
            "base_errors": total,
            "modifier_errors": total,
            "tone_errors": total,
            "total": total,
        }

    alignment = levenshtein_alignment(pred, gt)

    base_errors = 0
    modifier_errors = 0
    tone_errors = 0

    for operation, gt_char, pred_char in alignment:

        if operation == "equal":
            continue

        if operation in ("insert", "delete"):
            base_errors += 1
            modifier_errors += 1
            tone_errors += 1
            continue

        # operation == "replace"
        gt_parts = decompose(gt_char)
        pred_parts = decompose(pred_char)

        if gt_parts.base != pred_parts.base:
            base_errors += 1

        if gt_parts.modifier != pred_parts.modifier:
            modifier_errors += 1

        if gt_parts.tone != pred_parts.tone:
            tone_errors += 1

    return {
        "base_errors": base_errors,
        "modifier_errors": modifier_errors,
        "tone_errors": tone_errors,
        "total": total,
    }



def component_error_rates(
    predictions: list[str],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Compute dataset-level Base, Modifier and Tone Error Rates.
    """

    if len(predictions) != len(ground_truths):
        raise ValueError(
            "Predictions and ground truths must have the same length"
        )

    if not ground_truths:
        raise ValueError("Cannot evaluate an empty dataset")

    total_base_errors = 0
    total_modifier_errors = 0
    total_tone_errors = 0
    total_positions = 0

    for pred, gt in zip(predictions, ground_truths):

        counts = component_error_counts(pred, gt)

        total_base_errors += counts["base_errors"]
        total_modifier_errors += counts["modifier_errors"]
        total_tone_errors += counts["tone_errors"]
        total_positions += counts["total"]

    return {
        "base_er": total_base_errors / total_positions,
        "modifier_er": total_modifier_errors / total_positions,
        "tone_er": total_tone_errors / total_positions,
    }



def tone_confusion_matrix(
    predictions: list[str],
    ground_truths: list[str],
) -> dict[str, dict[str, int]]:
    """
    Build a 6x6 tone confusion matrix.

    Rows: Ground Truth tone
    Columns: Prediction tone

    Only compare aligned characters with
    matching Base and Modifier.
    """

    if len(predictions) != len(ground_truths):
        raise ValueError(
            "Predictions and ground truths must have the same length"
        )

    if not ground_truths:
        raise ValueError("Cannot evaluate an empty dataset")

    matrix = {
        gt_tone: {
            pred_tone: 0
            for pred_tone in TONES
        }
        for gt_tone in TONES
    }

    for pred, gt in zip(predictions, ground_truths):
        pred = normalize_text(pred)
        gt = normalize_text(gt)

        # Skip invalid crops
        if (
            contains_invalid_character(pred)
            or contains_invalid_character(gt)
        ):
            continue

        alignment = levenshtein_alignment(pred, gt)

        for operation, gt_char, pred_char in alignment:

            if operation in ("insert", "delete"):
                continue

            gt_parts = decompose(gt_char)
            pred_parts = decompose(pred_char)

            # Analyze tone independently of Base/Modifier
            if (
                gt_parts.base != pred_parts.base
                or gt_parts.modifier != pred_parts.modifier
            ):
                continue

            matrix[gt_parts.tone][pred_parts.tone] += 1

    return matrix


def evaluate_predictions(
    predictions: list[str],
    ground_truths: list[str],
) -> dict:
    """
    Evaluate STR predictions using all implemented metrics.
    """

    return {
        "word_accuracy": word_accuracy(
            predictions, ground_truths
        ),
        "one_minus_ned": one_minus_ned(
            predictions, ground_truths
        ),
        "component_error_rates": component_error_rates(
            predictions, ground_truths
        ),
        "tone_confusion_matrix": tone_confusion_matrix(
            predictions, ground_truths
        ),
    }
