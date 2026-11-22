"""Shared Vietnamese STR evaluation metrics.

Protocol (experiment plan, decision 4, 16 Nov 2026):
- NFC, remove ASCII spaces, lowercase (retain Vietnamese diacritics).
- Word accuracy and 1-NED score *every* crop; U+FFFD always makes it wrong.
- Levenshtein.opcodes(ground_truth, prediction) for alignment.
- Base/modifier/tone errors count only component mismatches in REPLACE pairs.
- Insertions and deletions are reported separately, never folded into component ER.
- Micro denominator is number of normalized ground-truth characters in the split.
- '|' is valid vocabulary punctuation; U+FFFD is the invalid marker.

An empty corpus, or a corpus with zero ground-truth characters for component
rates, has no meaningful component-ER denominator and raises ValueError.
"""

import unicodedata

from rapidfuzz.distance import Levenshtein
from dfd.vietchar import TONES, decompose

INVALID_CHARACTER = "\ufffd"


def normalize_text(text: str) -> str:
    """NFC + ignore ASCII spaces + lowercase, preserving diacritics."""
    if not isinstance(text, str):
        raise TypeError("Expected a string")
    return unicodedata.normalize("NFC", text).replace(" ", "").lower()


def contains_invalid_character(text: str) -> bool:
    """True only for the U+FFFD invalid-decoding sentinel."""
    return INVALID_CHARACTER in text


def _validate_pairs(predictions: list[str], ground_truths: list[str]) -> None:
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    if not ground_truths:
        raise ValueError("Cannot evaluate an empty dataset")


def word_accuracy(predictions: list[str], ground_truths: list[str]) -> float:
    """Fraction of exactly correct normalized words; U+FFFD is always wrong."""
    _validate_pairs(predictions, ground_truths)
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred, gt = normalize_text(pred), normalize_text(gt)
        if not (contains_invalid_character(pred) or contains_invalid_character(gt)) and pred == gt:
            correct += 1
    return correct / len(ground_truths)


def normalized_edit_similarity(prediction: str, ground_truth: str) -> float:
    """Normalized Levenshtein similarity for one word (1-NED)."""
    pred, gt = normalize_text(prediction), normalize_text(ground_truth)
    if contains_invalid_character(pred) or contains_invalid_character(gt):
        return 0.0
    max_length = max(len(pred), len(gt))
    return 1.0 if max_length == 0 else 1 - Levenshtein.distance(gt, pred) / max_length


def one_minus_ned(predictions: list[str], ground_truths: list[str]) -> float:
    """Word-macro average of normalized Levenshtein similarity."""
    _validate_pairs(predictions, ground_truths)
    return sum(normalized_edit_similarity(p, g) for p, g in zip(predictions, ground_truths)) / len(ground_truths)


def levenshtein_alignment(
    prediction: str, ground_truth: str
) -> list[tuple[str, str | None, str | None]]:
    """Alignment GT -> prediction, as (op, gt_char, pred_char).

    'insert' means extra character in prediction; 'delete' means missing
    character in prediction. U+FFFD is intentionally allowed in the alignment.
    """
    pred, gt = normalize_text(prediction), normalize_text(ground_truth)
    alignment = []
    for op, gt_start, gt_end, pred_start, pred_end in Levenshtein.opcodes(gt, pred):
        if op in ("equal", "replace"):
            gt_chunk, pred_chunk = gt[gt_start:gt_end], pred[pred_start:pred_end]
            if len(gt_chunk) != len(pred_chunk):
                raise RuntimeError("Unexpected unequal opcode span")
            alignment.extend((op, a, b) for a, b in zip(gt_chunk, pred_chunk))
        elif op == "insert":
            alignment.extend(("insert", None, c) for c in pred[pred_start:pred_end])
        elif op == "delete":
            alignment.extend(("delete", c, None) for c in gt[gt_start:gt_end])
        else:
            raise RuntimeError(f"Unknown Levenshtein opcode: {op}")
    return alignment


def _replacement_error_flags(gt_char: str, pred_char: str) -> tuple[int, int, int]:
    """Compare components, with any U+FFFD position counting as 3 errors."""
    if INVALID_CHARACTER in (gt_char, pred_char):
        return (1, 1, 1)
    gt_parts, pred_parts = decompose(gt_char), decompose(pred_char)
    return (
        int(gt_parts.base != pred_parts.base),
        int(gt_parts.modifier != pred_parts.modifier),
        int(gt_parts.tone != pred_parts.tone),
    )


def component_error_counts(prediction: str, ground_truth: str) -> dict[str, int]:
    """Count per-word substitution component errors, insertions and deletions.

    'total' means the normalized GT length, NOT max(pred_len, gt_len).
    A U+FFFD at an aligned position is penalized in all 3 components, including
    when both sides are U+FFFD. Insert/delete remain separate.
    """
    gt = normalize_text(ground_truth)
    counts = dict(base_errors=0, modifier_errors=0, tone_errors=0,
                  insertions=0, deletions=0, total=len(gt))
    for op, gt_char, pred_char in levenshtein_alignment(prediction, ground_truth):
        if op == "insert":
            counts["insertions"] += 1
        elif op == "delete":
            counts["deletions"] += 1
        elif op == "equal":
            if gt_char == INVALID_CHARACTER:
                counts["base_errors"] += 1
                counts["modifier_errors"] += 1
                counts["tone_errors"] += 1
        elif op == "replace":
            b, m, t = _replacement_error_flags(gt_char, pred_char)
            counts["base_errors"] += b
            counts["modifier_errors"] += m
            counts["tone_errors"] += t
    return counts


def component_error_rates(predictions: list[str], ground_truths: list[str]) -> dict[str, float]:
    """Micro averaged ER; denominator = total normalized GT characters."""
    _validate_pairs(predictions, ground_truths)
    totals = {k: 0 for k in ("base_errors", "modifier_errors", "tone_errors", "insertions", "deletions", "total")}
    for pred, gt in zip(predictions, ground_truths):
        counts = component_error_counts(pred, gt)
        for key in totals:
            totals[key] += counts[key]
    denominator = totals["total"]
    if denominator == 0:
        raise ValueError("Component ER undefined when ground truths contain zero characters")
    return {
        "base_er": totals["base_errors"] / denominator,
        "modifier_er": totals["modifier_errors"] / denominator,
        "tone_er": totals["tone_errors"] / denominator,
        "insertion_rate": totals["insertions"] / denominator,
        "deletion_rate": totals["deletions"] / denominator,
    }


def tone_confusion_matrix(
    predictions: list[str], ground_truths: list[str]
) -> dict[str, dict[str, int]]:
    """6x6 counts: rows GT tone, columns predicted tone.

    Counts aligned, decomposable pairs with equal base and modifier. Excludes
    insertion/deletion and U+FFFD positions; does not discard an entire crop.
    """
    _validate_pairs(predictions, ground_truths)
    matrix = {gt: {pred: 0 for pred in TONES} for gt in TONES}
    for pred, gt in zip(predictions, ground_truths):
        for op, gt_char, pred_char in levenshtein_alignment(pred, gt):
            if op in ("insert", "delete") or INVALID_CHARACTER in (gt_char, pred_char):
                continue
            gt_parts, pred_parts = decompose(gt_char), decompose(pred_char)
            if gt_parts.base == pred_parts.base and gt_parts.modifier == pred_parts.modifier:
                matrix[gt_parts.tone][pred_parts.tone] += 1
    return matrix


def evaluate_predictions(
    predictions: list[str], ground_truths: list[str], vocabulary=None
) -> dict:
    """All shared metrics for pre-decoded text.

    Optional ``vocabulary`` is accepted for compatibility with an earlier
    3-argument placeholder; it is not used to transform already-decoded text.
    """
    _validate_pairs(predictions, ground_truths)
    return {
        "word_accuracy": word_accuracy(predictions, ground_truths),
        "one_minus_ned": one_minus_ned(predictions, ground_truths),
        "component_error_rates": component_error_rates(predictions, ground_truths),
        "tone_confusion_matrix": tone_confusion_matrix(predictions, ground_truths),
    }
