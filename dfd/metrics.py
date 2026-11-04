"""Recognition metric interfaces (implementation pending).

NFC-normalized, diacritic-sensitive and case-insensitive main scoring.
Report word accuracy (full and diacritic-heavy), 1-NED, aligned component ER,
insertions/deletions separately, and a tone confusion matrix. Freeze alignment
tie-breaking, denominators and heavy-word definitions before experiments.
"""


def evaluate_predictions(predictions, labels, vocabulary):
    raise NotImplementedError("Implement shared Levenshtein alignment and component metrics before baseline training")
