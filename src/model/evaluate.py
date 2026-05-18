from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report


def evaluate(
    clf: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    """Evaluate a trained classifier on the test set and print a report.

    Args:
        clf: A fitted scikit-learn classifier with a ``predict`` method.
        X_test: Feature matrix of shape ``(n_samples, 234)``.
        y_test: True label array of shape ``(n_samples,)``.

    Returns:
        A dict with keys:
            - ``"accuracy"``: overall accuracy as a float
            - ``"report"``: full per-class classification report as a string

    Raises:
        ValueError: If ``X_test`` and ``y_test`` have different lengths.
    """
    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test must have the same length, "
            f"got {len(X_test)} and {len(y_test)}."
        )

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    return {"accuracy": accuracy, "report": report}
