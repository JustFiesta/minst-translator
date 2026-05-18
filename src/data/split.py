from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into stratified train / val / test subsets.

    Performs two successive stratified splits to produce an 80 / 10 / 10
    partition by default.  All splits preserve the class distribution of ``y``.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Label array of shape ``(n_samples,)``.
        val_ratio: Fraction of total data reserved for validation.
        test_ratio: Fraction of total data reserved for testing.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(X_train, X_val, X_test, y_train, y_val, y_test)``.

    Raises:
        ValueError: If ``X`` and ``y`` have different numbers of samples, or if
            ratios leave no room for the training set.
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples, "
            f"got X={len(X)} and y={len(y)}."
        )

    holdout = val_ratio + test_ratio
    if not 0 < holdout < 1:
        raise ValueError(
            f"val_ratio + test_ratio must be between 0 and 1, got {holdout}."
        )

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=holdout, stratify=y, random_state=seed
    )

    relative_test = test_ratio / holdout
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_test, stratify=y_tmp, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
