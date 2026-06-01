from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.base import ClassifierMixin

from src.inference.protocol import FeatureSource


def load_model(path: Path) -> ClassifierMixin:
    """Load a serialised scikit-learn classifier from disk.

    Args:
        path: Path to a ``.pkl`` file produced by ``train_and_save``.

    Returns:
        The deserialised classifier.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. "
            "Run 'uv run python -m src.model.train' first."
        )
    return joblib.load(path)


def predict_sign(clf: ClassifierMixin, source: FeatureSource) -> str:
    """Classify one feature vector obtained from ``source``.

    Reads a single 234-dim vector from ``source``, passes it through the
    classifier, and returns the predicted sign label.  The caller is
    responsible for calling ``source.release()`` when done.

    Args:
        clf: A fitted scikit-learn classifier (loaded via ``load_model``).
        source: Any object satisfying the ``FeatureSource`` protocol.

    Returns:
        Predicted sign label as a Polish string (e.g. ``"A"``).
    """
    vector = source.read_features()
    return str(clf.predict(vector.reshape(1, -1))[0])
