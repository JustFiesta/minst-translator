from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest
from sklearn.svm import SVC

from src.data.protocol import DatasetSource

FEATURE_DIM = 234
N_SAMPLES = 30
LABELS = ["A", "B", "C"]


class _SyntheticDataset:
    """Minimal DatasetSource with synthetic data — never touches real CSV."""

    def __init__(self, n_samples: int = N_SAMPLES, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._X = rng.random((n_samples, FEATURE_DIM)).astype(np.float32)
        self._y = np.array(
            [LABELS[i % len(LABELS)] for i in range(n_samples)]
        )

    def __len__(self) -> int:
        return len(self._y)

    def iter_samples(self) -> Iterator[tuple[np.ndarray, str]]:
        for features, label in zip(self._X, self._y):
            yield features, str(label)


class _MockFeatureSource:
    """FeatureSource that always returns a zero vector."""

    def __init__(self, vector: np.ndarray | None = None) -> None:
        self._vector = (
            vector if vector is not None else np.zeros(FEATURE_DIM, dtype=np.float32)
        )
        self.released = False

    def read_features(self) -> np.ndarray:
        return self._vector

    def release(self) -> None:
        self.released = True


@pytest.fixture
def synthetic_dataset() -> DatasetSource:
    """Synthetic DatasetSource with 30 samples across 3 classes."""
    return _SyntheticDataset()


@pytest.fixture
def sample_X_y() -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix (30, 234) and label array (30,) built from synthetic data."""
    ds = _SyntheticDataset()
    X = np.stack([v for v, _ in ds.iter_samples()])
    y = np.array([lbl for _, lbl in ds.iter_samples()])
    return X, y


@pytest.fixture
def mock_feature_source() -> _MockFeatureSource:
    """FeatureSource stub returning a zero vector; exposes ``released`` flag."""
    return _MockFeatureSource()


@pytest.fixture
def trained_clf(sample_X_y: tuple[np.ndarray, np.ndarray]) -> SVC:
    """Tiny SVC fitted on synthetic data — fast, no disk I/O."""
    X, y = sample_X_y
    clf = SVC(kernel="rbf")
    clf.fit(X, y)
    return clf
