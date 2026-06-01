from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.ingest import CsvDataset


class CsvRowSource:
    """FeatureSource that serves a single row from a CSV dataset.

    Useful for offline inference and demos — no camera required.
    Implements the ``FeatureSource`` protocol without inheriting from it.

    Args:
        csv_path: Path to a raw PJM points CSV file.
        row_index: Zero-based index of the row to use as the feature vector.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        IndexError: If ``row_index`` is out of range for the dataset.
    """

    def __init__(self, csv_path: Path, row_index: int = 0) -> None:
        dataset = CsvDataset(csv_path)
        if row_index < 0 or row_index >= len(dataset):
            raise IndexError(
                f"row_index {row_index} is out of range for dataset "
                f"of {len(dataset)} samples."
            )
        vectors = list(dataset.iter_samples())
        self._vector: np.ndarray = vectors[row_index][0]
        self._label: str = vectors[row_index][1]

    @property
    def label(self) -> str:
        """Return the true sign label for this row (useful for verification)."""
        return self._label

    def read_features(self) -> np.ndarray:
        """Return the stored feature vector of shape ``(234,)``.

        Returns:
            float32 array of shape ``(234,)``.
        """
        return self._vector

    def release(self) -> None:
        """No-op — CSV data is held in memory; nothing to release."""
