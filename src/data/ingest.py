from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

LABEL_COL = "sign_label"
META_COLS = ("user_id", "lux_value")
EXPECTED_FEATURE_DIM = 234


class CsvDataset:
    """PJM points CSV dataset implementing the DatasetSource protocol.

    Loads a raw CSV file produced by the Kaggle sign-language-pjm dataset,
    strips metadata columns, and exposes labelled feature vectors.

    Args:
        csv_path: Path to the CSV file (e.g. ``data/raw/550-points.csv``).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If ``sign_label`` column is missing or feature count is wrong.
    """

    def __init__(self, csv_path: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        df = pd.read_csv(csv_path).drop(columns=list(META_COLS), errors="ignore")

        if LABEL_COL not in df.columns:
            raise ValueError(
                f"Required column '{LABEL_COL}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

        feature_cols = [c for c in df.columns if c != LABEL_COL]
        if len(feature_cols) != EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_DIM} feature columns, "
                f"got {len(feature_cols)} in {csv_path}."
            )

        self._features: np.ndarray = df[feature_cols].to_numpy(dtype=np.float32)
        self._labels: np.ndarray = df[LABEL_COL].to_numpy()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._labels)

    def iter_samples(self) -> Iterator[tuple[np.ndarray, str]]:
        """Yield one (feature_vector, label) pair per sample.

        Yields:
            A tuple of:
                - feature_vector: float32 array of shape ``(234,)``
                - label: non-empty Polish sign label string (e.g. ``"A"``)
        """
        for features, label in zip(self._features, self._labels):
            yield features, str(label)
