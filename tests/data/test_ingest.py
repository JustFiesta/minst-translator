from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.ingest import EXPECTED_FEATURE_DIM, CsvDataset


def test_csv_dataset_loads_real_file(tmp_path: Path) -> None:
    # Arrange
    csv_content = _make_csv(n_rows=5)
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    # Act
    ds = CsvDataset(csv_file)

    # Assert
    assert len(ds) == 5


def test_csv_dataset_iter_samples_yields_correct_shape(tmp_path: Path) -> None:
    # Arrange
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(_make_csv(n_rows=3))
    ds = CsvDataset(csv_file)

    # Act
    samples = list(ds.iter_samples())

    # Assert
    assert len(samples) == 3
    for vector, label in samples:
        assert vector.shape == (EXPECTED_FEATURE_DIM,)
        assert vector.dtype == np.float32
        assert isinstance(label, str)
        assert len(label) > 0


def test_csv_dataset_drops_metadata_columns(tmp_path: Path) -> None:
    # Arrange
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(_make_csv(n_rows=2))
    ds = CsvDataset(csv_file)

    # Act
    vectors = [v for v, _ in ds.iter_samples()]

    # Assert — feature count must equal EXPECTED_FEATURE_DIM (metadata dropped)
    assert all(v.shape == (EXPECTED_FEATURE_DIM,) for v in vectors)


def test_csv_dataset_raises_on_missing_file() -> None:
    # Arrange
    missing = Path("data/raw/does_not_exist.csv")

    # Act + Assert
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        CsvDataset(missing)


def test_csv_dataset_raises_on_missing_label_column(tmp_path: Path) -> None:
    # Arrange — CSV without sign_label column
    header = "user_id,lux_value," + ",".join(f"feat_{i}" for i in range(234))
    row = "1,550," + ",".join("0.0" for _ in range(234))
    csv_file = tmp_path / "no_label.csv"
    csv_file.write_text(f"{header}\n{row}\n")

    # Act + Assert
    with pytest.raises(ValueError, match="sign_label"):
        CsvDataset(csv_file)


def test_csv_dataset_raises_on_wrong_feature_count(tmp_path: Path) -> None:
    # Arrange — only 10 feature columns instead of 234
    header = "user_id,lux_value,sign_label," + ",".join(f"feat_{i}" for i in range(10))
    row = "1,550,A," + ",".join("0.0" for _ in range(10))
    csv_file = tmp_path / "short.csv"
    csv_file.write_text(f"{header}\n{row}\n")

    # Act + Assert
    with pytest.raises(ValueError, match="234"):
        CsvDataset(csv_file)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(n_rows: int) -> str:
    """Build a minimal valid CSV string with EXPECTED_FEATURE_DIM feature cols."""
    feature_cols = ",".join(f"feat_{i}" for i in range(EXPECTED_FEATURE_DIM))
    header = f"user_id,lux_value,sign_label,{feature_cols}"
    rows = [
        f"{i},550,A," + ",".join(str(float(i)) for _ in range(EXPECTED_FEATURE_DIM))
        for i in range(n_rows)
    ]
    return "\n".join([header] + rows) + "\n"
