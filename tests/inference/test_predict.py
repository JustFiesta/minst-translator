from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from src.inference.predict import load_model, predict_sign


def test_predict_sign_returns_string(trained_clf, mock_feature_source) -> None:
    # Arrange — trained_clf and mock_feature_source come from conftest

    # Act
    result = predict_sign(trained_clf, mock_feature_source)

    # Assert
    assert isinstance(result, str)
    assert len(result) > 0


def test_predict_sign_returns_known_label(trained_clf, mock_feature_source) -> None:
    # Arrange
    known_labels = {"A", "B", "C"}

    # Act
    result = predict_sign(trained_clf, mock_feature_source)

    # Assert — prediction must be one of the training labels
    assert result in known_labels


def test_predict_sign_does_not_call_release(trained_clf, mock_feature_source) -> None:
    # Arrange — caller is responsible for release; predict_sign must NOT call it

    # Act
    predict_sign(trained_clf, mock_feature_source)

    # Assert
    assert not mock_feature_source.released


def test_load_model_returns_classifier(tmp_path: Path, trained_clf) -> None:
    # Arrange — save clf to tmp file, then reload
    model_path = tmp_path / "model.pkl"
    joblib.dump(trained_clf, model_path)

    # Act
    loaded = load_model(model_path)

    # Assert — loaded object has predict method
    assert hasattr(loaded, "predict")


def test_load_model_raises_on_missing_file() -> None:
    # Arrange
    missing = Path("artifacts/nonexistent_model.pkl")

    # Act + Assert
    with pytest.raises(FileNotFoundError, match="nonexistent_model"):
        load_model(missing)
