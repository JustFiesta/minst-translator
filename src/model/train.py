from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import ClassifierMixin

from src.model.classifiers.cnn import train_and_save_cnn
from src.model.classifiers.random_forest import build_random_forest_classifier
from src.model.classifiers.svc import build_svc_classifier

DEFAULT_ARTIFACT = Path("artifacts/svc_model.pkl")

CLASSIFIERS = {
    "svc": build_svc_classifier,
    "rf": build_random_forest_classifier,
    "cnn": None,
}


def build_classifier(name: str) -> ClassifierMixin:
    """Instantiate a classifier by name.

    Args:
        name: One of ``"svc"``, ``"rf"``, ``"cnn"``.

    Returns:
        Unfitted sklearn estimator.

    Raises:
        ValueError: If ``name`` is not a recognised classifier key.
    """
    if name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown sklearn classifier '{name}'. Choose from: ['svc', 'rf', 'cnn']"
        )
    return CLASSIFIERS[name]()


def train_and_save(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: Path = DEFAULT_ARTIFACT,
    classifier: str = "svc",
) -> None:
    """Train a classifier and persist it to disk.

    Args:
        X_train: Feature matrix of shape ``(n_samples, 784)``.
        y_train: Label array of shape ``(n_samples,)``.
        output_path: Destination ``.pkl`` path for the serialised model.
        classifier: Classifier name — ``"svc"``, ``"rf"``, or ``"cnn"``.

    Raises:
        ValueError: If ``X_train`` and ``y_train`` have different lengths,
            or ``classifier`` is not recognised.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have the same length, "
            f"got {len(X_train)} and {len(y_train)}."
        )

    if classifier == "cnn":
        if output_path.suffix != ".keras":
            raise ValueError(
                "CNN artifacts must use the '.keras' extension. "
                f"Got: {output_path}"
            )
        labels_path = output_path.with_suffix(".labels.json")
        print("Classifier: KerasCNN")
        train_and_save_cnn(
            X_train=X_train,
            y_train=y_train,
            model_path=output_path,
            labels_path=labels_path,
        )
        print(f"Model saved to {output_path}")
        print(f"Labels saved to {labels_path}")
        return

    clf = build_classifier(classifier)
    print(f"Classifier: {clf.__class__.__name__}")
    clf.fit(X_train, y_train)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"Model saved to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sign-language classifier.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/raw/sign_mnist_train.csv"),
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="Destination path for the saved model.",
    )
    parser.add_argument(
        "--classifier",
        choices=list(CLASSIFIERS),
        default="svc",
        help="Classifier to train (default: svc).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from src.data.extract import build_feature_matrix
    from src.data.ingest import CsvDataset
    from src.data.split import stratified_split

    args = _parse_args()
    print(f"Loading dataset: {args.dataset}")
    X, y = build_feature_matrix(CsvDataset(args.dataset))
    X_train, _, _, y_train, _, _ = stratified_split(X, y)
    print(f"Training on {len(X_train)} samples …")
    train_and_save(X_train, y_train, args.output, args.classifier)
