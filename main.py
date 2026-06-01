from __future__ import annotations

import argparse
from pathlib import Path

DATA_PATH = Path("data/raw/550-points.csv")
MODEL_PATH = Path("artifacts/model.pkl")


def _train(dataset: Path, output: Path) -> None:
    from src.data.extract import build_feature_matrix
    from src.data.ingest import CsvDataset
    from src.data.split import stratified_split
    from src.model.train import train_and_save

    print(f"Loading dataset: {dataset}")
    X, y = build_feature_matrix(CsvDataset(dataset))
    X_train, _, _, y_train, _, _ = stratified_split(X, y)
    print(f"Training on {len(X_train)} samples…")
    train_and_save(X_train, y_train, output)


def _eval(dataset: Path, model: Path) -> None:
    from src.data.extract import build_feature_matrix
    from src.data.ingest import CsvDataset
    from src.data.split import stratified_split
    from src.inference.predict import load_model
    from src.model.evaluate import evaluate

    X, y = build_feature_matrix(CsvDataset(dataset))
    _, _, X_test, _, _, y_test = stratified_split(X, y)
    evaluate(load_model(model), X_test, y_test)


def _infer(dataset: Path, model: Path, row: int) -> None:
    from src.inference.csv_source import CsvRowSource
    from src.inference.predict import load_model, predict_sign

    source = CsvRowSource(dataset, row_index=row)
    clf = load_model(model)
    label = predict_sign(clf, source)
    source.release()
    print(f"Row {row} — true: {source.label!r}  predicted: {label!r}")


def main() -> None:
    """CLI entrypoint for pjm-translator."""
    parser = argparse.ArgumentParser(
        description="PJM sign-language classifier — train, evaluate, or infer."
    )
    parser.add_argument(
        "mode",
        choices=["train", "eval", "infer"],
        help="Operation mode.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_PATH,
        help=f"Path to the CSV dataset (default: {DATA_PATH}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the model artifact (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index to classify in 'infer' mode (default: 0).",
    )
    args = parser.parse_args()

    if args.mode == "train":
        _train(args.dataset, args.model)
    elif args.mode == "eval":
        _eval(args.dataset, args.model)
    elif args.mode == "infer":
        _infer(args.dataset, args.model, args.row)


if __name__ == "__main__":
    main()
