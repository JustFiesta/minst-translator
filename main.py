from __future__ import annotations

import argparse
from pathlib import Path

DATA_PATH = Path("data/raw/sign_mnist_train.csv")
MODEL_PATH = Path("artifacts/svc_model.pkl")


def _train(dataset: Path, output: Path, classifier: str) -> None:
    from src.data.extract import build_feature_matrix
    from src.data.ingest import CsvDataset
    from src.data.split import stratified_split
    from src.model.train import train_and_save

    print(f"Loading dataset: {dataset}")
    X, y = build_feature_matrix(CsvDataset(dataset))
    X_train, _, _, y_train, _, _ = stratified_split(X, y)
    print(f"Training on {len(X_train)} samples…")
    train_and_save(X_train, y_train, output, classifier)


def _eval(dataset: Path, model: Path) -> None:
    from src.data.extract import build_feature_matrix
    from src.data.ingest import CsvDataset
    from src.data.split import stratified_split
    from src.inference.predict import load_model
    from src.model.evaluate import evaluate

    X, y = build_feature_matrix(CsvDataset(dataset))
    _, _, X_test, _, _, y_test = stratified_split(X, y)
    evaluate(load_model(model), X_test, y_test)


def _infer_csv(dataset: Path, model: Path, row: int) -> None:
    from src.inference.csv_source import CsvRowSource
    from src.inference.predict import load_model, predict_sign

    source = CsvRowSource(dataset, row_index=row)
    clf = load_model(model)
    label = predict_sign(clf, source)
    source.release()
    print(f"Row {row} — true: {source.label!r}  predicted: {label!r}")


def _draw_hand_overlay(
    frame: "cv2.Mat",
    bbox: "tuple[int, int, int, int] | None",
) -> None:
    """Draw the hand bounding box and crop indicator on *frame* in-place.

    Args:
        frame: BGR image array from OpenCV.
        bbox: ``(x1, y1, x2, y2)`` pixel bounding box of the detected hand,
            or ``None`` when no hand was detected.
    """
    import cv2

    if bbox is None:
        return

    x1, y1, x2, y2 = bbox
    # Outer bounding box — green rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    # Corner markers for clarity
    corner = 14
    for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        dx = corner if cx == x1 else -corner
        dy = corner if cy == y1 else -corner
        cv2.line(frame, (cx, cy), (cx + dx, cy), (0, 255, 120), 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy), (0, 255, 120), 3)
    # Small "28×28" label above the box
    cv2.putText(
        frame, "28x28", (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1,
    )


def _draw_crop_preview(
    frame: "cv2.Mat",
    crop: "np.ndarray | None",
    label: str,
) -> None:
    """Embed the 28×28 model-input crop as a magnified thumbnail in the frame.

    Resizes the raw grayscale crop to 140×140 using nearest-neighbour
    interpolation (so individual pixels stay visible) and overlays it in the
    bottom-right corner with a labelled border.

    Args:
        frame: BGR image array from OpenCV — modified in-place.
        crop: Grayscale uint8 array of shape ``(28, 28)``, or ``None``.
        label: Currently predicted letter shown below the thumbnail.
    """
    import cv2
    import numpy as np

    THUMB = 140
    MARGIN = 10

    h, w = frame.shape[:2]
    bx = w - THUMB - MARGIN
    by = h - THUMB - MARGIN - 20   # 20 px for label below thumb

    if crop is None:
        # Grey placeholder when no hand is detected
        placeholder = np.full((THUMB, THUMB, 3), 50, dtype=np.uint8)
        frame[by:by + THUMB, bx:bx + THUMB] = placeholder
        cv2.putText(frame, "no hand", (bx + 4, by + THUMB + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    else:
        thumb = cv2.resize(crop, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        frame[by:by + THUMB, bx:bx + THUMB] = thumb_bgr
        cv2.putText(frame, f"-> {label}", (bx + 4, by + THUMB + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

    cv2.rectangle(frame, (bx - 1, by - 1), (bx + THUMB, by + THUMB), (180, 180, 180), 1)
    cv2.putText(frame, "model input", (bx, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


def _infer_camera(model: Path, camera_index: int) -> None:
    import cv2

    from src.inference.camera import CameraSource
    from src.inference.predict import load_model, predict_sign

    clf = load_model(model)
    source = CameraSource(camera_index=camera_index)
    print("Live camera inference — press Q to quit.")

    try:
        while True:
            label = predict_sign(clf, source)
            frame = source.last_frame
            if frame is None:
                break

            _draw_hand_overlay(frame, source.last_bbox)
            _draw_crop_preview(frame, source.last_crop, label)

            cv2.putText(
                frame, label, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3,
            )
            cv2.imshow("PJM Translator — Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


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
        "--source",
        choices=["csv", "camera"],
        default="csv",
        help="Inference source: 'csv' for a single row, 'camera' for live webcam (default: csv).",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index for 'infer --source csv' (default: 0).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for 'infer --source camera' (default: 0).",
    )
    parser.add_argument(
        "--classifier",
        choices=["svc", "rf", "cnn"],
        default="svc",
        help=(
            "Classifier to train: 'svc' (default), 'rf' (RandomForest), "
            "or 'cnn' (requires a .keras --model path)."
        ),
    )
    args = parser.parse_args()

    if args.mode == "train":
        _train(args.dataset, args.model, args.classifier)
    elif args.mode == "eval":
        _eval(args.dataset, args.model)
    elif args.mode == "infer":
        if args.source == "camera":
            _infer_camera(args.model, args.camera)
        else:
            _infer_csv(args.dataset, args.model, args.row)


if __name__ == "__main__":
    main()
