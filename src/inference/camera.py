from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    _DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DEPS_AVAILABLE = False

# Target image size matching the Sign Language MNIST training images.
_IMG_SIZE = 28
FEATURE_DIM = _IMG_SIZE * _IMG_SIZE   # 784

# Fractional padding added around the detected hand bounding box.
_BBOX_PAD = 0.1

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path("artifacts/hand_landmarker.task")


def _check_deps() -> None:
    if not _DEPS_AVAILABLE:
        raise RuntimeError(
            "opencv-python and mediapipe are required for CameraSource. "
            "Install them with: uv add opencv-python mediapipe"
        )


def _ensure_model(path: Path = _MODEL_PATH) -> Path:
    """Download the HandLandmarker model file if not already present.

    Args:
        path: Destination path for the ``.task`` model file.

    Returns:
        Path to the downloaded model file.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading HandLandmarker model → {path} …")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("Download complete.")
    return path


def _landmarks_to_bbox(
    landmarks: list,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int] | None:
    """Compute a padded bounding box from normalized hand landmarks.

    Args:
        landmarks: List of NormalizedLandmark from MediaPipe (21 items).
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` pixel coordinates clamped to the frame, or
        ``None`` when the landmark list is empty.
    """
    if not landmarks:
        return None

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    # Compute hand centre and span in pixel space so the square is truly
    # square in pixels regardless of frame aspect ratio.
    cx_px = int((min(xs) + max(xs)) / 2 * frame_w)
    cy_px = int((min(ys) + max(ys)) / 2 * frame_h)
    span_px = max(
        int((max(xs) - min(xs)) * frame_w),
        int((max(ys) - min(ys)) * frame_h),
    )
    half = int(span_px / 10 * (6 + _BBOX_PAD))

    x1 = max(0, cx_px - half)
    y1 = max(0, cy_px - half)
    x2 = min(frame_w, cx_px + half)
    y2 = min(frame_h, cy_px + half)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _crop_to_feature(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop and preprocess a hand region into a 784-dim feature vector.

    Crops the frame to the bounding box, converts to grayscale, resizes to
    28×28, and normalises pixel values to ``[0, 1]``.

    Args:
        frame: Full BGR camera frame.
        bbox: ``(x1, y1, x2, y2)`` bounding box in pixel coordinates.

    Returns:
        A tuple of:
            - float32 array of shape ``(784,)`` with values in ``[0, 1]``
            - uint8 array of shape ``(28, 28)`` — the raw grayscale crop
    """
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # If the bbox was clamped at a frame edge the crop may be non-square.
    # Pad the shorter side with zeros to restore the square before resizing.
    ch, cw = gray.shape
    if ch != cw:
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.uint8)
        square[:ch, :cw] = gray
        gray = square

    resized = cv2.resize(gray, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (resized.flatten().astype(np.float32)) / 255.0, resized


class CameraSource:
    """Live FeatureSource that reads hand images from a webcam via MediaPipe.

    Uses the MediaPipe Tasks API (HandLandmarker) to detect the hand bounding
    box, then crops and resizes the region to a 28×28 grayscale image matching
    the Sign Language MNIST training format.

    Implements the ``FeatureSource`` protocol without inheriting from it.
    Returns a zero vector when no hand is detected in the current frame.

    Args:
        camera_index: OpenCV camera device index (0 = default webcam).
        model_path: Path to the ``.task`` model file.  Downloaded automatically
            if it does not exist.

    Raises:
        RuntimeError: If ``opencv-python`` or ``mediapipe`` are not installed,
            or if the webcam cannot be opened.
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_path: Path = _MODEL_PATH,
    ) -> None:
        _check_deps()

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that the webcam is connected and not in use."
            )

        resolved = _ensure_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=str(resolved))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_frame: np.ndarray | None = None
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_crop: np.ndarray | None = None

    @property
    def last_frame(self) -> np.ndarray | None:
        """The last BGR frame captured by ``read_features``, or ``None``."""
        return self._last_frame

    @property
    def last_bbox(self) -> tuple[int, int, int, int] | None:
        """Bounding box ``(x1, y1, x2, y2)`` of the last detected hand, or ``None``."""
        return self._last_bbox

    @property
    def last_crop(self) -> np.ndarray | None:
        """28×28 grayscale array fed to the classifier, or ``None`` when no hand detected."""
        return self._last_crop

    def read_features(self) -> np.ndarray:
        """Capture one frame, detect the hand, return a 784-dim feature vector.

        Reads one frame from the webcam, runs MediaPipe HandLandmarker to find
        the hand bounding box, crops and resizes the region to 28×28 grayscale,
        and returns the flattened normalised pixel array.

        Returns:
            float32 array of shape ``(784,)`` — zero vector when no hand found.

        Raises:
            RuntimeError: If the webcam frame cannot be read.
        """
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read a frame from the webcam.")

        self._last_frame = frame
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            self._last_bbox = None
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        bbox = _landmarks_to_bbox(result.hand_landmarks[0], w, h)
        self._last_bbox = bbox

        if bbox is None:
            self._last_crop = None
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        feature_vec, raw_crop = _crop_to_feature(frame, bbox)
        self._last_crop = raw_crop
        return feature_vec

    def release(self) -> None:
        """Release the webcam and MediaPipe resources."""
        self._cap.release()
        self._landmarker.close()
