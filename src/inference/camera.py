from __future__ import annotations

import urllib.request
from collections import deque
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

# Each gesture sample = 3 temporal frames (beginning, middle, end).
# Per frame: 3-dim hand pointing vector + 25 landmarks × 3 coords = 78 values.
# Full feature vector: 3 × 78 = 234.
_N_FRAMES = 3
_N_LANDMARKS = 25   # dataset uses 25 points; MediaPipe gives 21, rest are zero-padded
_N_COORDS = 3
_FRAME_DIM = _N_COORDS + _N_LANDMARKS * _N_COORDS   # 3 + 75 = 78
FEATURE_DIM = _N_FRAMES * _FRAME_DIM                 # 3 × 78 = 234

# Assumed capture resolution for coordinate scaling.
# Training data was collected with an IR depth camera at approximately this resolution.
_IMAGE_W = 640
_IMAGE_H = 480

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path("artifacts/hand_landmarker.task")

# Middle finger MCP index in MediaPipe's 21-landmark model.
_MIDDLE_MCP_IDX = 9


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


def _frame_to_slice(norm_lms: list, world_lms: list) -> np.ndarray:
    """Convert one frame's landmark lists to a 78-dim feature slice.

    Produces the per-frame block matching the dataset schema:
    ``[vec_x, vec_y, vec_z, point_1 … point_75]``

    The hand pointing vector is the unit vector from the wrist to the
    middle-finger MCP joint (world coordinates).  The 25 hand points
    are derived from normalized image landmarks scaled to approximate
    the pixel-space coordinate system used during dataset collection.

    Args:
        norm_lms: Normalized image landmarks from MediaPipe (21 items), or
            an empty list when no hand is detected.
        world_lms: World-space landmarks from MediaPipe (21 items, in metres
            relative to the wrist), or an empty list.

    Returns:
        float32 array of shape ``(78,)``.
    """
    out = np.zeros(_FRAME_DIM, dtype=np.float32)

    if not norm_lms:
        return out

    # --- hand pointing vector (unit vector, range ≈ [-1, 1]) ---
    if world_lms and len(world_lms) > _MIDDLE_MCP_IDX:
        wrist = world_lms[0]
        middle_mcp = world_lms[_MIDDLE_MCP_IDX]
        v = np.array(
            [middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z],
            dtype=np.float32,
        )
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            v /= norm
        out[:_N_COORDS] = v

    # --- 25 hand points from normalized image landmarks ---
    # x: centred on image midpoint (range ≈ [-_IMAGE_W/2, _IMAGE_W/2])
    # y: from top edge              (range ≈ [0, _IMAGE_H])
    # z: depth estimate scaled like x
    for i, lm in enumerate(norm_lms[:_N_LANDMARKS]):
        offset = _N_COORDS + i * _N_COORDS
        out[offset] = (lm.x - 0.5) * _IMAGE_W
        out[offset + 1] = lm.y * _IMAGE_H
        out[offset + 2] = lm.z * _IMAGE_W

    return out


class CameraSource:
    """Live FeatureSource that reads hand landmarks from a webcam via MediaPipe.

    Uses the MediaPipe Tasks API (HandLandmarker).  The model file is
    downloaded automatically to ``artifacts/hand_landmarker.task`` on first use.

    Implements the ``FeatureSource`` protocol without inheriting from it.

    Each call to ``read_features`` captures one webcam frame, extracts
    landmarks, and builds one 78-dim temporal slice.  The source maintains
    a rolling buffer of the three most recent slices; the returned 234-dim
    vector concatenates them as ``[frame_oldest, frame_middle, frame_latest]``,
    matching the dataset's beginning/middle/end gesture structure.

    A zero slice is used for buffer positions not yet filled.
    A zero vector is returned for frames where no hand is detected.

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
        self._frame_buffer: deque[np.ndarray] = deque(
            [np.zeros(_FRAME_DIM, dtype=np.float32)] * _N_FRAMES,
            maxlen=_N_FRAMES,
        )
        self._last_frame: np.ndarray | None = None
        self._last_norm_lms: list = []
        self._last_world_lms: list = []

    @property
    def last_frame(self) -> np.ndarray | None:
        """The last BGR frame captured by ``read_features``, or ``None`` if not yet called."""
        return self._last_frame

    @property
    def last_norm_lms(self) -> list:
        """Normalized image landmarks from the last detected hand (21 items), or ``[]``."""
        return self._last_norm_lms

    @property
    def last_world_lms(self) -> list:
        """World-space landmarks from the last detected hand (21 items), or ``[]``."""
        return self._last_world_lms

    def read_features(self) -> np.ndarray:
        """Capture one frame, update the temporal buffer, return 234-dim vector.

        Reads one frame from the webcam, runs MediaPipe HandLandmarker, and
        converts the detected hand into a 78-dim slice.  The slice is pushed
        into the rolling 3-frame buffer; the concatenated buffer is returned.

        Returns:
            float32 array of shape ``(234,)`` — three consecutive 78-dim slices.

        Raises:
            RuntimeError: If the webcam frame cannot be read.
        """
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read a frame from the webcam.")

        self._last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        norm_lms = result.hand_landmarks[0] if result.hand_landmarks else []
        world_lms = result.hand_world_landmarks[0] if result.hand_world_landmarks else []
        self._last_norm_lms = norm_lms
        self._last_world_lms = world_lms

        self._frame_buffer.append(_frame_to_slice(norm_lms, world_lms))
        return np.concatenate(list(self._frame_buffer))

    def release(self) -> None:
        """Release the webcam and MediaPipe resources."""
        self._cap.release()
        self._landmarker.close()
