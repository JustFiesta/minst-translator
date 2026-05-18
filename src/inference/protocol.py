from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class FeatureSource(Protocol):
    """Source of a single 234-dimensional feature vector ready for classification.

    Any object that implements ``read_features`` and ``release`` satisfies this
    protocol — no inheritance required.  Current implementations:
    ``CsvRowSource`` (offline, from CSV) and ``CameraSource`` (live webcam).
    Future implementations: video files, network streams, etc.
    """

    def read_features(self) -> np.ndarray:
        """Return one feature vector of shape (234,) and dtype float32.

        Returns:
            1-D float32 numpy array with exactly 234 elements.
        """
        ...

    def release(self) -> None:
        """Release any underlying resources (camera handle, file descriptor, etc.).

        Must be safe to call multiple times.
        """
        ...
