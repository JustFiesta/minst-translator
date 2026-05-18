# STANDARDS: Coding

IMPORT: .ai/shared.md

NAMING: → see SHARED.NAMING

RUNTIME:
  Python 3.14 — use modern language features where they improve clarity.
  MUST NOT add / remove / upgrade deps without GATE approval.

---

[IMPORTS]
Order: stdlib → third-party → local (PEP 8). One blank line between groups.

```python
import os
from pathlib import Path

import numpy as np

from src.pipeline import preprocess
```

- MUST use explicit imports; MUST NOT use wildcard (`from module import *`)
- MUST alias third-party consistently: `import numpy as np`

---

[TYPE HINTS]
- MUST annotate all public function signatures (params + return)
- Use `from __future__ import annotations` for forward references
- Use `TypeAlias` / `type` statements for complex types (Python 3.12+)

```python
def load_frames(video_path: Path) -> np.ndarray: ...
```

---

[DOCSTRINGS]
Google-style on all public functions, classes, and modules.

```python
def classify_sign(frames: np.ndarray, threshold: float = 0.5) -> str:
    """Classifies a frame sequence as a PJM sign label.

    Args:
        frames: Array of shape (T, H, W, C) containing video frames.
        threshold: Minimum confidence score to accept a prediction.

    Returns:
        Predicted sign label as a Polish string, or "" if below threshold.

    Raises:
        ValueError: If `frames` is empty or has an unexpected shape.
    """
```

---

[CODE QUALITY]
- MUST NOT duplicate logic — extract to a shared helper
- MUST keep functions single-responsibility and short
- MUST NOT use magic numbers — name constants at module level
- MUST validate inputs at the top of a function (fail fast)
- MUST NOT execute I/O or heavy computation at module level (no import-time side effects)

---

[DEPENDENCY INVERSION]
RULE: High-level modules MUST NOT depend on low-level implementation details.
      Both MUST depend on abstractions (Protocol or ABC).
      Apply to every interchangeable component: input source, model backend, data loader, output sink.

Define Protocol in `<subpackage>/protocol.py`:

```python
# src/inference/protocol.py
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

class FeatureSource(Protocol):
    def read_features(self) -> np.ndarray: ...
    def release(self) -> None: ...
```

Use `TYPE_CHECKING` guard to avoid importing heavy packages (numpy, pandas) at module
load time — keeps import cost zero for code that only type-checks against the Protocol.

Concrete implementations satisfy the protocol without inheriting from it:

```python
# src/inference/csv_source.py
class CsvRowSource:
    def read_features(self) -> np.ndarray: ...
    def release(self) -> None: ...
```

High-level code depends on the abstraction only:

```python
# src/inference/predict.py
from src.inference.protocol import FeatureSource

def predict_sign(clf, source: FeatureSource) -> str: ...
```

Active Protocol files in this project:
  src/data/protocol.py      → DatasetSource (training data source)
  src/inference/protocol.py → FeatureSource (inference input source)

- Use `typing.Protocol` for structural (duck-typed) interfaces
- Use `abc.ABC` + `@abstractmethod` when enforced inheritance is required
- Wire concrete implementations at the application boundary (`main.py` or factory), never inside business logic

---

[FORMATTING]
- Line length: 88 chars (Black-compatible)
- No trailing whitespace; single blank line at end of file
- Formatter: Black | Linter: Ruff (planned; write code that would pass both)

---

[ERROR HANDLING]
- MUST use specific exception types; MUST NOT `except Exception` without re-raising or logging
- MUST include the bad value + expected format in error messages
- `ValueError` → programming errors | `RuntimeError` → unexpected system/IO failures

---

[ARCHITECTURE]
- `src/`    → core importable package; MUST NOT contain standalone scripts
- `data/`   → exploration scripts; MUST NOT be imported by `src/`
- `main.py` → thin entrypoint; MUST delegate logic to `src/`
- `tests/`  → mirrors `src/` layout; one test file per source module
- MUST NOT create new top-level directories without an architecture discussion
