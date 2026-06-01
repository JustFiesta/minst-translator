# CONTEXT: pjm-translator

IMPORT: .ai/shared.md

PURPOSE:
  Real-time PJM → Polish translation.
  Stage: in development — src/ scaffold created; data pipeline and model modules in progress.
  Dataset: https://www.kaggle.com/datasets/adamlaput/sign-language-pjm

STACK: → see SHARED.STACK
  future-deps (⚠️ GATE before adding any):
    - OpenCV / MediaPipe  → video capture + hand-landmark extraction (Python 3.14 compat TBD)

ARCH: → see SHARED.ARCH
  current status:
    main.py                      → stub (prints hello; CLI wiring pending)
    data/explore.py              → stub (single TODO comment)
    data/raw/                    → 550-points.csv + PJM-points.csv present (gitignored)
    src/                         → package scaffold created
      src/data/protocol.py       → DatasetSource Protocol (done)
      src/data/ingest.py         → CsvDataset (done)
      src/data/extract.py        → build_feature_matrix (done)
      src/data/split.py          → stratified_split (done)
      src/model/train.py         → train_and_save, CLI via __main__ (done)
      src/model/evaluate.py      → evaluate (done)
      src/inference/protocol.py  → FeatureSource Protocol (done)
      src/inference/predict.py   → load_model, predict_sign (done)
      src/inference/csv_source.py → CsvRowSource (done)
      src/inference/camera.py    → pending (optional, MediaPipe compat)
    tests/                       → all test files done; 25/25 passing
    .github/workflows/           → ci.yaml and train.yaml are single-line TODOs

CONVENTIONS:
  naming:      → see SHARED.NAMING
  entry_guard: if __name__ == "__main__": in every executable script
  imports:     stdlib → third-party → local (PEP 8)
  type_hints:  MUST on all public function signatures
  docstrings:  Google-style on all public functions and classes
  line_length: 88 chars (Black-compatible)
  formatter:   Ruff (configured; linting active)

DEPS:
  numpy>=2.4.4       → numerical arrays
  pandas>=3.0.3      → CSV loading and DataFrame operations
  scikit-learn>=1.8.0 → SVC / RandomForest classifier + train_test_split
  joblib>=1.5.3      → model serialisation (.pkl)
  pytest>=9.0.3      → test runner
  pytest-cov>=7.1.0  → coverage reporting
  ruff>=0.15.13      → linter

DEV_SETUP:
  pip install uv
  uv sync
  uv run main.py
  → full instructions: doc/dev-setup.md
