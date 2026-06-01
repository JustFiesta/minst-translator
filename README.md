# minst-translator

Real-time hand sign classifier based on the Sign Language MNIST dataset.
The app can:
- train a model (`train`)
- evaluate a trained model (`eval`)
- run inference from CSV row or live webcam (`infer`)

Dataset: [Sign Language MNIST (Kaggle)](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data)

## 1. What this project does

Pipeline overview:
1. Load Sign Language MNIST CSV (`label` + `pixel1..pixel784`).
2. Normalize pixels to `[0, 1]`.
3. Train classifier (`svc` or `rf`) and save to `.pkl`.
4. In inference:
   - `csv` source: classify one dataset row
   - `camera` source: detect hand with MediaPipe, crop to `28x28`, predict letter

Default paths:
- dataset: `data/raw/sign_mnist_train.csv`
- model: `artifacts/model.pkl`

## 2. Requirements

- Python `>=3.14`
- `uv` package manager
- Webcam (only for `infer --source camera`)

Main dependencies (from `pyproject.toml`):
- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `opencv-python`
- `mediapipe`

## 3. Setup and installation

### Windows (PowerShell)

```powershell
uv python install 3.14
uv venv --python 3.14
.venv\Scripts\Activate.ps1
uv sync
```

### Linux/macOS

```bash
uv python install 3.14
uv venv --python 3.14
source .venv/bin/activate
uv sync
```

## 4. Dataset files

Expected CSV files:
- `data/raw/sign_mnist_train.csv`
- `data/raw/sign_mnist_test.csv`

CSV schema:
- `label` (int class id)
- `pixel1` ... `pixel784` (grayscale pixel values 0..255)

If dataset path is wrong, project raises `FileNotFoundError`.

## 5. CLI usage (all arguments and flags)

Main entrypoint:

```bash
uv run python main.py <mode> [options]
```

Where `<mode>` is required and must be one of:
- `train`
- `eval`
- `infer`

### Global arguments in `main.py`

| Argument | Type / Choices | Default | Used in mode | Description |
|---|---|---|---|---|
| `mode` | `train`, `eval`, `infer` | - | all | Operation mode |
| `--dataset` | path | `data/raw/sign_mnist_train.csv` | train, eval, infer(csv) | CSV dataset path |
| `--model` | path | `artifacts/model.pkl` | train, eval, infer | Model artifact path |
| `--classifier` | `svc`, `rf` | `svc` | train | Classifier to train |
| `--source` | `csv`, `camera` | `csv` | infer | Inference source |
| `--row` | int | `0` | infer(csv) | CSV row index |
| `--camera` | int | `0` | infer(camera) | OpenCV camera device index |

Notes:
- In `train` mode, `--source`, `--row`, `--camera` are ignored.
- In `eval` mode, `--source`, `--row`, `--camera`, `--classifier` are ignored.
- In `infer --source camera`, `--dataset` is ignored.

## 6. Command examples

### Train model

Use default dataset/model with SVC:

```bash
uv run python main.py train
```

Train RandomForest and save to custom file:

```bash
uv run python main.py train --classifier rf --model artifacts/rf_model.pkl
```

### Evaluate model

```bash
uv run python main.py eval --model artifacts/model.pkl --dataset data/raw/sign_mnist_test.csv
```

Output includes:
- accuracy
- full `classification_report`

### Inference from CSV row

```bash
uv run python main.py infer --source csv --row 42 --dataset data/raw/sign_mnist_test.csv --model artifacts/model.pkl
```

Output example:
- true label for row
- predicted label

### Live webcam inference

```bash
uv run python main.py infer --source camera --camera 0 --model artifacts/model.pkl
```

Runtime behavior:
- opens webcam window
- overlays detected hand bounding box
- shows `28x28` model input preview
- press `q` to quit

## 7. Alternative module CLI

`src/model/train.py` also has standalone CLI:

```bash
uv run python -m src.model.train --dataset data/raw/sign_mnist_train.csv --output artifacts/model.pkl --classifier svc
```

Arguments there:
- `--dataset`
- `--output`
- `--classifier` (`svc`/`rf`)

Help for each CLI:

```bash
uv run python main.py --help
uv run python -m src.model.train --help
```

## 8. Development commands

Run tests:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

Lint:

```bash
uv run ruff check src/ tests/
```

Format:

```bash
uv run ruff format src/ tests/
```

## 9. Troubleshooting

### `Model artifact not found`

Error from `load_model` means `--model` file does not exist.
Fix:
1. Train first (`main.py train`), or
2. Point `--model` to existing `.pkl`.

### `Cannot open camera at index X`

Fix:
- verify webcam is connected
- close other apps using webcam
- try another index (`--camera 1`, `--camera 2`)

### `row_index ... out of range`

`--row` is outside dataset length.
Use valid row index based on selected `--dataset`.

### MediaPipe model download

On first camera run, project auto-downloads:
- `artifacts/hand_landmarker.task`

Make sure internet is available for first download.

## 10. Project structure (high-level)

```text
main.py                  # CLI entrypoint
src/data/                # CSV ingest, feature matrix, split
src/model/               # training and evaluation
src/inference/           # model loading, CSV source, camera source
tests/                   # pytest tests
data/raw/                # input dataset files
artifacts/               # trained model files (.pkl, .task)
```
