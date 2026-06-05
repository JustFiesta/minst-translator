# pjm-translator

Real-time hand sign classifier based on the Sign Language MNIST dataset.

The app can:
- train a classifier (`train`)
- evaluate a trained model (`eval`)
- run inference from a CSV row or live webcam (`infer`)

**Dataset:** [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data) —
27 455 training samples, 7 172 test samples, 24 ASL letter classes (A–Y, no J or Z),
each sample is a flattened 28×28 grayscale image (`pixel1`…`pixel784`, values 0–255).

---

## 1. Project presentation assumptions

These are the conditions used during live-inference testing in the lab.
Deviating from them may reduce prediction accuracy.

| Condition | Recommended value |
|---|---|
| Lighting | Even, stable — no hard shadows on the hand |
| Hand | Left hand, facing the camera front-on |
| Camera | DroidCam (or any webcam) as OpenCV source |
| Mirror | **Off** — disable mirroring in DroidCam / camera app |
| Background | Plain, uniform colour to reduce false detections |

---

## 2. How it works

The classifier does not recognise gestures semantically — it compares raw pixel
patterns against patterns seen during training.

**Camera inference pipeline:**

1. A webcam frame is captured.
2. **MediaPipe Gesture Recognizer** detects the hand and returns landmarks.
3. A bounding box is computed from the landmarks and the hand region is cropped.
4. The crop is resized to **28×28**, converted to **grayscale**, and flattened to a
   784-element float32 vector normalised to `[0, 1]`.
5. The classifier (`svc`, `rf`, or `cnn`) predicts the letter label.

Because the model operates on raw pixels after normalisation, frame quality
(lighting, hand position, contrast) directly affects prediction reliability.

---

## 3. Model benchmarks

All three classifiers were trained on the same 80 % split of
`data/raw/sign_mnist_train.csv` and evaluated on the same held-out test split
(`data/raw/sign_mnist_test.csv`):

| Model | Artifact | Test accuracy |
|---|---|:---:|
| SVC (RBF kernel) | `artifacts/svc_model.pkl` | **0.82** |
| Random Forest | `artifacts/rf_model.pkl` | **0.82** |
| CNN (Keras / TF) | `artifacts/cnn_model.keras` | **0.92** |

**Algorithm overview:**

- **SVC (RBF)** — support-vector classifier; finds a non-linear decision boundary
  in the 784-dimensional feature space using a radial-basis-function kernel.
- **Random Forest** — ensemble of 200 decision trees; classifies by majority vote.
- **CNN** — two convolutional blocks (Conv2D → MaxPool) followed by a dense head;
  learns spatial filters directly from the 28×28 pixel grid.

The architecture diagram is in [`diagrams/architecture.png`](diagrams/architecture.png).

---

## 4. Requirements

- Python `>=3.13`
- `uv` package manager
- DroidCam/webcam (only for `infer --source camera`)

Main dependencies (from `pyproject.toml`):

- `numpy`, `pandas`, `scikit-learn`, `joblib`
- `opencv-python`, `mediapipe`
- `tensorflow>=2.21.0`

---

## 5. Setup and installation

### Windows (PowerShell)

```powershell
uv python install 3.13
uv venv --python 3.13
.venv\Scripts\Activate.ps1
uv sync
```

### Linux / macOS

```bash
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv sync
```

---

## 6. Dataset files

Expected CSV files (downloaded manually from Kaggle):

- `data/raw/sign_mnist_train.csv`
- `data/raw/sign_mnist_test.csv`

```powershell
# Windows
New-Item -ItemType Directory -Path data/raw -Force
```

```bash
# Linux / macOS
mkdir -p data/raw
```

Download: [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data)

If the dataset path is wrong the app raises `FileNotFoundError`.

---

## 7. CLI usage

```bash
uv run python main.py <mode> [options]
```

`<mode>` is required:

| Mode | Description |
|---|---|
| `train` | Fit a classifier and save the artifact |
| `eval` | Evaluate a saved model on the test split |
| `infer` | Run inference from CSV row or live webcam |

### Arguments

| Argument | Choices / type | Default | Modes | Description |
|---|---|---|---|---|
| `--dataset` | path | `data/raw/sign_mnist_train.csv` | train, eval, infer(csv) | CSV dataset path |
| `--model` | path | `artifacts/svc_model.pkl` | train, eval, infer | Model artifact (use `.keras` for CNN) |
| `--classifier` | `svc` `rf` `cnn` | `svc` | train | Classifier to train |
| `--source` | `csv` `camera` | `csv` | infer | Inference input source |
| `--row` | int | `0` | infer csv | Row index in dataset CSV |
| `--camera` | int | `0` | infer camera | OpenCV camera device index |

Notes:
- `--classifier cnn` requires a `--model` path ending in `.keras`.
- `--source`, `--row`, `--camera` are ignored in `train` mode.
- `--source`, `--row`, `--camera`, `--classifier` are ignored in `eval` mode.
- `--dataset` is ignored in `infer --source camera`.

---

## 8. Command examples

### Train

```bash
# SVC (default)
uv run python main.py train

# Random Forest
uv run python main.py train --classifier rf --model artifacts/rf_model.pkl

# CNN
uv run python main.py train --classifier cnn --model artifacts/cnn_model.keras \
    --dataset data/raw/sign_mnist_train.csv
```

### Evaluate

```bash
uv run python main.py eval --model artifacts/svc_model.pkl \
    --dataset data/raw/sign_mnist_test.csv
```

Output: overall accuracy + per-class `classification_report`.

### Inference — CSV row

```bash
uv run python main.py infer --source csv --row 42 \
    --dataset data/raw/sign_mnist_test.csv --model artifacts/svc_model.pkl
```

### Inference — live webcam

```bash
uv run python main.py infer --source camera --camera 0 --model artifacts/svc_model.pkl
```

- Opens a webcam window with the hand bounding box overlay.
- Shows the 28×28 model-input thumbnail in the corner.
- Press `q` to quit.

---

## 9. Alternative module CLI

`src/model/train.py` exposes its own `__main__` entry:

```bash
uv run python -m src.model.train \
    --dataset data/raw/sign_mnist_train.csv \
    --output artifacts/svc_model.pkl \
    --classifier svc
```

---

## 10. Development commands

| Task | Command |
|---|---|
| Run all tests | `uv run pytest` |
| Run with coverage | `uv run pytest --cov=src --cov-report=term-missing` |
| Lint | `uv run ruff check src/ tests/` |
| Format | `uv run ruff format src/ tests/` |

---

## 11. Troubleshooting

**`Model artifact not found`** — run `main.py train` first, or point `--model`
to an existing file.

**`Cannot open camera at index X`** — check that the webcam is connected and not
used by another app; try `--camera 1` or `--camera 2`.

**`row_index … out of range`** — `--row` exceeds the dataset length; choose a
valid index for the selected `--dataset`.

**MediaPipe model download** — on the first camera run the app auto-downloads
`artifacts/gesture_recognizer.task`; ensure internet access is available.

---

## 12. Project structure

```text
main.py                        CLI entrypoint
src/
  data/                        ingest · feature extraction · split
  model/
    classifiers/               svc.py · random_forest.py · cnn.py
    train.py                   orchestrates training + artifact saving
    evaluate.py                accuracy + classification report
  inference/
    camera.py                  webcam capture + MediaPipe + 28×28 crop
    csv_source.py              offline feature source from CSV row
    predict.py                 load_model · predict_sign
tests/                         pytest suite mirroring src/
data/                          Sign Language MNIST CSVs (gitignored)
artifacts/                     trained model files (gitignored)
diagrams/                      architecture diagram
```
