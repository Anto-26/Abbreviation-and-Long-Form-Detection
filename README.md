# Abbreviation and Long-Form Detector

A token-level NLP model that identifies abbreviations (AC) and their long-forms (LF) in text using an RNN with pre-trained Word2Vec embeddings.

## Overview

Given a sentence like `"EPI = Echo planar imaging ."`, the model labels each token:

| Token | Label |
|-------|-------|
| EPI | B-AC (abbreviation) |
| = | B-O (other) |
| Echo | B-LF (long-form start) |
| planar | I-LF (long-form continuation) |
| imaging | I-LF |
| . | B-O |

Output: `abbreviations: ["EPI"]`, `long_forms: ["Echo planar imaging"]`

## Model

The best-performing model uses an RNN with 300-dimensional Word2Vec embeddings trained with RMSprop.

| Metric | Value |
|--------|-------|
| Test Accuracy | 87.38% |
| Weighted F1 | 0.8569 |
| Architecture | Single-layer RNN, hidden dim 128 |
| Optimizer | RMSprop, lr=0.001 |

Pre-trained weights are saved at `models/rnn_word2vec_rmsprop.pt`.

## Setup

```bash
pip install -r requirements.txt
```

The first run automatically downloads the Google News Word2Vec model (~1.5 GB) and caches it locally.

## Usage

### Web Interface

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Includes example sentences, color-coded token visualization, and extracted abbreviation/long-form lists.

### CLI

```bash
# Analyze a single sentence
python detect.py "EPI = Echo planar imaging ."

# Interactive mode (enter sentences one at a time)
python detect.py -i

# Use a custom model checkpoint
python detect.py "text here" --model path/to/model.pt
```

### Python API

```python
from src.predictor import AbbreviationDetector

detector = AbbreviationDetector("models/rnn_word2vec_rmsprop.pt")
result = detector.detect("EPI = Echo planar imaging .")

print(result["abbreviations"])  # ["EPI"]
print(result["long_forms"])     # ["Echo planar imaging"]
print(result["tokens"])         # [("EPI", "B-AC"), ("=", "B-O"), ...]
```

### Training

```bash
python train.py                            # Default: 10 epochs, batch size 16, lr=0.001
python train.py --epochs 20 --lr 0.0005   # Custom hyperparameters
python train.py --save-path my_model.pt   # Custom output path
```

Training uses the [PLOD-CW dataset](https://huggingface.co/datasets/surrey-nlp/PLOD-CW) from HuggingFace, downloaded automatically on first run.

## Project Structure

```
abbreviation_lf_detector/
├── app.py              # Streamlit web interface
├── detect.py           # CLI tool
├── train.py            # Training entry point
├── requirements.txt
├── models/
│   └── rnn_word2vec_rmsprop.pt   # Best trained model
├── notebooks/
│   ├── Data Analysis.ipynb
│   ├── Experiment-1.ipynb        # Adam optimizer
│   ├── Experiment-2.ipynb        # Adadelta optimizer
│   ├── Experiment-3.ipynb        # Adagrad optimizer
│   └── Experiment-4.ipynb        # RMSprop optimizer (best)
└── src/
    ├── model.py        # RNN architecture
    ├── dataset.py      # Data loading and preprocessing
    ├── trainer.py      # Training loop
    └── predictor.py    # Inference
```

## Dependencies

- `torch` — neural network framework
- `gensim` — Word2Vec embeddings
- `datasets` — HuggingFace PLOD-CW dataset
- `scikit-learn` — evaluation metrics
- `streamlit` — web interface
