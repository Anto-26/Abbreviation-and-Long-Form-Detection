from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import ID_TO_LABEL, LABEL_ENCODING, MAX_LEN, tokens_to_vectors
from .model import RNNModel

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "rnn_word2vec_rmsprop.pt"

DetectionResult = dict[str, Any]


class AbbreviationDetector:
    """Load a trained RNN model and detect abbreviations / long-forms in text."""

    def __init__(self, model_path: str | Path | None = None, word2vec_model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at '{model_path}'.\n"
                "Run  python train.py  first to train and save the model."
            )

        checkpoint = torch.load(model_path, map_location=self.device)
        self._label_encoding: dict[str, int] = checkpoint["label_encoding"]
        self._id_to_label: dict[int, str] = {v: k for k, v in self._label_encoding.items()}
        self._max_len: int = checkpoint.get("max_len", MAX_LEN)

        self.model = RNNModel(
            input_size=checkpoint["input_size"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        if word2vec_model is None:
            import gensim.downloader as api
            print("Loading Word2Vec model…")
            self._w2v = api.load("word2vec-google-news-300")
        else:
            self._w2v = word2vec_model

    def predict_tokens(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Return (token, label) pairs for every token in the list."""
        original_len = min(len(tokens), self._max_len)
        vec = tokens_to_vectors(tokens, self._w2v, self._max_len)
        x = torch.tensor([vec], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self.model(x)
            preds = torch.argmax(out, dim=-1)[0].cpu().numpy()

        labels = [self._id_to_label[int(p)] for p in preds[:original_len]]
        return list(zip(tokens[:original_len], labels))

    def detect(self, text: str) -> DetectionResult:
        """
        Detect abbreviations and long-forms in free text.

        Returns a dict with:
          - tokens:        list of (token, label) pairs
          - abbreviations: list of detected abbreviation strings
          - long_forms:    list of detected long-form strings (multi-token spans joined)
        """
        tokens = text.split()
        predictions = self.predict_tokens(tokens)

        abbreviations: list[str] = []
        long_forms: list[str] = []
        current_lf: list[str] = []

        for token, label in predictions:
            if label == "B-AC":
                abbreviations.append(token)
                if current_lf:
                    long_forms.append(" ".join(current_lf))
                    current_lf = []
            elif label == "B-LF":
                if current_lf:
                    long_forms.append(" ".join(current_lf))
                current_lf = [token]
            elif label == "I-LF":
                current_lf.append(token)
            else:
                if current_lf:
                    long_forms.append(" ".join(current_lf))
                    current_lf = []

        if current_lf:
            long_forms.append(" ".join(current_lf))

        return {
            "tokens": predictions,
            "abbreviations": abbreviations,
            "long_forms": long_forms,
        }
