from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

LABEL_ENCODING: dict[str, int] = {"B-O": 0, "B-AC": 1, "B-LF": 2, "I-LF": 3}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_ENCODING.items()}
MAX_LEN = 128


def load_plod_dataset():
    """Download and return the PLOD-CW dataset from HuggingFace."""
    return load_dataset("surrey-nlp/PLOD-CW")


def tokens_to_vectors(tokens: list[str], word2vec_model, max_len: int = MAX_LEN) -> np.ndarray:
    """Convert a token list to a zero-padded matrix of Word2Vec vectors."""
    vecs = []
    for token in tokens:
        vecs.append(word2vec_model[token] if token in word2vec_model else np.zeros(word2vec_model.vector_size))

    if len(vecs) > max_len:
        vecs = vecs[:max_len]
    else:
        vecs += [np.zeros(word2vec_model.vector_size)] * (max_len - len(vecs))

    return np.array(vecs, dtype=np.float32)


def prepare_split(split, word2vec_model, max_len: int = MAX_LEN) -> TensorDataset:
    """Convert a HuggingFace dataset split into a PyTorch TensorDataset."""
    all_vecs = []
    all_labels = []

    for sample in split:
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        all_vecs.append(tokens_to_vectors(tokens, word2vec_model, max_len))

        numerical = [LABEL_ENCODING.get(tag, -1) for tag in ner_tags]
        if len(numerical) > max_len:
            numerical = numerical[:max_len]
        else:
            numerical += [-1] * (max_len - len(numerical))

        all_labels.append(torch.tensor(numerical, dtype=torch.long))

    x = torch.tensor(np.array(all_vecs), dtype=torch.float32)
    y = torch.stack(all_labels)
    return TensorDataset(x, y)


def make_loader(dataset: TensorDataset, batch_size: int = 16, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
