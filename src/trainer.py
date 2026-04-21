from __future__ import annotations

from pathlib import Path

import gensim.downloader as api
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .dataset import (
    ID_TO_LABEL,
    LABEL_ENCODING,
    MAX_LEN,
    load_plod_dataset,
    make_loader,
    prepare_split,
)
from .model import RNNModel

MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "rnn_word2vec_rmsprop.pt"


def train(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.001,
    max_len: int = MAX_LEN,
    save_path: Path | None = None,
) -> RNNModel:
    save_path = Path(save_path) if save_path else DEFAULT_MODEL_PATH

    print("Loading Word2Vec (word2vec-google-news-300)… this may take a minute on first run.")
    word2vec = api.load("word2vec-google-news-300")

    print("Loading PLOD-CW dataset from HuggingFace…")
    dataset = load_plod_dataset()

    print("Preparing data splits…")
    train_ds = prepare_split(dataset["train"], word2vec, max_len)
    val_ds = prepare_split(dataset["validation"], word2vec, max_len)
    test_ds = prepare_split(dataset["test"], word2vec, max_len)

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=batch_size)
    test_loader = make_loader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = RNNModel(
        input_size=word2vec.vector_size,
        hidden_dim=128,
        output_dim=len(LABEL_ENCODING),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1, out.shape[-1]), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = _compute_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1:>2}/{epochs}  train_loss={total_loss:.4f}  val_loss={val_loss:.4f}")

    print("\nEvaluating on test set…")
    flat_pred, flat_true = _collect_predictions(model, test_loader, device)
    label_names = [ID_TO_LABEL[i] for i in range(len(LABEL_ENCODING))]
    print(classification_report(flat_true, flat_pred, target_names=label_names, zero_division=0))
    print(f"Accuracy : {accuracy_score(flat_true, flat_pred):.4f}")
    print(f"F1 (wt.) : {f1_score(flat_true, flat_pred, average='weighted'):.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": word2vec.vector_size,
            "hidden_dim": 128,
            "output_dim": len(LABEL_ENCODING),
            "label_encoding": LABEL_ENCODING,
            "max_len": max_len,
        },
        save_path,
    )
    print(f"\nModel saved → {save_path}")
    return model


def _compute_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total += criterion(out.view(-1, out.shape[-1]), y.view(-1)).item()
    model.train()
    return total


def _collect_predictions(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = torch.argmax(out, dim=-1).cpu().numpy()
            preds.append(p.flatten())
            truths.append(y.numpy().flatten())

    flat_pred = np.concatenate(preds)
    flat_true = np.concatenate(truths)
    valid = flat_true != -1
    return flat_pred[valid], flat_true[valid]
