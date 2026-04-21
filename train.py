#!/usr/bin/env python3
"""Train the RNN + Word2Vec model for abbreviation / long-form detection."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the abbreviation/LF detector")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for RMSprop (default: 0.001)")
    parser.add_argument("--max-len", type=int, default=128, help="Max token sequence length (default: 128)")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save trained model (optional)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
