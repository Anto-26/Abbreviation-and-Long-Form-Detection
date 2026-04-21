#!/usr/bin/env python3
"""Detect abbreviations and long-forms in text using the trained RNN model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.predictor import AbbreviationDetector

LABEL_COLORS = {
    "B-AC": "\033[92m",   # green
    "B-LF": "\033[94m",   # blue
    "I-LF": "\033[96m",   # cyan
    "B-O":  "\033[0m",    # reset (no colour)
}
RESET = "\033[0m"


def print_result(result: dict, color: bool = True) -> None:
    print("\nToken Labels:")
    for token, label in result["tokens"]:
        col = LABEL_COLORS.get(label, "") if color else ""
        marker = f" [{label}]" if label != "B-O" else ""
        print(f"  {col}{token}{RESET}{marker}")

    abbrevs = result["abbreviations"]
    lforms = result["long_forms"]
    print(f"\nAbbreviations : {abbrevs if abbrevs else 'none detected'}")
    print(f"Long-forms    : {lforms if lforms else 'none detected'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect abbreviations and long-forms in text")
    parser.add_argument("text", nargs="?", help="Text to analyse (wrap in quotes)")
    parser.add_argument("--model", type=str, default=None, help="Path to a saved model checkpoint")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive mode (enter text line-by-line)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable coloured output")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    print("Initialising detector…")
    detector = AbbreviationDetector(model_path=args.model)

    if args.interactive:
        print("Interactive mode — enter a sentence and press Enter (Ctrl+C to quit).\n")
        while True:
            try:
                text = input(">> ").strip()
                if text:
                    print_result(detector.detect(text), color=use_color)
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break

    elif args.text:
        print_result(detector.detect(args.text), color=use_color)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
