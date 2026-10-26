"""Prédit le sentiment d'une ou plusieurs critiques avec le modèle entraîné.

Usage : python -m sentiment.predict "This movie was great!" "Worst film ever."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .data import LABELS

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "sentiment_model.joblib"


def load_model(path: Path = DEFAULT_MODEL):
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {path}. Lance d'abord : python -m sentiment.train"
        )
    return joblib.load(path)


def predict(texts: list[str], model) -> list[dict]:
    """Renvoie, pour chaque texte, le sentiment prédit et la probabilité d'être positif."""
    positive = model.predict_proba(texts)[:, 1]
    return [
        {"text": text, "sentiment": LABELS[int(p >= 0.5)], "positive_probability": round(float(p), 4)}
        for text, p in zip(texts, positive)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("texts", nargs="+", help="critiques à analyser (en anglais)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    for result in predict(args.texts, load_model(args.model)):
        print(f"{result['sentiment']:<8} ({result['positive_probability']:.1%} positif)  {result['text']}")


if __name__ == "__main__":
    main()
