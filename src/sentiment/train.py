"""Entraîne les modèles candidats, choisit le meilleur et l'évalue sur le jeu de test.

Le choix du modèle se fait sur un jeu de validation tiré du train : le jeu de test
n'est utilisé qu'une fois, pour mesurer le modèle retenu.

Usage : python -m sentiment.train
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from .data import LABELS, load_split  # noqa: E402
from .models import build_models  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _scores(model, texts, labels) -> dict[str, float]:
    predicted = model.predict(texts)
    proba = model.predict_proba(texts)[:, 1]
    return {
        "accuracy": round(accuracy_score(labels, predicted), 4),
        "f1": round(f1_score(labels, predicted), 4),
        "roc_auc": round(roc_auc_score(labels, proba), 4),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    train_df, test_df = load_split("train"), load_split("test")
    print(f"Train : {len(train_df)} critiques · Test : {len(test_df)} critiques")

    x_train, x_val, y_train, y_val = train_test_split(
        train_df["text"], train_df["label"],
        test_size=args.val_size, stratify=train_df["label"], random_state=args.seed,
    )

    validation = {}
    for name, model in build_models(args.seed).items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        validation[name] = _scores(model, x_val, y_val)
        validation[name]["train_seconds"] = round(time.perf_counter() - start, 1)
        print(f"  {name:<20} validation {validation[name]}")

    best_name = max(validation, key=lambda n: validation[n]["accuracy"])
    print(f"Modèle retenu : {best_name} — réentraînement sur tout le train")
    best = build_models(args.seed)[best_name]
    best.fit(train_df["text"], train_df["label"])

    test_scores = _scores(best, test_df["text"], test_df["label"])
    predicted = best.predict(test_df["text"])
    print(f"Test : {test_scores}")
    print(classification_report(test_df["label"], predicted, target_names=list(LABELS.values()), digits=4))

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, args.models_dir / "sentiment_model.joblib", compress=3)

    metrics = {"best_model": best_name, "validation": validation, "test": test_scores}
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        test_df["label"], predicted, display_labels=list(LABELS.values()),
        cmap="Greens", values_format="d", ax=ax,
    )
    ax.set_title(f"Matrice de confusion — {best_name} (test)")
    fig.tight_layout()
    fig.savefig(args.results_dir / "confusion_matrix.png", dpi=120)
    print(f"Modèle sauvegardé dans {args.models_dir}, résultats dans {args.results_dir}")


if __name__ == "__main__":
    main()
