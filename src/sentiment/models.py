"""Pipelines TF-IDF + classifieur comparés pendant l'entraînement."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import clean_text


def _vectorizer() -> TfidfVectorizer:
    # Unigrammes + bigrammes pour capter les négations ("not good") ;
    # sublinear_tf atténue l'effet des mots très répétés dans les longues critiques.
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=200_000,
        sublinear_tf=True,
    )


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Renvoie les modèles candidats, chacun dans un pipeline complet texte → classe."""
    return {
        "naive_bayes": Pipeline([("tfidf", _vectorizer()), ("clf", MultinomialNB(alpha=0.1))]),
        "logistic_regression": Pipeline(
            [("tfidf", _vectorizer()), ("clf", LogisticRegression(C=4.0, max_iter=2000))]
        ),
        # LinearSVC ne donne pas de probabilités : la calibration en ajoute.
        "linear_svm": Pipeline(
            [
                ("tfidf", _vectorizer()),
                ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, random_state=random_state), cv=3)),
            ]
        ),
    }
