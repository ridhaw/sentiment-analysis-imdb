import pytest

from sentiment.models import build_models
from sentiment.predict import predict

POSITIVE = [
    "a wonderful movie with brilliant acting",
    "great story and great music, loved it",
    "brilliant film, wonderful cast",
    "i loved this great and wonderful film",
    "excellent direction and brilliant script",
    "loved every minute, excellent movie",
]
NEGATIVE = [
    "a terrible movie with awful acting",
    "boring story and bad music, hated it",
    "awful film, terrible cast",
    "i hated this boring and awful film",
    "bad direction and terrible script",
    "hated every minute, boring movie",
]
TEXTS = POSITIVE + NEGATIVE
LABELS = [1] * len(POSITIVE) + [0] * len(NEGATIVE)


@pytest.mark.parametrize("name", ["naive_bayes", "logistic_regression", "linear_svm"])
def test_each_model_learns_obvious_sentiment(name):
    model = build_models()[name]
    model.set_params(tfidf__min_df=1)
    model.fit(TEXTS, LABELS)

    results = predict(["wonderful brilliant film", "awful boring film"], model)

    assert [r["sentiment"] for r in results] == ["positif", "négatif"]
    assert all(0.0 <= r["positive_probability"] <= 1.0 for r in results)
