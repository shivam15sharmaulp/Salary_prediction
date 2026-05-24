import numpy as np
from sklearn.ensemble import RandomForestClassifier

from starter.starter.ml.model import train_model, compute_model_metrics, inference


def test_train_model_returns_fitted_model(training_data):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    trained_model = train_model(model, training_data["X_train"], training_data["y_train"])

    assert trained_model is model
    assert hasattr(trained_model, "classes_")


def test_compute_model_metrics_returns_expected_values():
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 2 / 3
    assert fbeta == 0.8


def test_inference_returns_predictions(trained_model, training_data):
    preds = inference(trained_model, training_data["X_test"])

    assert len(preds) == len(training_data["X_test"])
    assert set(np.unique(preds)).issubset({0, 1})