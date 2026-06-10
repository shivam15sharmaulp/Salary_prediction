import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from starter.starter.ml.data import process_data
from starter.starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)


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


def test_compute_slice_metrics_writes_report(tmp_path):
    data = pd.DataFrame(
        [
            {
                "age": 25,
                "workclass": " Private",
                "fnlgt": 100000,
                "education": " Bachelors",
                "education-num": 13,
                "marital-status": " Never-married",
                "occupation": " Adm-clerical",
                "relationship": " Not-in-family",
                "race": " White",
                "sex": " Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": " United-States",
                "salary": " <=50K",
            },
            {
                "age": 45,
                "workclass": " Self-emp-not-inc",
                "fnlgt": 120000,
                "education": " HS-grad",
                "education-num": 9,
                "marital-status": " Married-civ-spouse",
                "occupation": " Exec-managerial",
                "relationship": " Husband",
                "race": " White",
                "sex": " Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 50,
                "native-country": " United-States",
                "salary": " >50K",
            },
            {
                "age": 36,
                "workclass": " Private",
                "fnlgt": 95000,
                "education": " Masters",
                "education-num": 14,
                "marital-status": " Divorced",
                "occupation": " Prof-specialty",
                "relationship": " Unmarried",
                "race": " Black",
                "sex": " Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": " United-States",
                "salary": " >50K",
            },
            {
                "age": 29,
                "workclass": " State-gov",
                "fnlgt": 88000,
                "education": " Some-college",
                "education-num": 10,
                "marital-status": " Never-married",
                "occupation": " Tech-support",
                "relationship": " Own-child",
                "race": " White",
                "sex": " Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 38,
                "native-country": " Canada",
                "salary": " <=50K",
            },
        ]
    )
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        data,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    output_path = tmp_path / "slice_output.txt"
    slice_metrics = compute_slice_metrics(
        model,
        data,
        categorical_features,
        "salary",
        encoder,
        lb,
        output_path=output_path,
    )

    report = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert len(slice_metrics) > 0
    assert "feature" in report
    assert "precision" in report
    assert "workclass" in report
    assert "Private" in report