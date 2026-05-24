import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import train_model


@pytest.fixture
def training_data():
    X_train = np.array([
        [25, 0],
        [30, 1],
        [45, 0],
        [35, 1],
        [50, 0],
        [28, 1],
    ])
    y_train = np.array([0, 1, 0, 1, 0, 1])

    X_test = np.array([
        [40, 1],
        [23, 0],
    ])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
    }


@pytest.fixture
def trained_model(training_data):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    return train_model(model, training_data["X_train"], training_data["y_train"])