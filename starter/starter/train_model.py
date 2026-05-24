# Script to train machine learning model.

from starter.starter.ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    compute_slice_metrics,
)
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from starter.starter.ml.data import process_data
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]   # .../starter
DATA_PATH = BASE_DIR / "data" / "census.csv"
MODEL_PATH = BASE_DIR / "model" / "model.joblib"

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
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
    train, categorical_features=cat_features, label="salary", training=True
)

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kfold,
    scoring="f1"
)
print("Cross-validation scores:", scores)
print("Mean CV F1:", scores.mean())
print("Std CV F1:", scores.std())


# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# Train and save a model.
model = train_model(model, X_train, y_train)
joblib.dump(model, MODEL_PATH)

# Only load again if you actually need to.
loaded_model = joblib.load(MODEL_PATH)
y_pred = inference(loaded_model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print(f" Precision: {precision}, Recall: {recall}, fbeta: {fbeta}")

slice_metrics = compute_slice_metrics(
    model=loaded_model,
    data=test,
    categorical_features=cat_features,
    label="salary",
    encoder=encoder,
    lb=lb,
)
