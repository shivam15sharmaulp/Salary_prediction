from pathlib import Path

from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.starter.ml.data import process_data


SLICE_OUTPUT_PATH = Path(__file__).resolve().parent / "slice_output.txt"


def _format_slice_metrics_table(slice_metrics):
    """Render slice metrics as a plain-text table."""
    headers = ["feature", "value", "precision", "recall", "fbeta"]
    rows = [
        [
            str(metric["feature"]),
            str(metric["value"]).strip(),
            f"{metric['precision']:.3f}",
            f"{metric['recall']:.3f}",
            f"{metric['fbeta']:.3f}",
        ]
        for metric in slice_metrics
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    header_row = " | ".join(
        header.ljust(width) for header, width in zip(headers, widths)
    )
    separator_row = "-+-".join("-" * width for width in widths)
    data_rows = [
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths))
        for row in rows
    ]

    return "\n".join([header_row, separator_row, *data_rows]) + "\n"



def train_model(model,X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    model : RandomForestClassifier Model to train.
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds=model.predict(X)
    return preds


def compute_slice_metrics(
    model,
    data,
    categorical_features,
    label,
    encoder,
    lb,
    output_path=SLICE_OUTPUT_PATH,
):
    """
    Compute model metrics on slices of the data for each categorical feature.

    Returns
    -------
    slice_metrics : list[dict]
        Each dict contains the feature name, slice value, and metrics.
    """
    slice_metrics = []

    for feature in categorical_features:
        for value in data[feature].unique():
            data_slice = data[data[feature] == value]

            X_slice, y_slice, _, _ = process_data(
                data_slice,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            result = {
                "feature": feature,
                "value": value,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
            slice_metrics.append(result)

            print(
                f"{feature}={value}: "
                f"precision={precision:.3f}, "
                f"recall={recall:.3f}, "
                f"fbeta={fbeta:.3f}"
            )

    output_path = Path(output_path)
    output_path.write_text(_format_slice_metrics_table(slice_metrics), encoding="utf-8")

    return slice_metrics