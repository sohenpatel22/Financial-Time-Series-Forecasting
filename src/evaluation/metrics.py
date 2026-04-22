import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    actual_direction = y_true > 0
    predicted_direction = y_pred > 0

    return (actual_direction == predicted_direction).mean()


def evaluate_predictions(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = regression_metrics(y_true, y_pred)
    metrics["directional_accuracy"] = directional_accuracy(y_true, y_pred)

    return metrics


def evaluation_dataframe(model_name, y_true, y_pred, train_mse=None):
    metrics = evaluate_predictions(y_true, y_pred)

    row = {
        "model": model_name,
        "test_mse": metrics["mse"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "directional_accuracy": metrics["directional_accuracy"],
    }

    if train_mse is not None:
        row["train_mse"] = train_mse

    return pd.DataFrame([row])