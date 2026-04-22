from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from src.data.load_data import load_dataset, split_features_target
from src.evaluation.metrics import evaluation_dataframe


RESULTS_PATH = Path("outputs/results/arima_results.csv")
PREDICTIONS_PATH = Path("outputs/results/arima_predictions.csv")


def train_test_split_time_series(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def get_arima_features(X_train, X_test):
    arima_cols = ["GOOGL", "IBM", "DEXJPUS", "DJIA", "VIXCLS"]
    arima_cols = [col for col in arima_cols if col in X_train.columns]

    X_train_arima = X_train[arima_cols].copy()
    X_test_arima = X_test[arima_cols].copy()

    return X_train_arima, X_test_arima, arima_cols


def fit_arima_model(y_train, X_train_arima, order=(1, 0, 0)):
    model = ARIMA(endog=y_train, exog=X_train_arima, order=order)
    fitted_model = model.fit()
    return fitted_model


def evaluate_arima(fitted_model, y_train, y_test, X_test_arima):
    train_pred = pd.Series(fitted_model.fittedvalues.values, index=y_train.index)

    test_forecast = fitted_model.forecast(steps=len(y_test), exog=X_test_arima)
    test_pred = pd.Series(test_forecast.values, index=y_test.index)

    train_df = pd.DataFrame(
        {"actual": y_train.values, "predicted": train_pred.values},
        index=y_train.index,
    ).dropna()

    test_df = pd.DataFrame(
        {"actual": y_test.values, "predicted": test_pred.values},
        index=y_test.index,
    ).dropna()

    if len(test_df) == 0:
        raise ValueError("ARIMA evaluation dataframe is empty after prediction alignment.")

    train_mse = mean_squared_error(train_df["actual"], train_df["predicted"])
    results_df = evaluation_dataframe(
        "ARIMA",
        test_df["actual"],
        test_df["predicted"],
        train_mse=train_mse,
    )

    return results_df, test_df


def save_outputs(results_df, predictions_df):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH)


if __name__ == "__main__":
    df = load_dataset()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)
    X_train_arima, X_test_arima, arima_cols = get_arima_features(X_train, X_test)

    print("ARIMA exogenous features used:", arima_cols)

    fitted_model = fit_arima_model(y_train, X_train_arima, order=(1, 0, 0))
    results_df, predictions_df = evaluate_arima(
        fitted_model, y_train, y_test, X_test_arima
    )

    save_outputs(results_df, predictions_df)

    print("\nARIMA results:")
    print(results_df)