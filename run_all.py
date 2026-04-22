import pandas as pd

from src.data.build_features import build_dataset
from src.models.train_ml_models import run_ml_experiments, save_results as save_ml_results
from src.models.train_arima import (
    train_test_split_time_series as arima_split,
    get_arima_features,
    fit_arima_model,
    evaluate_arima,
    save_outputs as save_arima_outputs,
)
from src.models.train_lstm import (
    train_test_split_time_series as lstm_split,
    prepare_data,
    LSTMModel,
    train_model,
    evaluate_lstm,
    save_outputs as save_lstm_outputs,
)
from src.data.load_data import split_features_target


def main():
    print("Building dataset...")
    df = build_dataset()
    X, y = split_features_target(df)
    print("Dataset ready.")
    print(f"Shape: {df.shape}")

    print("\nRunning machine learning models...")
    ml_results = run_ml_experiments(X, y)
    save_ml_results(ml_results)
    print("ML model results saved.")

    print("\nRunning ARIMA model...")
    X_train_a, X_test_a, y_train_a, y_test_a = arima_split(X, y)
    X_train_arima, X_test_arima, arima_cols = get_arima_features(X_train_a, X_test_a)
    print("ARIMA exogenous features used:", arima_cols)

    arima_model = fit_arima_model(y_train_a, X_train_arima)
    arima_results, arima_predictions = evaluate_arima(
        arima_model, y_train_a, y_test_a, X_test_arima
    )
    save_arima_outputs(arima_results, arima_predictions)
    print("ARIMA results saved.")

    print("\nRunning LSTM model...")
    X_train_l, X_test_l, y_train_l, y_test_l = lstm_split(X, y)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = prepare_data(
        X_train_l, X_test_l, y_train_l, y_test_l
    )

    lstm_model = LSTMModel(input_size=X_train_l.shape[1])
    train_model(lstm_model, X_train_seq, y_train_seq, epochs=50)

    lstm_results, lstm_predictions = evaluate_lstm(
        lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq
    )
    save_lstm_outputs(lstm_results, lstm_predictions)
    print("LSTM results saved.")

    print("\nCombining results...")
    final_results = pd.concat(
        [ml_results, arima_results, lstm_results],
        ignore_index=True,
        sort=False,
    )

    final_results = final_results.sort_values("test_mse").reset_index(drop=True)
    final_results.to_csv("outputs/results/final_model_comparison.csv", index=False)

    print("\nAll experiments completed.")
    print("\nFinal model comparison:")
    print(final_results)


if __name__ == "__main__":
    main()