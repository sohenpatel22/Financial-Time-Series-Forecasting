from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from src.data.load_data import load_dataset, split_features_target
from src.evaluation.metrics import evaluation_dataframe


RESULTS_PATH = Path("outputs/results/lstm_results.csv")
PREDICTIONS_PATH = Path("outputs/results/lstm_predictions.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_test_split_time_series(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def create_sequences(X, y, seq_len=5):
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])

    return np.array(X_seq), np.array(y_seq)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def prepare_data(X_train, X_test, y_train, y_test, seq_len=5):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, seq_len)

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq


def train_model(model, X_train_seq, y_train_seq, epochs=50, lr=0.001):
    model.to(DEVICE)

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")


def evaluate_lstm(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq):
    model.eval()

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        train_pred = model(X_train_tensor).cpu().numpy().flatten()
        test_pred = model(X_test_tensor).cpu().numpy().flatten()

    train_mse = mean_squared_error(y_train_seq, train_pred)

    results_df = evaluation_dataframe(
        "LSTM",
        y_test_seq,
        test_pred,
        train_mse=train_mse,
    )

    predictions_df = pd.DataFrame(
        {
            "actual": y_test_seq,
            "predicted": test_pred,
        }
    )

    return results_df, predictions_df


def save_outputs(results_df, predictions_df):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)


if __name__ == "__main__":
    df = load_dataset()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = prepare_data(
        X_train, X_test, y_train, y_test, seq_len=5
    )

    model = LSTMModel(input_size=X_train.shape[1])
    train_model(model, X_train_seq, y_train_seq, epochs=50)

    results_df, predictions_df = evaluate_lstm(
        model, X_train_seq, y_train_seq, X_test_seq, y_test_seq
    )
    save_outputs(results_df, predictions_df)

    print("\nLSTM results:")
    print(results_df)