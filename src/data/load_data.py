from pathlib import Path
import pandas as pd

from src.utils.config import DATA_PATH

TARGET_COLUMN = "MSFT_pred"


def load_dataset(data_path=DATA_PATH):
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run the feature building step first."
        )

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def split_features_target(df, target_column=TARGET_COLUMN):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    return X, y


def save_dataset(df, data_path=DATA_PATH):
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path)


if __name__ == "__main__":
    df = load_dataset()
    X, y = split_features_target(df)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("Number of features:", X.shape[1])
    print("Target name:", y.name)