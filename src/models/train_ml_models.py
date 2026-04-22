from pathlib import Path

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from src.data.load_data import load_dataset, split_features_target
from src.evaluation.metrics import evaluation_dataframe


RESULTS_PATH = Path("outputs/results/ml_model_results.csv")


def train_test_split_time_series(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def get_models():
    models = {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "LASSO": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso())
        ]),
        "EN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet())
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor())
        ]),
        "CART": DecisionTreeRegressor(random_state=42),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]),
        "ABR": AdaBoostRegressor(random_state=42),
        "GBR": GradientBoostingRegressor(random_state=42),
        "RFR": RandomForestRegressor(random_state=42),
        "ETR": ExtraTreesRegressor(random_state=42),
        "XGB": XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
        ),
    }
    return models


def run_ml_experiments(X, y, n_splits=5):
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

    models = get_models()
    results = []

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for name, model in models.items():
        cv_scores = -cross_val_score(
            model,
            X_train,
            y_train,
            cv=tscv,
            scoring="neg_mean_squared_error",
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, train_pred)

        row_df = evaluation_dataframe(name, y_test, test_pred, train_mse=train_mse)
        row_df["cv_mse_mean"] = cv_scores.mean()
        row_df["cv_mse_std"] = cv_scores.std()

        results.append(row_df)

        print(
            f"{name}: "
            f"CV MSE={cv_scores.mean():.6f}, "
            f"Test MSE={row_df.loc[0, 'test_mse']:.6f}, "
            f"Direction Acc={row_df.loc[0, 'directional_accuracy']:.3f}"
        )

    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df[
        [
            "model",
            "cv_mse_mean",
            "cv_mse_std",
            "train_mse",
            "test_mse",
            "test_rmse",
            "test_mae",
            "directional_accuracy",
        ]
    ].sort_values("test_mse").reset_index(drop=True)

    return results_df


def save_results(results_df, save_path=RESULTS_PATH):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    df = load_dataset()
    X, y = split_features_target(df)

    results_df = run_ml_experiments(X, y)
    save_results(results_df)

    print("\nFinal model comparison:")
    print(results_df)