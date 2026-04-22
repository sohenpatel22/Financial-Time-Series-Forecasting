from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(actual, predicted, title="Actual vs Predicted Returns", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_cumulative_returns(history_df, title="Strategy vs Buy-and-Hold", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["strategy_cumulative_return"], label="Strategy")
    plt.plot(history_df["buy_and_hold_cumulative_return"], label="Buy and Hold")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_model_comparison(results_df, metric="test_mse", title=None, save_path=None):
    if metric not in results_df.columns:
        raise ValueError(f"Column '{metric}' not found in results dataframe.")

    plot_df = results_df.sort_values(metric).copy()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["model"], plot_df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(title if title is not None else f"Model Comparison ({metric})")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def load_results_csv(path):
    return pd.read_csv(path)
    