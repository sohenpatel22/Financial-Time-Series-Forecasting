import numpy as np
import pandas as pd


def generate_signals(predictions, threshold=0.0):
    predictions = np.array(predictions)

    signals = np.where(predictions > threshold, 1, -1)
    return signals


def strategy_returns(actual_returns, signals):
    actual_returns = np.array(actual_returns)
    signals = np.array(signals)

    returns = signals * actual_returns
    return returns


def cumulative_returns(returns):
    returns = np.array(returns)
    cumulative = (1 + returns).cumprod() - 1
    return cumulative


def max_drawdown(cumulative_ret):
    cumulative_ret = np.array(cumulative_ret)

    wealth = 1 + cumulative_ret
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max

    return drawdown.min()


def sharpe_ratio(returns):
    returns = np.array(returns)

    if returns.std() == 0:
        return 0.0

    return returns.mean() / returns.std()


def backtest_predictions(actual_returns, predicted_returns, threshold=0.0):
    actual_returns = np.array(actual_returns)
    predicted_returns = np.array(predicted_returns)

    signals = generate_signals(predicted_returns, threshold=threshold)

    strat_returns = strategy_returns(actual_returns, signals)
    buy_hold_returns = actual_returns.copy()

    strat_cum = cumulative_returns(strat_returns)
    buy_hold_cum = cumulative_returns(buy_hold_returns)

    results = {
        "strategy_total_return": strat_cum[-1],
        "buy_and_hold_return": buy_hold_cum[-1],
        "strategy_sharpe": sharpe_ratio(strat_returns),
        "buy_and_hold_sharpe": sharpe_ratio(buy_hold_returns),
        "strategy_max_drawdown": max_drawdown(strat_cum),
        "buy_and_hold_max_drawdown": max_drawdown(buy_hold_cum),
    }

    history = pd.DataFrame({
        "actual_return": actual_returns,
        "predicted_return": predicted_returns,
        "signal": signals,
        "strategy_return": strat_returns,
        "buy_and_hold_return": buy_hold_returns,
        "strategy_cumulative_return": strat_cum,
        "buy_and_hold_cumulative_return": buy_hold_cum,
    })

    return results, history
    