import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import ta

from src.data.load_data import save_dataset
from src.utils.config import DATA_PATH, START_DATE, END_DATE, RETURN_PERIOD


def download_market_data(start, end):
    stk_tickers = ["MSFT", "IBM", "GOOGL"]
    for attempt in range(3):
        stk_data = yf.download(stk_tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
        stk_data = stk_data.ffill()
        stk_data.columns = stk_tickers
        if not stk_data.empty:
            break
    
    for ticker in stk_tickers:
        if stk_data[ticker].isna().all():
            print(f"Retrying {ticker} individually...")
            fallback = yf.download(ticker, start=start, end=end, auto_adjust=False)["Adj Close"]
            stk_data[ticker] = fallback.values

    ccy_tickers = ["DEXJPUS", "DEXUSUK"]
    ccy_data = web.DataReader(ccy_tickers, "fred", start, end).ffill()

    idx_tickers = ["SP500", "DJIA", "VIXCLS"]
    idx_data = web.DataReader(idx_tickers, "fred", start, end).ffill()

    msft_raw = yf.download("MSFT", start=start, end=end, auto_adjust=False).ffill()
    spy_raw = yf.download("SPY", start=start, end=end, auto_adjust=False).ffill()
    gold_raw = yf.download("GC=F", start=start, end=end, auto_adjust=False).ffill()
    oil_raw = yf.download("CL=F", start=start, end=end, auto_adjust=False).ffill()
    btc_raw = yf.download("BTC-USD", start=start, end=end, auto_adjust=False).ffill()
    treasury = web.DataReader("DGS10", "fred", start, end).ffill()

    return {
        "stocks": stk_data,
        "currencies": ccy_data,
        "indices": idx_data,
        "msft_raw": msft_raw,
        "spy_raw": spy_raw,
        "gold_raw": gold_raw,
        "oil_raw": oil_raw,
        "btc_raw": btc_raw,
        "treasury": treasury,
    }


def build_target_and_features(data, return_period=RETURN_PERIOD):
    stk_data = data["stocks"]
    ccy_data = data["currencies"]
    idx_data = data["indices"]
    msft_raw = data["msft_raw"]
    spy_raw = data["spy_raw"]
    gold_raw = data["gold_raw"]
    oil_raw = data["oil_raw"]
    btc_raw = data["btc_raw"]
    treasury = data["treasury"]

    #normalize
    stk_data.index = pd.to_datetime(stk_data.index).normalize()
    ccy_data.index = pd.to_datetime(ccy_data.index).normalize()
    idx_data.index = pd.to_datetime(idx_data.index).normalize()
    msft_raw.index = pd.to_datetime(msft_raw.index).normalize()
    spy_raw.index = pd.to_datetime(spy_raw.index).normalize()
    gold_raw.index = pd.to_datetime(gold_raw.index).normalize()
    oil_raw.index = pd.to_datetime(oil_raw.index).normalize()
    btc_raw.index = pd.to_datetime(btc_raw.index).normalize()
    treasury.index = pd.to_datetime(treasury.index).normalize()

    close = msft_raw["Adj Close"].squeeze()
    high = msft_raw["High"].squeeze()
    low = msft_raw["Low"].squeeze()
    spy_close = spy_raw["Adj Close"].squeeze()

    y = np.log(stk_data["MSFT"]).diff(return_period).shift(-return_period)
    y.name = "MSFT_pred"

    x1 = np.log(stk_data[["GOOGL", "IBM"]]).diff(return_period)
    x2 = np.log(ccy_data).diff(return_period)
    x3 = np.log(idx_data).diff(return_period)

    x4 = pd.concat(
        [np.log(stk_data["MSFT"]).diff(i) for i in [return_period, 15, 30, 60]],
        axis=1,
    )
    x4.columns = ["MSFT_DT", "MSFT_3DT", "MSFT_6DT", "MSFT_12DT"]

    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd = ta.trend.MACD(close=close).macd()
    bb_width = ta.volatility.BollingerBands(close=close, window=20).bollinger_wband()
    atr = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    roc = ta.momentum.ROCIndicator(close=close, window=10).roc()

    x5 = pd.concat([rsi, macd, bb_width, atr, roc], axis=1)
    x5.columns = ["RSI", "MACD", "BB_Width", "ATR", "ROC"]

    gold_return = np.log(gold_raw["Adj Close"].squeeze()).diff(return_period).rename("Gold_Return")
    oil_return = np.log(oil_raw["Adj Close"].squeeze()).diff(return_period).rename("Oil_Return")
    btc_return = np.log(btc_raw["Adj Close"].squeeze()).diff(return_period).rename("BTC_Return")

    treasury_change = treasury.diff(return_period)
    treasury_change.columns = ["Treasury_Change"]

    msft_log_ret = np.log(close).diff()
    spy_log_ret = np.log(spy_close).diff()
    rolling_beta = (
        msft_log_ret.rolling(52).cov(spy_log_ret) / spy_log_ret.rolling(52).var()
    ).rename("Rolling_Beta")

    x6 = pd.concat(
        [gold_return, oil_return, treasury_change, btc_return, rolling_beta],
        axis=1,
    )

    for df in [x1, x2, x3, x4, x5, x6]:
        df.index = pd.to_datetime(df.index).normalize()
    y.index = pd.to_datetime(y.index).normalize()

    X = pd.concat([x1, x2, x3, x4, x5, x6], axis=1)
    dataset = pd.concat([y, X], axis=1).dropna().iloc[::return_period, :]

    return dataset


def build_dataset(start=START_DATE, end=END_DATE, save_path=DATA_PATH):
    data = download_market_data(start, end)
    dataset = build_target_and_features(data)

    save_dataset(dataset, save_path)
    return dataset


if __name__ == "__main__":
    dataset = build_dataset()

    print("Dataset built and saved successfully.")
    print("Shape:", dataset.shape)
    print("Columns:")
    print(list(dataset.columns))
    print(dataset.head())