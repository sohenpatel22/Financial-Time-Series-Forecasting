# Financial Time-Series Forecasting

### Forecasting Microsoft Weekly Returns Using Market, Macro, and Technical Features

## Overview

This project builds an end-to-end time-series forecasting pipeline to predict **future 5-day returns of Microsoft (MSFT)** using:

- historical MSFT returns
- correlated equity assets
- foreign exchange rates
- market indices
- technical indicators
- macro and alternative asset signals

The goal of the project is not just to fit one forecasting model, but to compare how different modeling approaches behave on financial time-series data under a consistent workflow.

The project includes:

- data collection and feature construction
- exploratory data analysis
- feature selection analysis
- baseline machine learning models
- ARIMA with exogenous variables
- LSTM-based sequence modeling
- model comparison using regression and directional metrics
- simple strategy-style evaluation of predictions

---

## Motivation

Financial forecasting is a difficult problem because markets are noisy, non-stationary, and influenced by many interacting factors.

I built this project to better understand:

- how to frame stock return prediction as a supervised learning problem
- how different feature groups contribute to predictive performance
- whether classical statistical models, machine learning models, or neural networks work better on a small financial dataset
- how forecasting quality should be evaluated beyond only regression error

Instead of treating this as a pure stock-price prediction problem, I focused on **return forecasting**, which is a more realistic and better-behaved target for modeling.

---

## Problem Statement

The target variable in this project is:

- **MSFT future 5-business-day return**

The model uses current and past information from several sources to predict the next weekly return.

This is a supervised regression task with a time-series structure, so the train-test split and validation design are handled sequentially instead of random shuffling.

---

## Features Used

The input features are constructed from multiple groups.

### 1. Correlated Equity Features
- GOOGL 5-day return
- IBM 5-day return

### 2. Currency Features
- USD/JPY 5-day return
- GBP/USD 5-day return

### 3. Market Index Features
- S&P 500 5-day return
- Dow Jones 5-day return
- VIX 5-day return

### 4. MSFT Lagged Return Features
- MSFT 5-day return
- MSFT 15-day return
- MSFT 30-day return
- MSFT 60-day return

### 5. Technical Indicators
- RSI
- MACD
- Bollinger Band Width
- ATR
- ROC

### 6. Macro / Alternative Asset Features
- Gold return
- Oil return
- 10-Year Treasury yield change
- Bitcoin return
- Rolling beta of MSFT vs S&P 500

---

## Models Compared

This project compares multiple modeling families:

### Classical Machine Learning Models
- Linear Regression
- Lasso Regression
- Elastic Net
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Support Vector Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- Random Forest Regressor
- Extra Trees Regressor
- XGBoost Regressor

### Time-Series / Sequence Models
- ARIMA with exogenous variables
- LSTM

---

## Evaluation

The models are evaluated using a **sequential train-test split** to preserve time order.

### Regression Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### Directional Metrics
- Directional Accuracy
- Up/Down prediction accuracy

### Strategy-Oriented Evaluation
A simple signal-based strategy is also tested using model predictions to check whether good regression performance translates into useful decision-making.

This makes the project more meaningful than reporting error alone.

---

## Project Structure

```text
Financial-Time-Series-Forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   └── msft_dataset.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_ml_models.py
│   │   ├── train_arima.py
│   │   └── train_lstm.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── backtest.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── plotting.py
│
├── outputs/
│   ├── figures/
│   └── results/
│
└── reports/
    └── final_report.pdf