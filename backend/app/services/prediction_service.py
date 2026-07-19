"""
Prediction service: loads a trained Random Forest model per ticker,
using the LABELED dataset for training (which needs a target) and the
LIVE FEATURES dataset for inference (which doesn't need a target, so
it includes every row up to the most recent trading day - unlike the
labeled dataset, which drops the last `horizon` rows).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.explainability.shap_explainer import (  # noqa: E402
    compute_shap_values,
    explain_single_prediction,
)
from src.ml.baseline_models import get_feature_columns, train_random_forest  # noqa: E402
from src.ml.splitting import time_aware_split  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SUPPORTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Cache: one trained model + feature list per ticker, so we don't
# retrain on every request - only on first request per ticker.
_model_cache: dict = {}


def _get_model_and_features(ticker: str):
    """Lazily train and cache a model for this ticker."""
    ticker = ticker.upper()

    if ticker not in SUPPORTED_TICKERS:
        raise ValueError(
            f"Ticker '{ticker}' is not supported. Supported tickers: {SUPPORTED_TICKERS}"
        )

    if ticker not in _model_cache:
        labeled_path = PROCESSED_DIR / f"{ticker}_labeled.csv"
        df = pd.read_csv(labeled_path, parse_dates=["Date"])
        feature_cols = get_feature_columns(df)
        df = df.dropna(subset=feature_cols + ["target_direction_5d"]).reset_index(drop=True)

        train_df, _, _ = time_aware_split(df)
        X_train = train_df[feature_cols]
        y_train = train_df["target_direction_5d"]

        model = train_random_forest(X_train, y_train)
        _model_cache[ticker] = (model, feature_cols)

    return _model_cache[ticker]


def _get_live_features(ticker: str) -> pd.DataFrame:
    """Load the freshest available feature row(s) - no target needed,
    so this includes data up to the most recent trading day, unlike
    the labeled training dataset."""
    live_path = PROCESSED_DIR / f"{ticker.upper()}_live_features.csv"
    if not live_path.exists():
        raise FileNotFoundError(
            f"No live features file found for {ticker}. Run scripts/build_live_features.py first."
        )
    return pd.read_csv(live_path, parse_dates=["Date"])


def predict_latest(ticker: str = "AAPL") -> dict:
    """
    Generate a prediction for the most recent available trading day,
    using a model trained on historical labeled data but applied to
    the freshest available feature row.
    """
    model, feature_cols = _get_model_and_features(ticker)
    live_df = _get_live_features(ticker)

    latest_row = live_df.iloc[[-1]]
    X_latest = latest_row[feature_cols]

    probability_up = float(model.predict_proba(X_latest)[0, 1])
    signal = "BUY" if probability_up >= 0.55 else ("SELL" if probability_up <= 0.45 else "HOLD")

    shap_values = compute_shap_values(model, X_latest)
    explanation = explain_single_prediction(shap_values, feature_cols, row_index=0, top_n=5)

    return {
        "ticker": ticker.upper(),
        "prediction_date": str(latest_row["Date"].iloc[0].date()),
        "probability_up": probability_up,
        "signal": signal,
        "top_contributors": explanation.to_dict(orient="records"),
    }


def get_price_history(ticker: str = "AAPL", days: int = 30) -> dict:
    """Return the most recent `days` of close price and volume, from
    the live features file (freshest available data)."""
    ticker = ticker.upper()
    if ticker not in SUPPORTED_TICKERS:
        raise ValueError(
            f"Ticker '{ticker}' is not supported. Supported tickers: {SUPPORTED_TICKERS}"
        )

    df = _get_live_features(ticker)
    recent = df.tail(days)

    history = [
        {
            "date": str(row["Date"].date()),
            "close": float(row["Adj Close"]),
            "volume": int(row["Volume"]),
        }
        for _, row in recent.iterrows()
    ]

    return {"ticker": ticker, "history": history}  