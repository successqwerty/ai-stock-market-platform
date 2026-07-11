"""
Prediction service: loads the trained Random Forest model, builds
features for the latest available data, and returns a prediction
with a SHAP-based explanation.

Kept separate from the API route (backend/app/api/predictions.py) so
the actual logic is testable independent of FastAPI/HTTP concerns -
a standard layered-architecture practice.
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

# NOTE: for now we retrain the model on startup rather than loading a
# saved artifact - this keeps things simple while we only support one
# ticker (AAPL). A future improvement would persist the trained model
# (e.g. via joblib) so the API doesn't retrain on every restart.
_model = None
_feature_cols = None
_full_df = None


def _get_model_and_data():
    """Lazily train the model once, cache it for subsequent requests."""
    global _model, _feature_cols, _full_df

    if _model is None:
        df = pd.read_csv(PROCESSED_DIR / "AAPL_labeled.csv", parse_dates=["Date"])
        feature_cols = get_feature_columns(df)
        df = df.dropna(subset=feature_cols + ["target_direction_5d"]).reset_index(drop=True)

        train_df, _, _ = time_aware_split(df)
        X_train = train_df[feature_cols]
        y_train = train_df["target_direction_5d"]

        _model = train_random_forest(X_train, y_train)
        _feature_cols = feature_cols
        _full_df = df

    return _model, _feature_cols, _full_df


def predict_latest(ticker: str = "AAPL") -> dict:
    """
    Generate a prediction for the most recent available row of data.

    NOTE: currently only AAPL is supported, since that's the only
    ticker we've built a full feature/target pipeline for.
    """
    if ticker.upper() != "AAPL":
        raise ValueError(f"Ticker '{ticker}' is not supported yet. Only AAPL is available.")

    model, feature_cols, df = _get_model_and_data()

    latest_row = df.iloc[[-1]]
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
    """Return the most recent `days` of close price and volume."""
    if ticker.upper() != "AAPL":
        raise ValueError(f"Ticker '{ticker}' is not supported yet. Only AAPL is available.")

    _, _, df = _get_model_and_data()
    recent = df.tail(days)

    history = [
        {
            "date": str(row["Date"].date()),
            "close": float(row["Adj Close"]),
            "volume": int(row["Volume"]),
        }
        for _, row in recent.iterrows()
    ]

    return {"ticker": ticker.upper(), "history": history} 