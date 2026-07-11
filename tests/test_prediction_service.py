"""Tests for backend/app/services/prediction_service.py"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.services.prediction_service import get_price_history, predict_latest


def test_predict_latest_returns_expected_keys():
    result = predict_latest("AAPL")
    expected_keys = {"ticker", "prediction_date", "probability_up", "signal", "top_contributors"}
    assert expected_keys == set(result.keys())


def test_predict_latest_probability_in_valid_range():
    result = predict_latest("AAPL")
    assert 0.0 <= result["probability_up"] <= 1.0


def test_predict_latest_signal_matches_probability():
    result = predict_latest("AAPL")
    prob = result["probability_up"]
    if prob >= 0.55:
        assert result["signal"] == "BUY"
    elif prob <= 0.45:
        assert result["signal"] == "SELL"
    else:
        assert result["signal"] == "HOLD"


def test_predict_latest_rejects_unsupported_ticker():
    with pytest.raises(ValueError):
        predict_latest("MSFT")


def test_get_price_history_returns_requested_days():
    result = get_price_history("AAPL", days=10)
    assert len(result["history"]) == 10
    assert result["ticker"] == "AAPL"


def test_get_price_history_rejects_unsupported_ticker():
    with pytest.raises(ValueError):
        get_price_history("TSLA") 