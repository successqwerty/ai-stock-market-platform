"""Tests for src/preprocessing/validation.py"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing.validation import validate_ohlcv


def _base_df() -> pd.DataFrame:
    """A minimal, valid 3-row OHLCV DataFrame for testing."""
    return pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "Ticker": ["TEST", "TEST", "TEST"],
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [104.0, 105.0, 106.0],
        "Adj Close": [104.0, 105.0, 106.0],
        "Volume": [1000, 1100, 1200],
    })


def test_valid_data_has_no_issues():
    df = _base_df()
    report = validate_ohlcv(df, "TEST")
    assert report.issues["invalid_ohlc_relationships"] == 0
    assert report.issues["negative_volume"] == 0
    assert report.issues["non_positive_prices"] == 0


def test_detects_invalid_ohlc_high_below_low():
    df = _base_df()
    df.loc[0, "High"] = 50.0  # High < Low -> invalid
    report = validate_ohlcv(df, "TEST")
    assert report.issues["invalid_ohlc_relationships"] == 1


def test_detects_negative_price():
    df = _base_df()
    df.loc[0, "Close"] = -10.0
    report = validate_ohlcv(df, "TEST")
    assert report.issues["non_positive_prices"] == 1


def test_detects_negative_volume():
    df = _base_df()
    df.loc[0, "Volume"] = -500
    report = validate_ohlcv(df, "TEST")
    assert report.issues["negative_volume"] == 1


def test_detects_duplicate_dates():
    df = _base_df()
    df.loc[3] = df.loc[0]  # duplicate the first row entirely
    report = validate_ohlcv(df, "TEST")
    assert report.issues["duplicate_dates"] == 1