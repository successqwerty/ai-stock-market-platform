"""Tests for src/features/technical_indicators.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.technical_indicators import (
    add_momentum_indicators,
    add_price_features,
    add_trend_indicators,
    build_all_features,
)


def _sample_df(n: int = 100) -> pd.DataFrame:
    """Synthetic OHLCV data for testing - random walk, always positive."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + returns)

    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Date": dates,
        "Ticker": "TEST",
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Adj Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    })
    return df


def test_no_lookahead_in_returns():
    """Changing a future row must not change a past feature value."""
    df1 = _sample_df(50)
    df2 = df1.copy()
    df2.loc[49, "Adj Close"] = 99999.0  # drastically change the LAST row only

    result1 = add_price_features(df1)
    result2 = add_price_features(df2)

    # Every row except the last must be identical - changing the future
    # must not affect the past.
    pd.testing.assert_series_equal(
        result1["daily_return"].iloc[:-1],
        result2["daily_return"].iloc[:-1],
    )


def test_rsi_bounded_between_0_and_100():
    df = _sample_df(100)
    df = add_momentum_indicators(df)
    valid_rsi = df["rsi_14"].dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_macd_histogram_equals_macd_minus_signal():
    df = _sample_df(100)
    df = add_trend_indicators(df)
    diff = df["macd"] - df["macd_signal"]
    pd.testing.assert_series_equal(
        diff, df["macd_histogram"], check_names=False
    )


def test_build_all_features_no_infinite_values():
    df = _sample_df(100)
    result = build_all_features(df)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    assert not np.isinf(result[numeric_cols]).any().any()


def test_build_all_features_last_row_has_no_nans():
    """By the last row, all rolling windows should be fully populated."""
    df = _sample_df(100)
    result = build_all_features(df)
    last_row = result.iloc[-1]
    assert not last_row.isna().any()