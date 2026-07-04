"""Tests for src/targets/target_engineering.py - focused heavily on
preventing look-ahead bias, since this is the highest-risk area of
the entire pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.targets.target_engineering import (
    add_binary_direction_target,
    add_future_return_target,
    add_three_class_target,
    drop_undefined_targets,
)


def _sample_df(n: int = 50) -> pd.DataFrame:
    prices = 100 + np.arange(n, dtype=float)  # simple deterministic uptrend
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "Adj Close": prices,
    })


def test_future_return_manual_calculation():
    """Verify the exact formula against hand-computed values."""
    df = _sample_df(20)
    df = add_future_return_target(df, horizon=5)

    # Row 0: price=100, price at row 5=105 -> return = (105-100)/100 = 0.05
    assert np.isclose(df.loc[0, "target_return_5d"], 0.05)


def test_last_horizon_rows_are_nan():
    """The last `horizon` rows must have NaN targets - no future data exists."""
    horizon = 5
    df = _sample_df(20)
    df = add_future_return_target(df, horizon=horizon)

    last_rows = df.tail(horizon)
    assert last_rows["target_return_5d"].isna().all()

    # And every row BEFORE that must be defined
    earlier_rows = df.iloc[:-horizon]
    assert not earlier_rows["target_return_5d"].isna().any()


def test_target_does_not_leak_into_features():
    """
    Changing a value far in the future must not change the target
    for early rows beyond what the formula defines - i.e. target at
    row 0 must depend ONLY on row 0 and row `horizon`, nothing else.
    """
    df1 = _sample_df(30)
    df2 = df1.copy()
    df2.loc[29, "Adj Close"] = 99999.0  # change only the very last row

    df1 = add_future_return_target(df1, horizon=5)
    df2 = add_future_return_target(df2, horizon=5)

    # Row 0's target uses row 5 - unaffected by changing row 29
    assert df1.loc[0, "target_return_5d"] == df2.loc[0, "target_return_5d"]


def test_binary_direction_matches_sign_of_return():
    df = _sample_df(20)
    df = add_binary_direction_target(df, horizon=5)
    valid = df.dropna(subset=["target_return_5d"])

    for _, row in valid.iterrows():
        expected = 1 if row["target_return_5d"] > 0 else 0
        assert row["target_direction_5d"] == expected


def test_three_class_uses_correct_ordering():
    """BUY should correspond to the largest returns, SELL to the smallest."""
    df = _sample_df(60)
    df = add_three_class_target(df, horizon=5)
    valid = df.dropna(subset=["target_class_5d", "target_return_5d"])

    if (valid["target_class_5d"] == 1).any() and (valid["target_class_5d"] == -1).any():
        buy_returns = valid.loc[valid["target_class_5d"] == 1, "target_return_5d"]
        sell_returns = valid.loc[valid["target_class_5d"] == -1, "target_return_5d"]
        assert buy_returns.min() > sell_returns.max()


def test_drop_undefined_targets_removes_exact_count():
    horizon = 5
    df = _sample_df(30)
    df = add_future_return_target(df, horizon=horizon)

    df_clean, dropped = drop_undefined_targets(df, "target_return_5d")
    assert dropped == horizon
    assert not df_clean["target_return_5d"].isna().any()
    assert len(df_clean) == 30 - horizon