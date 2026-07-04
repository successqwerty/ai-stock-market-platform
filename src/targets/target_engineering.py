"""
Target engineering for supervised learning on stock price data.

CRITICAL CONCEPT - Entry/Exit Timing:
    At row T, all FEATURES use only information available through T's
    close (see src/features/technical_indicators.py).

    The TARGET at row T looks FORWARD to T+horizon - this is
    intentional and is the one place in the pipeline where "future"
    data is used, because it IS the thing being predicted, not an
    input feature.

    Convention used here:
        - Entry:  at close of day T  (the day the prediction is made)
        - Exit:   at close of day T + horizon
        - Target: percentage return from entry to exit

    This means the LAST `horizon` rows of any dataset will have NaN
    targets (there's no future data yet to compute them) - these rows
    MUST be dropped before training, never filled or guessed.
"""

import numpy as np
import pandas as pd


def add_future_return_target(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Add a regression target: forward return over `horizon` trading days.

    target_return[T] = (price[T + horizon] - price[T]) / price[T]

    Args:
        df: DataFrame sorted chronologically, with a price column.
        price_col: Column to compute forward return on.
        horizon: Number of trading days ahead to look.

    Returns:
        DataFrame with a new column `target_return_{horizon}d`.
        The last `horizon` rows will have NaN (no future data exists yet).
    """
    df = df.copy()
    future_price = df[price_col].shift(-horizon)
    df[f"target_return_{horizon}d"] = (future_price - df[price_col]) / df[price_col]
    return df


def add_binary_direction_target(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Add a binary classification target: 1 if forward return is
    strictly positive, 0 otherwise (zero or negative).

    Depends on `target_return_{horizon}d` already existing; computes
    it if missing.
    """
    df = df.copy()
    return_col = f"target_return_{horizon}d"
    if return_col not in df.columns:
        df = add_future_return_target(df, price_col, horizon)

    target_col = f"target_direction_{horizon}d"
    df[target_col] = np.where(df[return_col] > 0, 1, 0)
    # Preserve NaN rows (can't know direction without a valid return)
    df.loc[df[return_col].isna(), target_col] = np.nan

    return df


def add_three_class_target(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    horizon: int = 5,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Add a three-class target: SELL (-1), HOLD (0), BUY (1).

    Threshold methodology (documented, not arbitrary):
        If `threshold` is not provided, it defaults to the rolling
        20-day return volatility at each row, scaled to the horizon.
        This means "significant" move is defined RELATIVE TO EACH
        STOCK'S OWN RECENT VOLATILITY, not a fixed number like "+1%"
        that would mean very different things for a calm stock vs a
        volatile one.

        BUY  : forward return >  +threshold
        SELL : forward return <  -threshold
        HOLD : otherwise

    Args:
        df: DataFrame sorted chronologically.
        price_col: Column to compute forward return on.
        horizon: Trading days ahead.
        threshold: Fixed threshold to use instead of the adaptive one.
            If None, uses horizon-scaled rolling volatility.
    """
    df = df.copy()
    return_col = f"target_return_{horizon}d"
    if return_col not in df.columns:
        df = add_future_return_target(df, price_col, horizon)

    if threshold is None:
        daily_return = df[price_col].pct_change()
        rolling_daily_vol = daily_return.rolling(20).std()
        # Scale daily volatility to the horizon (variance scales linearly
        # with time under a random-walk assumption, so std scales with sqrt(time))
        dynamic_threshold = rolling_daily_vol * np.sqrt(horizon)
        threshold_series = dynamic_threshold
    else:
        threshold_series = pd.Series(threshold, index=df.index)

    target_col = f"target_class_{horizon}d"
    conditions = [
        df[return_col] > threshold_series,
        df[return_col] < -threshold_series,
    ]
    choices = [1, -1]  # BUY, SELL
    df[target_col] = np.select(conditions, choices, default=0)  # default = HOLD

    # Preserve NaN where return or threshold is undefined
    invalid_mask = df[return_col].isna() | threshold_series.isna()
    df.loc[invalid_mask, target_col] = np.nan

    return df


def drop_undefined_targets(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Drop rows where the target is undefined (NaN) - i.e. the last
    `horizon` rows where no future price exists yet.

    This must always be called before training, and must be the LAST
    step before splitting into train/val/test, so the dropped rows
    are consistently excluded everywhere.
    """
    before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    dropped = before - len(df)
    return df, dropped