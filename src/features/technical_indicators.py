"""
Technical indicator feature engineering for OHLCV data.

All functions here are careful to only use information available at
or before each row's timestamp (no look-ahead). Rolling/expanding
windows in pandas naturally enforce this, but every function is
documented and tested for that property.
"""

import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Add return, momentum, and rolling min/max features."""
    df = df.copy()

    df["daily_return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    for period in [5, 10, 20]:
        df[f"return_{period}d"] = df[price_col].pct_change(periods=period)

    df["momentum_10"] = df[price_col] - df[price_col].shift(10)

    df["rolling_min_20"] = df[price_col].rolling(20).min()
    df["rolling_max_20"] = df[price_col].rolling(20).max()

    sma_20 = df[price_col].rolling(20).mean()
    df["dist_from_sma20"] = (df[price_col] - sma_20) / sma_20

    return df


def add_trend_indicators(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Add SMA, EMA, and MACD."""
    df = df.copy()

    df["sma_20"] = df[price_col].rolling(20).mean()
    df["sma_50"] = df[price_col].rolling(50).mean()

    ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
    df["ema_12"] = ema_12
    df["ema_26"] = ema_26

    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_histogram"] = macd_line - macd_signal

    return df


def add_momentum_indicators(df: pd.DataFrame, price_col: str = "Adj Close", period: int = 14) -> pd.DataFrame:
    """Add RSI and rate of change."""
    df = df.copy()

    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["roc_10"] = df[price_col].pct_change(periods=10) * 100

    return df


def add_volatility_indicators(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Add rolling volatility, ATR, and Bollinger Bands."""
    df = df.copy()

    daily_return = df[price_col].pct_change()
    df["rolling_vol_20"] = daily_return.rolling(20).std()

    # ATR (Average True Range) - needs High, Low, Close
    high_low = df["High"] - df["Low"]
    high_close_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_close_prev = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    sma_20 = df[price_col].rolling(20).mean()
    std_20 = df[price_col].rolling(20).std()
    df["bb_upper"] = sma_20 + (2 * std_20)
    df["bb_lower"] = sma_20 - (2 * std_20)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume change, rolling volume mean, volume ratio, and OBV."""
    df = df.copy()

    df["volume_change"] = df["Volume"].pct_change()
    df["rolling_vol_mean_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["rolling_vol_mean_20"]

    # On-Balance Volume: cumulative volume, signed by price direction
    price_direction = np.sign(df["Adj Close"].diff())
    df["obv"] = (price_direction * df["Volume"]).fillna(0).cumsum()

    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline in sequence."""
    df = add_price_features(df)
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_features(df)
    return df