"""
Data cleaning for OHLCV market data.

Takes raw data (already checked by validation.py) and produces a
clean, analysis-ready version saved to data/interim/. Raw source
files in data/raw/ are never modified.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

INTERIM_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw OHLCV DataFrame.

    Steps:
        1. Parse Date column to datetime.
        2. Drop exact duplicate rows.
        3. Sort chronologically.
        4. Reset index.
        5. Enforce numeric dtypes on price/volume columns.

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d exact duplicate rows", dropped)

    df = df.sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If coercion introduced any NaNs, that's a real problem - surface it loudly
    n_nans = df[numeric_cols].isna().sum().sum()
    if n_nans > 0:
        logger.warning("Cleaning introduced %d NaNs during numeric coercion", n_nans)

    return df


def save_interim_data(df: pd.DataFrame, ticker: str) -> Path:
    """Save cleaned data to data/interim/, overwriting any previous
    cleaned version for this ticker (interim data is reproducible
    from raw data, so overwriting here is safe, unlike raw data)."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize ticker for filesystem (handles tickers like ^GSPC)
    safe_ticker = ticker.replace("^", "").replace("/", "-")
    filepath = INTERIM_DATA_DIR / f"{safe_ticker}_clean.csv"

    df.to_csv(filepath, index=False)
    logger.info("Saved cleaned data to %s (%d rows)", filepath, len(df))

    return filepath