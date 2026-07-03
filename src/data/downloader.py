"""
Market data downloader.

Downloads daily OHLCV data for a given ticker over a date range using
yfinance, validates the response, and saves it as raw, unmodified data.

Raw data is never overwritten silently: each download is saved with a
timestamped filename plus a matching metadata file.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Project root is two levels up from this file (src/data/downloader.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


class DataValidationError(Exception):
    """Raised when downloaded data fails basic sanity checks."""


def download_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: End date in "YYYY-MM-DD" format.
        max_retries: Number of download attempts before giving up.

    Returns:
        DataFrame with columns: Date, Ticker, Open, High, Low, Close,
        Adj Close, Volume.

    Raises:
        DataValidationError: If data cannot be downloaded or fails
            basic validation after all retries.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading %s from %s to %s (attempt %d/%d)",
                ticker, start_date, end_date, attempt, max_retries,
            )
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )

            _validate_raw_data(df, ticker)

            # Flatten yfinance's multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df["Ticker"] = ticker

            # Reorder columns for consistency
            expected_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            df = df[[c for c in expected_cols if c in df.columns]]

            logger.info("Downloaded %d rows for %s", len(df), ticker)
            return df

        except Exception as e:  # noqa: BLE001 - we deliberately catch broadly to retry
            last_error = e
            logger.warning("Attempt %d failed for %s: %s", attempt, ticker, e)

    raise DataValidationError(
        f"Failed to download valid data for {ticker} after {max_retries} attempts"
    ) from last_error


def _validate_raw_data(df: pd.DataFrame, ticker: str) -> None:
    """Basic sanity checks on data returned by the API, before we trust it."""
    if df is None or df.empty:
        raise DataValidationError(f"No data returned for {ticker}")

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    actual_cols = set(df.columns.get_level_values(0)) if isinstance(df.columns, pd.MultiIndex) else set(df.columns)
    missing = required_cols - actual_cols
    if missing:
        raise DataValidationError(f"Missing expected columns for {ticker}: {missing}")


def save_raw_data(df: pd.DataFrame, ticker: str) -> Path:
    """
    Save raw data to data/raw/ with a timestamped filename plus a
    metadata JSON file. Never overwrites an existing file silently.

    Returns:
        Path to the saved CSV file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{ticker}_{timestamp}.csv"
    filepath = RAW_DATA_DIR / filename

    if filepath.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {filepath}")

    df.to_csv(filepath, index=False)
    logger.info("Saved raw data to %s", filepath)

    metadata = {
        "ticker": ticker,
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": {
            "start": str(df["Date"].min()),
            "end": str(df["Date"].max()),
        },
        "downloaded_at_utc": timestamp,
        "source": "yfinance",
    }
    metadata_path = RAW_DATA_DIR / f"{ticker}_{timestamp}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", metadata_path)

    return filepath