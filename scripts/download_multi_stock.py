"""
Extends Phase 2's single-stock download to a small multi-stock
universe, reusing the existing downloader with no changes to
src/data/downloader.py.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.downloader import download_ohlcv, save_raw_data  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START_DATE = "2015-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

def main() -> None:
    for ticker in TICKERS:
        print(f"\nDownloading {ticker}...")
        df = download_ohlcv(ticker, START_DATE, END_DATE)
        filepath = save_raw_data(df, ticker)
        print(f"  {len(df)} rows saved to {filepath}")


if __name__ == "__main__":
    main() 