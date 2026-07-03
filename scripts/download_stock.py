"""
Phase 2 script: download one stock and one benchmark index,
save raw data, and print a summary.
"""

import sys
from pathlib import Path

# Allow running this script directly without installing the package
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.downloader import download_ohlcv, save_raw_data  # noqa: E402


def main() -> None:
    ticker = "AAPL"
    benchmark = "^GSPC"  # S&P 500 index
    start_date = "2015-01-01"
    end_date = "2025-01-01"

    for symbol in [ticker, benchmark]:
        df = download_ohlcv(symbol, start_date, end_date)
        filepath = save_raw_data(df, symbol)

        print("-" * 50)
        print(f"Ticker: {symbol}")
        print(f"Rows: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Saved to: {filepath}")
        print(df.head(3))


if __name__ == "__main__":
    main()