"""
Phase 3 script: run validation on the most recently downloaded raw
CSV files for AAPL and ^GSPC, and print the reports.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.preprocessing.validation import validate_ohlcv, validate_schema  # noqa: E402

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def latest_file_for(ticker: str) -> Path:
    """Find the most recently downloaded raw CSV for a given ticker."""
    matches = sorted(RAW_DATA_DIR.glob(f"{ticker}_*.csv"))
    matches = [m for m in matches if not m.name.endswith("_metadata.json")]
    if not matches:
        raise FileNotFoundError(f"No raw data file found for {ticker} in {RAW_DATA_DIR}")
    return matches[-1]


def main() -> None:
    for ticker in ["AAPL", "^GSPC"]:
        filepath = latest_file_for(ticker)
        print(f"\nValidating {filepath.name}")
        df = pd.read_csv(filepath)

        missing_cols = validate_schema(df)
        if missing_cols:
            print(f"  SCHEMA ERROR - missing columns: {missing_cols}")
            continue

        report = validate_ohlcv(df, ticker)
        print(report.summary())


if __name__ == "__main__":
    main()