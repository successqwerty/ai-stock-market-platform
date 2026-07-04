"""
Phase 3 script: clean the most recently downloaded raw data for
AAPL and ^GSPC, save to data/interim/, and print a summary.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.preprocessing.cleaning import clean_ohlcv, save_interim_data  # noqa: E402

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def latest_file_for(ticker: str) -> Path:
    matches = sorted(RAW_DATA_DIR.glob(f"{ticker}_*.csv"))
    matches = [m for m in matches if not m.name.endswith("_metadata.json")]
    if not matches:
        raise FileNotFoundError(f"No raw data file found for {ticker} in {RAW_DATA_DIR}")
    return matches[-1]


def main() -> None:
    for ticker in ["AAPL", "^GSPC"]:
        raw_path = latest_file_for(ticker)
        df = pd.read_csv(raw_path)

        cleaned = clean_ohlcv(df)
        interim_path = save_interim_data(cleaned, ticker)

        print(f"\n{ticker}: {len(df)} raw rows -> {len(cleaned)} cleaned rows")
        print(f"Saved to: {interim_path}")
        print(cleaned.dtypes)


if __name__ == "__main__":
    main()