"""
Builds a feature-only dataset (no target column) for LIVE INFERENCE,
separate from the labeled training dataset.

Unlike AAPL_labeled.csv (which drops the last `horizon` rows because
their targets can't be computed yet), this file keeps every row up to
the most recent available trading day - since live prediction doesn't
need a target, only features.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.features.technical_indicators import build_all_features  # noqa: E402
from src.ml.baseline_models import get_feature_columns  # noqa: E402
from src.preprocessing.cleaning import clean_ohlcv  # noqa: E402

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def latest_raw_file_for(ticker: str) -> Path:
    matches = sorted(RAW_DIR.glob(f"{ticker}_*.csv"))
    matches = [m for m in matches if not m.name.endswith("_metadata.json")]
    if not matches:
        raise FileNotFoundError(f"No raw data found for {ticker}")
    return matches[-1]


def main() -> None:
    for ticker in TICKERS:
        raw_path = latest_raw_file_for(ticker)
        df = pd.read_csv(raw_path)

        df = clean_ohlcv(df)
        df = build_all_features(df)

        feature_cols = get_feature_columns(df)
        # Only drop rows where FEATURES are missing (early rolling-window
        # warmup rows) - NOT rows missing a target, since there's no
        # target column here at all.
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / f"{ticker}_live_features.csv"
        df.to_csv(out_path, index=False)

        print(f"{ticker}: {len(df)} rows (latest date: {df['Date'].max()}) -> {out_path}")


if __name__ == "__main__":
    main()