"""
Phase 5 script: build technical indicator features for AAPL and save
to data/processed/.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.features.technical_indicators import build_all_features  # noqa: E402

INTERIM_DIR = Path(__file__).resolve().parents[1] / "data" / "interim"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    df = pd.read_csv(INTERIM_DIR / "AAPL_clean.csv", parse_dates=["Date"])
    df = build_all_features(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "AAPL_features.csv"
    df.to_csv(out_path, index=False)

    print(f"Built {df.shape[1]} columns, {df.shape[0]} rows")
    print(f"Saved to: {out_path}")
    print("\nColumns:", list(df.columns))
    print("\nLast 3 rows (should have no NaNs, since rolling windows have filled in by then):")
    print(df.tail(3).T)


if __name__ == "__main__":
    main()