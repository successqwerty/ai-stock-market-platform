"""
Phase 6 script: add prediction targets to the AAPL feature dataset
and save the final labeled dataset.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.targets.target_engineering import (  # noqa: E402
    add_binary_direction_target,
    add_future_return_target,
    add_three_class_target,
    drop_undefined_targets,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "AAPL_features.csv", parse_dates=["Date"])

    horizon = 5
    df = add_future_return_target(df, horizon=horizon)
    df = add_binary_direction_target(df, horizon=horizon)
    df = add_three_class_target(df, horizon=horizon)

    print(f"Before dropping undefined targets: {len(df)} rows")

    target_col = f"target_direction_{horizon}d"
    df_final, dropped = drop_undefined_targets(df, target_col)

    print(f"Dropped {dropped} rows with undefined targets (expected: {horizon})")
    print(f"Final dataset: {len(df_final)} rows")

    print("\nBinary direction target distribution:")
    print(df_final[f"target_direction_{horizon}d"].value_counts(normalize=True))

    print("\nThree-class target distribution:")
    print(df_final[f"target_class_{horizon}d"].value_counts(normalize=True))

    out_path = PROCESSED_DIR / "AAPL_labeled.csv"
    df_final.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()