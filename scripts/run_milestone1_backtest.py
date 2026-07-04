"""
Milestone 1 (Part B): use Random Forest's probabilities (our best
validation-set model from Part A) to generate signals, then backtest
against the raw forward returns on the TEST set (final unseen data).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.backtesting.backtest_engine import (  # noqa: E402
    compute_performance_metrics,
    run_backtest,
)
from src.backtesting.signal_engine import generate_signals  # noqa: E402
from src.ml.baseline_models import get_feature_columns, train_random_forest  # noqa: E402
from src.ml.splitting import time_aware_split  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "AAPL_labeled.csv", parse_dates=["Date"])
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target_direction_5d", "target_return_5d"]).reset_index(drop=True)

    train_df, val_df, test_df = time_aware_split(df)
    # CRITICAL FIX: forward returns overlap (each row's target spans 5 days,
    # but rows are daily), so naively chaining them with cumprod() massively
    # distorts cumulative return. We resample to non-overlapping windows by
    # taking every `horizon`-th row, so each period is sequential and
    # genuinely chainable.
    horizon = 5
    test_df = test_df.iloc[::horizon].reset_index(drop=True)
    print(f"Test rows after resampling to non-overlapping {horizon}-day windows: {len(test_df)}")
    # Train on train, evaluate signal/backtest on TEST (final unseen period)
    X_train, y_train = train_df[feature_cols], train_df["target_direction_5d"]
    X_test = test_df[feature_cols]

    rf = train_random_forest(X_train, y_train)
    test_probabilities = rf.predict_proba(X_test)[:, 1]

    signals = generate_signals(test_probabilities, buy_threshold=0.55, sell_threshold=0.45)

    print("Signal distribution on TEST set:")
    print(signals.value_counts())

    backtest_df = run_backtest(
        dates=test_df["Date"],
        signals=signals,
        forward_returns=test_df["target_return_5d"],
        transaction_cost_pct=0.001,
    )

    metrics = compute_performance_metrics(backtest_df)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (Test Set - Final Unseen Period)")
    print("=" * 60)
    print(f"Test period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value}")


if __name__ == "__main__":
    main()