"""
Runs the full pipeline (validate -> clean -> features -> targets ->
train Random Forest -> evaluate) for each ticker in a small multi-stock
universe, and prints a side-by-side comparison table.

Reuses all existing modules from src/ - no duplicated logic.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.features.technical_indicators import build_all_features  # noqa: E402
from src.ml.baseline_models import (  # noqa: E402
    evaluate_predictions,
    get_feature_columns,
    naive_baseline_predict,
    train_random_forest,
)
from src.ml.splitting import time_aware_split  # noqa: E402
from src.preprocessing.cleaning import clean_ohlcv, save_interim_data  # noqa: E402
from src.targets.target_engineering import (  # noqa: E402
    add_binary_direction_target,
    add_future_return_target,
    drop_undefined_targets,
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
HORIZON = 5


def latest_raw_file_for(ticker: str) -> Path:
    matches = sorted(RAW_DIR.glob(f"{ticker}_*.csv"))
    matches = [m for m in matches if not m.name.endswith("_metadata.json")]
    if not matches:
        raise FileNotFoundError(f"No raw data found for {ticker}")
    return matches[-1]


def process_ticker(ticker: str) -> dict:
    """Run the full pipeline for one ticker, return its evaluation metrics."""
    raw_path = latest_raw_file_for(ticker)
    df = pd.read_csv(raw_path)

    # Clean
    df = clean_ohlcv(df)
    save_interim_data(df, ticker)

    # Features
    df = build_all_features(df)

    # Targets
    df = add_future_return_target(df, horizon=HORIZON)
    df = add_binary_direction_target(df, horizon=HORIZON)

    target_col = f"target_direction_{HORIZON}d"
    df, _ = drop_undefined_targets(df, target_col)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    # Save labeled dataset per ticker (for later API use)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{ticker}_labeled.csv", index=False)

    # Time-aware split + train + evaluate
    train_df, val_df, _ = time_aware_split(df)
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]

    naive_preds = naive_baseline_predict(train_df, len(val_df))
    naive_accuracy = (naive_preds == y_val.values).mean()

    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_predictions(y_val, preds, proba)
    metrics["ticker"] = ticker
    metrics["rows"] = len(df)
    metrics["naive_accuracy"] = naive_accuracy
    metrics["val_start"] = str(val_df["Date"].min())
    metrics["val_end"] = str(val_df["Date"].max())
    return metrics


def main() -> None:
    all_results = []
    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")
        try:
            metrics = process_ticker(ticker)
            all_results.append(metrics)
            print(f"  Done. Accuracy={metrics['accuracy']:.4f}, ROC-AUC={metrics.get('roc_auc', float('nan')):.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n" + "=" * 70)
    print("MULTI-STOCK MODEL COMPARISON (Random Forest, validation set)")
    print("=" * 70)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    results_df = pd.DataFrame(all_results).set_index("ticker")
    print(results_df.round(4))


if __name__ == "__main__":
    main()  