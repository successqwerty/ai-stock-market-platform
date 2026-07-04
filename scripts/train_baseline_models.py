"""
Milestone 1 (Part A): time-aware split, train naive baseline,
Logistic Regression, Random Forest, and XGBoost on the binary
direction target, and compare their evaluation metrics.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.ml.baseline_models import (  # noqa: E402
    evaluate_predictions,
    get_feature_columns,
    naive_baseline_predict,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.ml.splitting import time_aware_split  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "AAPL_labeled.csv", parse_dates=["Date"])

    # Drop any remaining NaN rows (early rows before rolling windows fill up)
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target_direction_5d"]).reset_index(drop=True)
    print(f"Rows after dropping NaN feature rows: {len(df)}")

    train_df, val_df, test_df = time_aware_split(df)
    print(f"Train: {len(train_df)} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"Val:   {len(val_df)} rows ({val_df['Date'].min().date()} to {val_df['Date'].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    X_train, y_train = train_df[feature_cols], train_df["target_direction_5d"]
    X_val, y_val = val_df[feature_cols], val_df["target_direction_5d"]

    results = {}

    # 1. Naive baseline
    naive_preds = naive_baseline_predict(train_df, len(val_df))
    results["Naive"] = evaluate_predictions(y_val, naive_preds)

    # 2. Logistic Regression (with scaling)
    lr_scaler, lr = train_logistic_regression(X_train, y_train)
    X_val_scaled = lr_scaler.transform(X_val)
    lr_preds = lr.predict(X_val_scaled)
    lr_proba = lr.predict_proba(X_val_scaled)[:, 1]
    results["LogisticRegression"] = evaluate_predictions(y_val, lr_preds, lr_proba)

    # 3. Random Forest
    rf = train_random_forest(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_proba = rf.predict_proba(X_val)[:, 1]
    results["RandomForest"] = evaluate_predictions(y_val, rf_preds, rf_proba)

    # 4. XGBoost
    xgb = train_xgboost(X_train, y_train)
    xgb_preds = xgb.predict(X_val)
    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    results["XGBoost"] = evaluate_predictions(y_val, xgb_preds, xgb_proba)

    print("\n" + "=" * 70)
    print("VALIDATION SET RESULTS (comparison)")
    print("=" * 70)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))


if __name__ == "__main__":
    main()