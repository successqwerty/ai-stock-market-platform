"""
Phase 16: SHAP explainability on the Random Forest model (our best
classical baseline). Produces global feature importance and a local
explanation for one example prediction.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.explainability.shap_explainer import (  # noqa: E402
    compute_shap_values,
    explain_single_prediction,
    get_global_feature_importance,
)
from src.ml.baseline_models import get_feature_columns, train_random_forest  # noqa: E402
from src.ml.splitting import time_aware_split  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "AAPL_labeled.csv", parse_dates=["Date"])
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target_direction_5d"]).reset_index(drop=True)

    train_df, val_df, test_df = time_aware_split(df)
    X_train, y_train = train_df[feature_cols], train_df["target_direction_5d"]
    X_val = val_df[feature_cols]

    rf = train_random_forest(X_train, y_train)

    print("Computing SHAP values on validation set...")
    shap_values = compute_shap_values(rf, X_val)

    # Global feature importance
    importance_df = get_global_feature_importance(shap_values, feature_cols)
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES (global)")
    print("=" * 60)
    print(importance_df.head(10).to_string(index=False))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Global importance bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    top_features = importance_df.head(10)
    ax.barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Feature Importance (SHAP) - Random Forest")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_global_importance.png", dpi=150)
    print(f"\nSaved global importance plot to {OUTPUTS_DIR / 'shap_global_importance.png'}")

    # Local explanation for one example prediction
    example_idx = 0
    local_explanation = explain_single_prediction(shap_values, feature_cols, example_idx)

    print("\n" + "=" * 60)
    print(f"EXAMPLE: explaining prediction for validation row {example_idx}")
    print(f"Date: {val_df.loc[example_idx, 'Date']}")
    print(f"Actual direction: {'UP' if val_df.loc[example_idx, 'target_direction_5d'] == 1 else 'DOWN'}")
    print("=" * 60)
    print(local_explanation.to_string(index=False))


if __name__ == "__main__":
    main()