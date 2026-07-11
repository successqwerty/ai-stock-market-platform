"""
SHAP-based explainability for tree-based models (Random Forest, XGBoost).

Provides:
    - Global feature importance (which features matter most, overall)
    - Local explanation for a single prediction (why THIS prediction
      was made)
"""

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values for a tree-based model's predictions.

    Uses TreeExplainer, which is fast and exact for tree-based models
    (Random Forest, XGBoost) - not an approximation like model-agnostic
    SHAP methods.

    Returns:
        shap.Explanation object containing per-feature contributions
        for every row in X.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def get_global_feature_importance(shap_values, feature_names: list[str]) -> pd.DataFrame:
    """
    Rank features by mean absolute SHAP value across all rows - i.e.
    which features have the largest average impact on predictions,
    regardless of direction.
    """
    # For binary classification, shap_values.values may have shape
    # (n_samples, n_features, 2) - one set per class. We use class 1
    # (the "up" class) since that's what we care about explaining.
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs_shap = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance_df


def explain_single_prediction(
    shap_values,
    feature_names: list[str],
    row_index: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Explain one specific prediction: which features pushed it toward
    BUY (positive SHAP) vs SELL (negative SHAP), ranked by magnitude.
    """
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    row_values = values[row_index]

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": row_values,
    })
    explanation_df["direction"] = np.where(
        explanation_df["shap_value"] > 0, "pushes toward UP", "pushes toward DOWN"
    )
    explanation_df["abs_shap"] = explanation_df["shap_value"].abs()

    return explanation_df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")