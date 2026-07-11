"""Tests for src/explainability/shap_explainer.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import RandomForestClassifier

from src.explainability.shap_explainer import (
    compute_shap_values,
    explain_single_prediction,
    get_global_feature_importance,
)


def _sample_model_and_data():
    """Small synthetic dataset + trained RF for fast SHAP tests."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "feature_a": rng.normal(size=100),
        "feature_b": rng.normal(size=100),
        "feature_c": rng.normal(size=100),
    })
    # Make feature_a clearly predictive, others noise
    y = (X["feature_a"] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, X


def test_global_importance_ranks_predictive_feature_highest():
    """feature_a was constructed to be the only real predictor -
    SHAP should rank it as most important."""
    model, X = _sample_model_and_data()
    shap_values = compute_shap_values(model, X)

    importance_df = get_global_feature_importance(shap_values, list(X.columns))

    assert importance_df.iloc[0]["feature"] == "feature_a"


def test_local_explanation_returns_requested_top_n():
    model, X = _sample_model_and_data()
    shap_values = compute_shap_values(model, X)

    explanation = explain_single_prediction(shap_values, list(X.columns), row_index=0, top_n=2)

    assert len(explanation) == 2
    assert "direction" in explanation.columns


def test_local_explanation_direction_matches_shap_sign():
    model, X = _sample_model_and_data()
    shap_values = compute_shap_values(model, X)

    explanation = explain_single_prediction(shap_values, list(X.columns), row_index=0, top_n=3)

    for _, row in explanation.iterrows():
        if row["shap_value"] > 0:
            assert row["direction"] == "pushes toward UP"
        else:
            assert row["direction"] == "pushes toward DOWN"