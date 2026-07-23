"""
Weighted-average ensemble combining Random Forest, XGBoost, and LSTM
predictions.

Weights are computed from VALIDATION set performance only (never the
test set), per the project's rule against optimizing on held-out
test data. A model's weight is proportional to its validation ROC-AUC
above 0.5 (random-chance baseline) - models that perform no better
than random contribute ~zero weight, rather than being manually
excluded.
"""

import numpy as np


def compute_ensemble_weights(model_roc_aucs: dict) -> dict:
    """
    Convert per-model validation ROC-AUC scores into normalized
    ensemble weights.

    Args:
        model_roc_aucs: e.g. {"random_forest": 0.631, "xgboost": 0.524,
            "lstm": 0.646}

    Returns:
        Normalized weights summing to 1.0, e.g.
        {"random_forest": 0.43, "xgboost": 0.12, "lstm": 0.45}
    """
    # Skill above random chance (0.5) - a model at exactly 0.5 gets 0 weight
    skill_scores = {name: max(auc - 0.5, 0.0) for name, auc in model_roc_aucs.items()}
    total_skill = sum(skill_scores.values())

    if total_skill == 0:
        # Fallback: equal weights if no model beats random chance
        n = len(model_roc_aucs)
        return {name: 1.0 / n for name in model_roc_aucs}

    return {name: score / total_skill for name, score in skill_scores.items()}


def ensemble_predict_proba(predictions: dict, weights: dict) -> np.ndarray:
    """
    Combine multiple models' predicted probabilities into one
    weighted-average probability.

    Args:
        predictions: e.g. {"random_forest": array([...]), "xgboost": array([...]), "lstm": array([...])}
            Each array is that model's predicted probability of "up"
            for the same set of rows, in the same order.
        weights: e.g. {"random_forest": 0.43, "xgboost": 0.12, "lstm": 0.45}

    Returns:
        Combined probability array.
    """
    combined = np.zeros_like(next(iter(predictions.values())), dtype=float)
    for name, proba in predictions.items():
        combined += weights[name] * proba
    return combined