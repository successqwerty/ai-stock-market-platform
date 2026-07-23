"""Tests for src/ensemble/ensemble_model.py"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ensemble.ensemble_model import compute_ensemble_weights, ensemble_predict_proba


def test_compute_ensemble_weights_normal():
    aucs = {"random_forest": 0.6, "xgboost": 0.7, "lstm": 0.8}
    # Skill scores above 0.5: RF=0.1, XGB=0.2, LSTM=0.3 -> Total = 0.6
    # Weights: RF=1/6, XGB=2/6, LSTM=3/6
    weights = compute_ensemble_weights(aucs)

    assert pytest.approx(weights["random_forest"], 1e-4) == 1.0 / 6.0
    assert pytest.approx(weights["xgboost"], 1e-4) == 2.0 / 6.0
    assert pytest.approx(weights["lstm"], 1e-4) == 3.0 / 6.0
    assert pytest.approx(sum(weights.values()), 1e-6) == 1.0


def test_compute_ensemble_weights_below_random_chance():
    aucs = {"random_forest": 0.45, "xgboost": 0.48, "lstm": 0.47}
    # All below 0.5, total skill = 0 -> Equal weights fallback
    weights = compute_ensemble_weights(aucs)

    assert pytest.approx(weights["random_forest"], 1e-4) == 1.0 / 3.0
    assert pytest.approx(weights["xgboost"], 1e-4) == 1.0 / 3.0
    assert pytest.approx(weights["lstm"], 1e-4) == 1.0 / 3.0
    assert pytest.approx(sum(weights.values()), 1e-6) == 1.0


def test_compute_ensemble_weights_partial_below_random():
    aucs = {"model_a": 0.4, "model_b": 0.6}
    # model_a skill = 0, model_b skill = 0.1 -> total skill = 0.1
    weights = compute_ensemble_weights(aucs)

    assert weights["model_a"] == 0.0
    assert weights["model_b"] == 1.0


def test_ensemble_predict_proba():
    predictions = {
        "model_a": np.array([0.2, 0.8, 0.5]),
        "model_b": np.array([0.4, 0.6, 0.9]),
    }
    weights = {"model_a": 0.25, "model_b": 0.75}

    combined = ensemble_predict_proba(predictions, weights)
    expected = 0.25 * np.array([0.2, 0.8, 0.5]) + 0.75 * np.array([0.4, 0.6, 0.9])

    np.testing.assert_allclose(combined, expected)
