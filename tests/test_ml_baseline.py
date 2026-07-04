"""Tests for src/ml/splitting.py and src/ml/baseline_models.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ml.baseline_models import get_feature_columns, naive_baseline_predict
from src.ml.splitting import time_aware_split


def _sample_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "Ticker": "TEST",
        "Open": np.arange(n, dtype=float),
        "High": np.arange(n, dtype=float),
        "Low": np.arange(n, dtype=float),
        "Close": np.arange(n, dtype=float),
        "Adj Close": np.arange(n, dtype=float),
        "Volume": np.arange(n, dtype=float),
        "some_feature": np.random.default_rng(1).normal(size=n),
        "target_return_5d": np.random.default_rng(2).normal(size=n),
        "target_direction_5d": np.random.default_rng(3).integers(0, 2, size=n),
        "target_class_5d": np.random.default_rng(4).integers(-1, 2, size=n),
    })


def test_time_aware_split_preserves_chronological_order():
    df = _sample_df(100)
    train, val, test = time_aware_split(df, train_frac=0.7, val_frac=0.15)

    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()


def test_time_aware_split_sizes_approximately_correct():
    df = _sample_df(100)
    train, val, test = time_aware_split(df, train_frac=0.7, val_frac=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_time_aware_split_rejects_unsorted_data():
    df = _sample_df(50)
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

    with pytest.raises(AssertionError):
        time_aware_split(shuffled)


def test_get_feature_columns_excludes_targets_and_raw_price():
    df = _sample_df(20)
    features = get_feature_columns(df)

    assert "target_direction_5d" not in features
    assert "target_return_5d" not in features
    assert "target_class_5d" not in features
    assert "Adj Close" not in features
    assert "Date" not in features
    assert "some_feature" in features


def test_naive_baseline_predicts_majority_class_only():
    df = _sample_df(100)
    # Force a clear majority: 90 rows of class 1, 10 rows of class 0
    df["target_direction_5d"] = [1] * 90 + [0] * 10

    preds = naive_baseline_predict(df, n_predictions=20)
    assert (preds == 1).all()
    assert len(preds) == 20