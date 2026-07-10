"""Tests for src/dl/sequence_dataset.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dl.sequence_dataset import build_sequences, fit_scaler_on_train, scale_features


def _sample_df(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_a": np.arange(n, dtype=float),
        "feature_b": np.arange(n, dtype=float) * 2,
        "target_direction_5d": np.random.default_rng(1).integers(0, 2, size=n),
    })


def test_build_sequences_shape():
    df = _sample_df(50)
    X, y = build_sequences(df, ["feature_a", "feature_b"], "target_direction_5d", sequence_length=10)

    assert X.shape == (41, 10, 2)  # 50 - 10 + 1 = 41 sequences
    assert y.shape == (41,)


def test_build_sequences_last_row_matches_label():
    """The label for sequence i should be the target at the sequence's LAST row."""
    df = _sample_df(20)
    X, y = build_sequences(df, ["feature_a"], "target_direction_5d", sequence_length=5)

    # First sequence covers rows 0-4, label should be target at row 4
    assert y[0] == df.loc[4, "target_direction_5d"]
    # Sequence's last feature value should match the row's feature_a
    assert X[0, -1, 0] == df.loc[4, "feature_a"]


def test_scaler_fit_on_train_only():
    """Scaler statistics must come only from training data, not val/test."""
    train_df = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    val_df = pd.DataFrame({"f": [100.0, 200.0]})  # very different scale

    scaler = fit_scaler_on_train(train_df, ["f"])

    # Scaler's mean should reflect ONLY train_df's values (mean=2.0), not val_df
    assert np.isclose(scaler.mean_[0], 2.0)

    # Applying it to val_df should NOT refit - val values should be
    # scaled using train's statistics, producing large scaled values
    # (since val is far outside train's range)
    scaled_val = scale_features(val_df, ["f"], scaler)
    assert scaled_val["f"].iloc[0] > 10  # far from 0 because train's scale is tiny