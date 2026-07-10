"""
Sequence dataset construction for LSTM/GRU training.

Converts row-per-day tabular data into overlapping sequences of
`sequence_length` consecutive days, where each sequence's label is
the direction target of its LAST day.

CRITICAL LEAKAGE NOTE:
    A sequence ending at day T uses feature rows [T-seq_len+1, ..., T]
    - all at or before T. The label for that sequence is
    target_direction_5d at day T, which itself looks forward to T+5
    (by design, per target_engineering.py). This is consistent with
    our tabular models: the "prediction moment" is day T, using only
    information through T, predicting an outcome that resolves at T+5.

SCALING NOTE:
    The scaler must be fit ONLY on training sequences' underlying rows,
    never validation/test - same principle as Logistic Regression,
    but neural networks are even more sensitive to unscaled inputs.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class StockSequenceDataset(Dataset):
    """PyTorch Dataset yielding (sequence, label) pairs."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a chronologically-sorted DataFrame into overlapping
    sequences.

    Args:
        df: DataFrame sorted by date, with feature and target columns,
            no NaNs in feature_cols or target_col.
        feature_cols: Columns to use as input features per timestep.
        target_col: Column to use as the label (from the sequence's
            LAST row).
        sequence_length: Number of consecutive days per sequence.

    Returns:
        (sequences, labels):
            sequences shape = (n_sequences, sequence_length, n_features)
            labels shape = (n_sequences,)
    """
    feature_array = df[feature_cols].values
    label_array = df[target_col].values

    sequences = []
    labels = []

    for i in range(sequence_length - 1, len(df)):
        seq = feature_array[i - sequence_length + 1: i + 1]
        sequences.append(seq)
        labels.append(label_array[i])

    return np.array(sequences), np.array(labels)


def fit_scaler_on_train(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Fit a StandardScaler on training rows only."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def scale_features(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    """Apply an already-fitted scaler to a DataFrame's feature columns."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df