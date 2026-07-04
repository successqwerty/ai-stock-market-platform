"""
Time-aware train/validation/test splitting.

CRITICAL RULE: for time-series data, we NEVER randomly shuffle rows
into splits. Splits must be chronological blocks - training only on
the past, validating and testing only on periods strictly after
training ends. Randomly shuffling would leak future information into
training (the model would implicitly "see the future" via correlated
neighboring rows), producing misleadingly good but meaningless metrics.
"""

import pandas as pd


def time_aware_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a chronologically-sorted DataFrame into train/val/test blocks
    by position, not randomly.

    Args:
        df: DataFrame already sorted by date_col ascending.
        date_col: Name of the date column (used only to assert sorting).
        train_frac: Fraction of rows for training (from the start).
        val_frac: Fraction of rows for validation (immediately after train).
            The remainder goes to test.

    Returns:
        (train_df, val_df, test_df), each with a reset index.
    """
    assert df[date_col].is_monotonic_increasing, (
        "DataFrame must be sorted chronologically before splitting"
    )

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df