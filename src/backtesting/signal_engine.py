"""
Milestone 1 signal engine (simplified).

Converts a model's predicted probability of upward direction into a
BUY/HOLD/SELL signal using fixed probability thresholds.

NOTE: This is a deliberately simple version for Milestone 1. The full
signal engine (Phase 15) will incorporate expected return, transaction
costs, and confidence/uncertainty. Documenting this now so it's clear
this is a known simplification, not an oversight.

Signal rule:
    BUY  if probability_up >= buy_threshold
    SELL if probability_up <= sell_threshold
    HOLD otherwise
"""

import numpy as np
import pandas as pd


def generate_signals(
    probabilities: np.ndarray,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> pd.Series:
    """
    Convert predicted probabilities into BUY/HOLD/SELL signals.

    Args:
        probabilities: Model's predicted probability of "up" direction.
        buy_threshold: Probability at or above which we signal BUY.
        sell_threshold: Probability at or below which we signal SELL.

    Returns:
        Series of signals: "BUY", "HOLD", or "SELL".
    """
    signals = np.where(
        probabilities >= buy_threshold, "BUY",
        np.where(probabilities <= sell_threshold, "SELL", "HOLD"),
    )
    return pd.Series(signals)