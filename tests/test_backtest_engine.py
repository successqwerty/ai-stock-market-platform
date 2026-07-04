"""Tests for src/backtesting/signal_engine.py and backtest_engine.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtesting.backtest_engine import compute_performance_metrics, run_backtest
from src.backtesting.signal_engine import generate_signals


def test_generate_signals_thresholds():
    probs = np.array([0.9, 0.5, 0.1, 0.56, 0.44])
    signals = generate_signals(probs, buy_threshold=0.55, sell_threshold=0.45)
    assert list(signals) == ["BUY", "HOLD", "SELL", "BUY", "SELL"]


def test_run_backtest_hold_and_sell_earn_zero_return():
    dates = pd.Series(pd.date_range("2024-01-01", periods=3, freq="B"))
    signals = pd.Series(["HOLD", "SELL", "BUY"])
    forward_returns = pd.Series([0.05, 0.05, 0.05])

    result = run_backtest(dates, signals, forward_returns, transaction_cost_pct=0.0)

    assert result.loc[0, "strategy_return"] == 0.0
    assert result.loc[1, "strategy_return"] == 0.0
    assert result.loc[2, "strategy_return"] == 0.05


def test_run_backtest_applies_transaction_cost_only_on_buy():
    dates = pd.Series(pd.date_range("2024-01-01", periods=2, freq="B"))
    signals = pd.Series(["HOLD", "BUY"])
    forward_returns = pd.Series([0.05, 0.05])

    result = run_backtest(dates, signals, forward_returns, transaction_cost_pct=0.01)

    assert result.loc[0, "transaction_cost"] == 0.0
    assert result.loc[1, "transaction_cost"] == 0.02  # 0.01 * 2 (entry + exit)
    assert np.isclose(result.loc[1, "strategy_return_net"], 0.05 - 0.02)


def test_compute_performance_metrics_no_trades_gives_zero_return():
    dates = pd.Series(pd.date_range("2024-01-01", periods=3, freq="B"))
    signals = pd.Series(["HOLD", "HOLD", "SELL"])
    forward_returns = pd.Series([0.02, -0.01, 0.03])

    backtest_df = run_backtest(dates, signals, forward_returns)
    metrics = compute_performance_metrics(backtest_df)

    assert metrics["number_of_trades"] == 0
    assert np.isclose(metrics["cumulative_return"], 0.0)


def test_compute_performance_metrics_matches_manual_calculation():
    dates = pd.Series(pd.date_range("2024-01-01", periods=2, freq="B"))
    signals = pd.Series(["BUY", "BUY"])
    forward_returns = pd.Series([0.10, -0.05])

    backtest_df = run_backtest(dates, signals, forward_returns, transaction_cost_pct=0.0)
    metrics = compute_performance_metrics(backtest_df)

    # (1.10 * 0.95) - 1 = 0.045
    expected_cumulative = (1.10 * 0.95) - 1
    assert np.isclose(metrics["cumulative_return"], expected_cumulative)