"""
Backtesting engine for evaluating trading signals.

Models the full chain:
    Signal -> Position -> Trade Execution -> Transaction Cost -> Portfolio Value

Key assumptions (documented explicitly, per project rules):
    - Entry: at the CLOSE of the day the signal is generated (T).
    - Exit: at the CLOSE of day T + horizon (matching how the target
      was defined - see src/targets/target_engineering.py).
    - Position sizing: fully invested (100% of capital) when BUY,
      fully in cash when SELL/HOLD - no leverage, no shorting, for
      Milestone 1 simplicity.
    - Transaction costs: applied as a fixed percentage per trade
      (entry AND exit each incur the cost).
    - No look-ahead: signal at row T only uses probability computed
      from information available through T's close.
"""

import numpy as np
import pandas as pd


def run_backtest(
    dates: pd.Series,
    signals: pd.Series,
    forward_returns: pd.Series,
    transaction_cost_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Run a simple backtest: BUY signals take a full position for the
    holding period (matching the target's horizon), earning the
    forward return minus transaction costs. SELL/HOLD stay in cash
    (0% return, 0 cost).

    Args:
        dates: Date for each row (signal generation date).
        signals: "BUY"/"HOLD"/"SELL" per row.
        forward_returns: The ACTUAL realized forward return over the
            horizon for each row (i.e. target_return_Nd - ground
            truth, used here only for evaluation, never for generating
            the signal itself).
        transaction_cost_pct: Cost per trade, applied twice for a
            BUY (once to enter, once to exit), e.g. 0.001 = 10 bps.

    Returns:
        DataFrame with columns: Date, signal, strategy_return,
        benchmark_return (unmodified forward_returns for comparison),
        cumulative_strategy, cumulative_benchmark.
    """
    df = pd.DataFrame({
        "Date": dates.values,
        "signal": signals.values,
        "forward_return": forward_returns.values,
    })

    is_buy = df["signal"] == "BUY"

    # Strategy return: forward return if BUY, else 0 (cash, no return)
    df["strategy_return"] = np.where(is_buy, df["forward_return"], 0.0)

    # Transaction cost: charged only when we actually enter a BUY trade
    # (entry + exit combined into one round-trip cost)
    df["transaction_cost"] = np.where(is_buy, transaction_cost_pct * 2, 0.0)
    df["strategy_return_net"] = df["strategy_return"] - df["transaction_cost"]

    df["cumulative_strategy"] = (1 + df["strategy_return_net"]).cumprod()
    df["cumulative_benchmark"] = (1 + df["forward_return"]).cumprod()

    return df


def compute_performance_metrics(
    backtest_df: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute standard strategy performance metrics.

    Note: this function assumes `backtest_df` rows are NON-OVERLAPPING
    periods (see run_milestone1_backtest.py, which resamples to every
    Nth row before calling this). If rows overlap, cumulative return
    and drawdown become meaningless because the same calendar days
    get compounded multiple times.
    
    """
    strategy_returns = backtest_df["strategy_return_net"]
    benchmark_returns = backtest_df["forward_return"]

    n_trades = int((backtest_df["signal"] == "BUY").sum())

    cumulative_return = backtest_df["cumulative_strategy"].iloc[-1] - 1
    benchmark_cumulative_return = backtest_df["cumulative_benchmark"].iloc[-1] - 1

    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else 0.0

    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (mean_return / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    running_max = backtest_df["cumulative_strategy"].cummax()
    drawdown = (backtest_df["cumulative_strategy"] - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = (strategy_returns[backtest_df["signal"] == "BUY"] > 0).mean() if n_trades > 0 else float("nan")

    return {
        "cumulative_return": cumulative_return,
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "number_of_trades": n_trades,
        "exposure_pct": n_trades / len(backtest_df) if len(backtest_df) > 0 else 0.0,
    }