"""
Data validation for raw OHLCV market data.

This module only *checks* data quality and reports issues — it never
modifies data. Cleaning happens separately in cleaning.py.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass
class ValidationReport:
    """Collects all validation issues found in a dataset."""

    ticker: str
    total_rows: int
    issues: dict[str, int] = field(default_factory=dict)
    duplicate_dates: list = field(default_factory=list)
    missing_dates: list = field(default_factory=list)
    invalid_ohlc_rows: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return sum(self.issues.values()) == 0

    def summary(self) -> str:
        lines = [f"Validation report for {self.ticker} ({self.total_rows} rows):"]
        if self.is_valid:
            lines.append("  No issues found.")
        else:
            for issue, count in self.issues.items():
                if count > 0:
                    lines.append(f"  {issue}: {count}")
        return "\n".join(lines)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Check that all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> ValidationReport:
    """
    Run all validation checks on a single ticker's OHLCV data.

    Args:
        df: DataFrame with columns Date, Ticker, Open, High, Low, Close,
            Adj Close, Volume. Assumed already schema-validated.
        ticker: Ticker symbol, used for reporting.

    Returns:
        ValidationReport summarizing all issues found.
    """
    report = ValidationReport(ticker=ticker, total_rows=len(df))

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Missing values
    report.issues["missing_values"] = int(df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum())

    # Duplicate rows (exact duplicates)
    report.issues["duplicate_rows"] = int(df.duplicated().sum())

    # Duplicate timestamps (same date appearing more than once)
    dup_dates_mask = df["Date"].duplicated(keep=False)
    report.duplicate_dates = df.loc[dup_dates_mask, "Date"].dt.strftime("%Y-%m-%d").tolist()
    report.issues["duplicate_dates"] = int(df["Date"].duplicated().sum())

    # Chronological ordering
    is_sorted = df["Date"].is_monotonic_increasing
    report.issues["out_of_order"] = 0 if is_sorted else 1

    # Negative or zero prices
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    negative_prices = (df[price_cols] <= 0).sum().sum()
    report.issues["non_positive_prices"] = int(negative_prices)

    # Negative volume
    report.issues["negative_volume"] = int((df["Volume"] < 0).sum())

    # Invalid OHLC relationships:
    # High must be >= Open, Close, Low
    # Low must be <= Open, Close, High
    invalid_mask = (
        (df["High"] < df["Low"])
        | (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    )
    report.invalid_ohlc_rows = df.loc[invalid_mask, "Date"].dt.strftime("%Y-%m-%d").tolist()
    report.issues["invalid_ohlc_relationships"] = int(invalid_mask.sum())

    # Missing trading dates: compare against a business-day calendar.
    # Note: this is approximate (doesn't account for market holidays),
    # flagged as informational rather than a hard failure.
    full_range = pd.bdate_range(df["Date"].min(), df["Date"].max())
    actual_dates = set(df["Date"])
    missing = sorted(set(full_range) - actual_dates)
    report.missing_dates = [d.strftime("%Y-%m-%d") for d in missing]
    report.issues["missing_business_days"] = len(missing)

    logger.info(report.summary())
    return report