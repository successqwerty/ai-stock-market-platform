"""
Baseline classification models for direction prediction:
naive baseline, Logistic Regression, Random Forest, XGBoost.

All models predict the BINARY DIRECTION target (target_direction_5d).
Feature columns exclude anything with "target" in the name, and
exclude raw identifiers (Date, Ticker) and raw price levels (which
are non-stationary and would leak trend information trivially).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Columns that must NEVER be used as model inputs
EXCLUDE_COLS = {
    "Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "target_return_5d", "target_direction_5d", "target_class_5d",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns safe to use as model features."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def naive_baseline_predict(train_df: pd.DataFrame, n_predictions: int) -> np.ndarray:
    """
    Naive baseline: always predict the majority class observed in
    training data. This is the bar every real model must beat.
    """
    majority_class = train_df["target_direction_5d"].mode()[0]
    return np.full(n_predictions, majority_class)


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train Logistic Regression with feature scaling.

    Unlike tree-based models, Logistic Regression is sensitive to
    feature scale - our features span wildly different ranges (e.g.
    OBV in the billions vs RSI in 0-100 vs daily_return around 0.01).
    Without scaling, the optimizer effectively can't converge properly.

    The scaler is fit ONLY on training data (never validation/test),
    consistent with the rule: never fit transformers on data beyond
    the training set.

    Returns:
        (scaler, model) tuple - the scaler must be reused to transform
        validation/test data before prediction.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    return scaler, model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics