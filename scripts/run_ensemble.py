"""
Builds a weighted-average ensemble of Random Forest, XGBoost, and
LSTM, using validation-set ROC-AUC to determine weights, then
evaluates the ensemble on the held-out test set.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.dl.lstm_model import StockLSTM  # noqa: E402
from src.dl.sequence_dataset import build_sequences, fit_scaler_on_train, scale_features  # noqa: E402
from src.ensemble.ensemble_model import compute_ensemble_weights, ensemble_predict_proba  # noqa: E402
from src.ml.baseline_models import get_feature_columns, train_random_forest  # noqa: E402
from src.ml.splitting import time_aware_split  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SEQUENCE_LENGTH = 15


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv(PROCESSED_DIR / "AAPL_labeled.csv", parse_dates=["Date"])
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target_direction_5d"]).reset_index(drop=True)

    train_df, val_df, test_df = time_aware_split(df)
    X_train, y_train = train_df[feature_cols], train_df["target_direction_5d"]
    X_val, y_val = val_df[feature_cols], val_df["target_direction_5d"]
    X_test, y_test = test_df[feature_cols], test_df["target_direction_5d"]

    # --- Random Forest ---
    rf = train_random_forest(X_train, y_train)
    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    rf_test_proba = rf.predict_proba(X_test)[:, 1]

    # --- XGBoost ---
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_test_proba = xgb.predict_proba(X_test)[:, 1]

    # --- LSTM ---
    scaler = fit_scaler_on_train(train_df, feature_cols)
    train_scaled = scale_features(train_df, feature_cols, scaler)
    val_scaled = scale_features(val_df, feature_cols, scaler)
    test_scaled = scale_features(test_df, feature_cols, scaler)

    X_val_seq, y_val_seq = build_sequences(val_scaled, feature_cols, "target_direction_5d", SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = build_sequences(test_scaled, feature_cols, "target_direction_5d", SEQUENCE_LENGTH)

    lstm_model_path = Path(__file__).resolve().parents[1] / "models" / "dl" / "lstm_direction_model.pt"
    lstm = StockLSTM(n_features=len(feature_cols), hidden_size=16, num_layers=1, dropout=0.4)
    lstm.load_state_dict(torch.load(lstm_model_path))
    lstm.eval()

    with torch.no_grad():
        lstm_val_proba = torch.sigmoid(lstm(torch.tensor(X_val_seq, dtype=torch.float32))).numpy()
        lstm_test_proba = torch.sigmoid(lstm(torch.tensor(X_test_seq, dtype=torch.float32))).numpy()

    # NOTE: LSTM sequences are shorter (SEQUENCE_LENGTH-1 fewer rows) than
    # RF/XGBoost's row-per-day predictions, since building a sequence
    # consumes the first (SEQUENCE_LENGTH-1) rows as warmup. We align by
    # trimming RF/XGBoost's val/test predictions to match the LSTM's
    # sequence-aligned rows (the last len(X_val_seq) rows).
    rf_val_proba_aligned = rf_val_proba[-len(lstm_val_proba):]
    xgb_val_proba_aligned = xgb_val_proba[-len(lstm_val_proba):]
    y_val_aligned = y_val.values[-len(lstm_val_proba):]

    rf_test_proba_aligned = rf_test_proba[-len(lstm_test_proba):]
    xgb_test_proba_aligned = xgb_test_proba[-len(lstm_test_proba):]
    y_test_aligned = y_test.values[-len(lstm_test_proba):]

    # --- Compute validation ROC-AUC per model (for weighting) ---
    val_roc_aucs = {
        "random_forest": roc_auc_score(y_val_aligned, rf_val_proba_aligned),
        "xgboost": roc_auc_score(y_val_aligned, xgb_val_proba_aligned),
        "lstm": roc_auc_score(y_val_aligned, lstm_val_proba),
    }
    print("Validation ROC-AUC per model:", {k: round(v, 4) for k, v in val_roc_aucs.items()})

    weights = compute_ensemble_weights(val_roc_aucs)
    print("Ensemble weights (from validation performance):", {k: round(v, 4) for k, v in weights.items()})

    # --- Evaluate ensemble on TEST set (final, unseen) ---
    test_predictions = {
        "random_forest": rf_test_proba_aligned,
        "xgboost": xgb_test_proba_aligned,
        "lstm": lstm_test_proba,
    }
    ensemble_test_proba = ensemble_predict_proba(test_predictions, weights)
    ensemble_test_preds = (ensemble_test_proba >= 0.5).astype(int)

    ensemble_accuracy = accuracy_score(y_test_aligned, ensemble_test_preds)
    ensemble_roc_auc = roc_auc_score(y_test_aligned, ensemble_test_proba)

    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS (Test Set - Final Unseen Period)")
    print("=" * 60)
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"Ensemble ROC-AUC:  {ensemble_roc_auc:.4f}")

    print("\nIndividual model test-set performance (for comparison):")
    for name, proba in test_predictions.items():
        preds = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_test_aligned, preds)
        auc = roc_auc_score(y_test_aligned, proba)
        print(f"  {name:15s}: accuracy={acc:.4f}, roc_auc={auc:.4f}")


if __name__ == "__main__":
    main()