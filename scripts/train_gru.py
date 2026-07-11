"""
Phase 10: train a GRU on sequences of engineered features to predict
binary direction, with early stopping, and compare against Milestone 1
baselines and the tuned LSTM (Phase 9).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.dl.gru_model import StockGRU  # noqa: E402
from src.dl.sequence_dataset import (  # noqa: E402
    StockSequenceDataset,
    build_sequences,
    fit_scaler_on_train,
    scale_features,
)
from src.ml.baseline_models import get_feature_columns  # noqa: E402
from src.ml.splitting import time_aware_split  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "dl"

SEQUENCE_LENGTH = 15
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10  # early stopping patience
LEARNING_RATE = 0.0005


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv(PROCESSED_DIR / "AAPL_labeled.csv", parse_dates=["Date"])
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target_direction_5d"]).reset_index(drop=True)

    train_df, val_df, test_df = time_aware_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Fit scaler on TRAIN ONLY, apply to all splits
    scaler = fit_scaler_on_train(train_df, feature_cols)
    train_scaled = scale_features(train_df, feature_cols, scaler)
    val_scaled = scale_features(val_df, feature_cols, scaler)
    test_scaled = scale_features(test_df, feature_cols, scaler)

    X_train, y_train = build_sequences(train_scaled, feature_cols, "target_direction_5d", SEQUENCE_LENGTH)
    X_val, y_val = build_sequences(val_scaled, feature_cols, "target_direction_5d", SEQUENCE_LENGTH)
    X_test, y_test = build_sequences(test_scaled, feature_cols, "target_direction_5d", SEQUENCE_LENGTH)

    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}, Test sequences: {X_test.shape}")

    train_loader = DataLoader(StockSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockSequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = StockGRU(n_features=len(feature_cols), hidden_size=16, num_layers=1, dropout=0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_model_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "gru_direction_model.pt")
    print(f"Saved best model to {MODELS_DIR / 'gru_direction_model.pt'}")
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32))
        val_probs = torch.sigmoid(val_logits).numpy()
        val_preds = (val_probs >= 0.5).astype(int)

    val_accuracy = accuracy_score(y_val, val_preds)
    val_roc_auc = roc_auc_score(y_val, val_probs)

    print("\n" + "=" * 60)
    print("GRU VALIDATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"ROC-AUC:  {val_roc_auc:.4f}")
    print("\nCompare to previous models (validation set):")
    print("  Naive:               accuracy=0.5366")
    print("  Logistic Regression: accuracy=0.5393, roc_auc=0.5878")
    print("  Random Forest:       accuracy=0.5908, roc_auc=0.6313")
    print("  XGBoost:              accuracy=0.5583, roc_auc=0.5244")
    print("  LSTM (tuned):        accuracy=0.6085, roc_auc=0.6460")

if __name__ == "__main__":
    main()