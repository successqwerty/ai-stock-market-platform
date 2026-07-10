"""
LSTM model for binary direction classification.
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    A simple LSTM classifier: LSTM layer(s) -> take last timestep's
    hidden state -> small feedforward head -> single logit (binary
    classification via BCEWithLogitsLoss).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, sequence_length, n_features)
        Returns:
            logits: shape (batch,) - raw scores, apply sigmoid for
                probability (done outside the model, paired with
                BCEWithLogitsLoss for numerical stability)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take the last timestep's output (most recent day's hidden state)
        last_hidden = lstm_out[:, -1, :]
        logits = self.head(last_hidden).squeeze(-1)
        return logits  