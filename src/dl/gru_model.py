"""
GRU model for binary direction classification.

Architecturally simpler than LSTM (no separate cell state), which is
directly relevant here: our LSTM experiment showed that reducing model
capacity was necessary to avoid overfitting on our small dataset. GRU
has inherently fewer parameters per layer than LSTM, making it a
natural, well-motivated comparison.
"""

import torch
import torch.nn as nn


class StockGRU(nn.Module):
    """A GRU classifier, mirroring StockLSTM's structure for a fair comparison."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.gru = nn.GRU(
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
        gru_out, h_n = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        logits = self.head(last_hidden).squeeze(-1)
        return logits