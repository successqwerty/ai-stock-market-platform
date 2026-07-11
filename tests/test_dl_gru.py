"""Tests for src/dl/gru_model.py - structural tests mirroring the LSTM tests."""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dl.gru_model import StockGRU


def test_gru_output_shape():
    """Model should output one logit per sequence in the batch."""
    model = StockGRU(n_features=27, hidden_size=16, num_layers=1)
    batch = torch.randn(8, 15, 27)  # batch_size=8, seq_len=15, features=27

    output = model(batch)
    assert output.shape == (8,)


def test_gru_has_no_sigmoid_layer():
    """
    Model must output raw logits for use with BCEWithLogitsLoss, not
    pre-activated probabilities. Check structurally (no Sigmoid module
    anywhere in the model) rather than checking random output values,
    which can coincidentally fall in [0, 1] purely by chance with
    small, untrained weights.
    """
    model = StockGRU(n_features=27, hidden_size=16, num_layers=1)
    has_sigmoid = any(isinstance(module, torch.nn.Sigmoid) for module in model.modules())
    assert not has_sigmoid   