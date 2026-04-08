"""GRU sequence decoder with heteroscedastic output heads."""

from __future__ import annotations

import torch
from torch import nn


class GRUDecoder(nn.Module):
    """Stacked GRU decoder with the same interface as :class:`LSTMDecoder`."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            msg = "num_layers must be >= 1"
            raise ValueError(msg)

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.mean_head = nn.Linear(hidden_size, 2)
        self.log_var_head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and log-variance predictions for each timestep."""
        features, _ = self.rnn(x)
        features = self.output_dropout(features)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        return mean, log_var
