"""Permutation-invariant pore-cloud to stress-strain response validator."""

from __future__ import annotations

import torch
from torch import nn

from src.models.poregen_cvae import MaskedSetEncoder


class GeoLoopStressNet(nn.Module):
    """Predict the complete normalized stress response from a pore set."""

    def __init__(
        self, pore_feature_dim: int = 10, macro_dim: int = 5,
        set_hidden: int = 192, hidden: int = 256, curve_points: int = 50,
    ):
        super().__init__()
        self.set_encoder = MaskedSetEncoder(pore_feature_dim, set_hidden)
        self.macro_encoder = nn.Sequential(
            nn.Linear(macro_dim, 64), nn.LayerNorm(64), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(2 * set_hidden + 64, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, curve_points), nn.Sigmoid(),
        )

    def forward(
        self, pores: torch.Tensor, mask: torch.Tensor, macro: torch.Tensor
    ) -> torch.Tensor:
        set_features = self.set_encoder(pores, mask)
        macro_features = self.macro_encoder(macro)
        return self.head(torch.cat((set_features, macro_features), dim=-1))
