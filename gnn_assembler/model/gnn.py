"""Message-passing network that scores the next part to remove."""

from __future__ import annotations

import torch
from torch import nn

from gnn_assembler.geometry.graph import FEATURE_DIM


def neighbor_mean(features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=-1, keepdim=True)
    pooled = adjacency @ features
    mean = torch.zeros_like(features)
    mask = degree.squeeze(-1) > 0
    mean[mask] = pooled[mask] / degree[mask]
    return mean


class RemovalGNN(nn.Module):
    """Two rounds of neighbor-mean message passing, then a part score."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.enc = nn.Linear(in_dim, hidden)
        self.mp1 = nn.Linear(hidden * 2, hidden)
        self.mp2 = nn.Linear(hidden * 2, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.enc(features))
        hidden = torch.relu(self.mp1(torch.cat([hidden, neighbor_mean(hidden, adjacency)], dim=-1)))
        hidden = torch.relu(self.mp2(torch.cat([hidden, neighbor_mean(hidden, adjacency)], dim=-1)))
        return self.out(hidden).squeeze(-1)


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
