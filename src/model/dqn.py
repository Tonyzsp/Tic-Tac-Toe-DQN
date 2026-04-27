from __future__ import annotations

import torch
from torch import nn


class DQNNet(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def legal_mask_from_state(state: torch.Tensor) -> torch.Tensor:
    # state shape: [B, 9], empty cells are exactly 0
    return state.eq(0.0)


def apply_invalid_mask(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return q_values.masked_fill(~legal_mask, float("-inf"))


def masked_argmax(q_values: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked_q = apply_invalid_mask(q_values, legal_mask)
    return torch.argmax(masked_q, dim=-1)


def masked_softmax(q_values: torch.Tensor, legal_mask: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    scaled = q_values / max(tau, 1e-6)
    masked = apply_invalid_mask(scaled, legal_mask)
    return torch.softmax(masked, dim=-1)
