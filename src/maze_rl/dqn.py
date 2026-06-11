"""Shared DQN network and device helpers."""

from typing import Optional

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Simple feed-forward Q-network for MazeEnv continuous state."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve training device."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")
