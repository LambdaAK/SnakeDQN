"""Convolutional actor-critic over the egocentric grid observation."""

from __future__ import annotations

import torch
from torch import nn

from snake_rl.env.observations import NUM_CHANNELS


class PolicyNet(nn.Module):
    """obs float32 (B, C, H, W) -> (action logits (B, 4), value (B,)).

    A constant plane of ones is appended before the convolutions; with zero padding
    it tells the network where the walls are."""

    def __init__(self, board_size: int = 11, channels: int = NUM_CHANNELS, hidden: int = 256):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.hidden = hidden
        self.conv = nn.Sequential(
            nn.Conv2d(channels + 1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * board_size * board_size, hidden), nn.ReLU()
        )
        self.policy = nn.Linear(hidden, 4)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        ones = torch.ones_like(obs[:, :1])
        h = self.trunk(self.conv(torch.cat([obs, ones], dim=1)))
        return self.policy(h), self.value(h).squeeze(-1)

    def config(self) -> dict:
        return {"board_size": self.board_size, "channels": self.channels, "hidden": self.hidden}


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Forbid moves the mask rules out; rows with no allowed move keep all moves."""
    mask = mask | ~mask.any(dim=1, keepdim=True)
    return logits.masked_fill(~mask, -1e9)


def load_policy(path, device: str = "cpu") -> PolicyNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = PolicyNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
