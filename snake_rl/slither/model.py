"""MLP actor-critic for the slither observation (spec §3) and an evaluation wrapper."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from snake_rl.slither.actions import NUM_ACTIONS

OBS_DIM = 263


class SlitherNet(nn.Module):
    """obs float32 (B, obs_dim) -> (action logits (B, actions), value (B,))."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, actions: int = NUM_ACTIONS):
        super().__init__()
        self.obs_dim, self.hidden, self.actions = obs_dim, hidden, actions
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.policy(h), self.value(h).squeeze(-1)

    def config(self) -> dict:
        return {"obs_dim": self.obs_dim, "hidden": self.hidden, "actions": self.actions}


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Forbid actions the mask rules out; rows with no allowed action keep all of them.
    Same semantics as snake_rl.agents.model.masked_logits, kept separate so the two modes
    don't depend on each other."""
    mask = mask | ~mask.any(dim=1, keepdim=True)
    return logits.masked_fill(~mask, -1e9)


def load_slither_policy(path, device: str = "cpu") -> SlitherNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SlitherNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


class SlitherPolicy:
    """policy(obs[..., obs_dim], mask[..., actions]) -> int64 actions[...]."""

    def __init__(self, model: SlitherNet, deterministic: bool = True, seed: int = 0):
        self.model = model
        self.deterministic = deterministic
        self.generator = torch.Generator().manual_seed(seed)
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def __call__(self, obs, mask) -> np.ndarray:
        obs = np.asarray(obs, np.float32)
        mask = np.asarray(mask, bool)
        lead = obs.shape[:-1]
        o = torch.from_numpy(obs.reshape(-1, obs.shape[-1])).to(self.device)
        m = torch.from_numpy(mask.reshape(-1, mask.shape[-1])).to(self.device)
        logits, _ = self.model(o)
        logits = masked_logits(logits, m)
        if self.deterministic:
            action = logits.argmax(dim=1).cpu()
        else:
            action = torch.multinomial(torch.softmax(logits, dim=1).cpu(), 1, generator=self.generator)[:, 0]
        return action.numpy().astype(np.int64).reshape(lead)
