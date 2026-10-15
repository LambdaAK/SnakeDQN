"""Wrap a PolicyNet as a harness policy: (state, snake) -> actions."""

from __future__ import annotations

import numpy as np
import torch

from snake_rl.agents.model import PolicyNet, masked_logits
from snake_rl.env.observations import build_observations, build_window_observations
from snake_rl.env.rules import GameState, action_mask


class NeuralPolicy:
    def __init__(self, model: PolicyNet, name: str = "ppo", deterministic: bool = True,
                 seed: int | None = None):
        self.model = model
        self.name = name
        self.deterministic = deterministic
        self.generator = torch.Generator().manual_seed(0 if seed is None else seed)
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def __call__(self, state: GameState, snake: int) -> np.ndarray:
        if self.model.window:
            obs = build_window_observations(state, self.model.window)[:, snake]
        else:
            obs = build_observations(state)[:, snake]
        obs = torch.from_numpy(obs).to(self.device)
        mask = torch.from_numpy(action_mask(state)[:, snake]).to(self.device)
        logits, _ = self.model(obs)
        logits = masked_logits(logits, mask).cpu()
        if self.deterministic:
            return logits.argmax(dim=1).numpy()
        probs = torch.softmax(logits, dim=1)
        return torch.multinomial(probs, 1, generator=self.generator).squeeze(1).numpy()
