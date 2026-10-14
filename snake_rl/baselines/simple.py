"""Simple scripted bots. Each bot maps (state, snake index) to one action per game."""

from __future__ import annotations

import numpy as np

from snake_rl.env.rules import DIRECTIONS, GameState, action_mask


class RandomBot:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, state: GameState, snake: int) -> np.ndarray:
        return self.rng.integers(0, 4, size=state.num_envs)


class SafeRandomBot:
    """Uniform over moves that don't die immediately; any move when none are safe."""

    name = "safe_random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, state: GameState, snake: int) -> np.ndarray:
        mask = action_mask(state)[:, snake]
        return np.argmax(self.rng.random(mask.shape) + mask, axis=1)


class GreedyBot:
    """Among safe moves, step toward the nearest food (Manhattan distance)."""

    name = "greedy"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, state: GameState, snake: int) -> np.ndarray:
        n, b = state.num_envs, state.board_size
        mask = action_mask(state)[:, snake]
        head = state.body[:, snake, 0].astype(np.int64)
        nxt = head[:, None, :] + DIRECTIONS  # (N, 4, 2)
        cells = np.arange(b * b)
        dist = (np.abs(nxt[..., 0, None] - cells // b)
                + np.abs(nxt[..., 1, None] - cells % b))  # (N, 4, B*B)
        dist = np.where(state.food.reshape(n, 1, b * b), dist, 4 * b)
        nearest = dist.min(axis=2)
        noise = self.rng.random((n, 4))  # < 1, so it only breaks ties
        score = np.where(mask, -nearest + 0.5 * noise, -1e6 + noise)
        return np.argmax(score, axis=1)
