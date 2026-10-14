"""Reward functions: StepEvents -> float32 (N, S)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from snake_rl.env.rules import StepEvents

RewardFn = Callable[[StepEvents], np.ndarray]


def winloss(events: StepEvents) -> np.ndarray:
    """+1 to the winner when a game ends, -1 on death, 0 when the last snakes die together."""
    rewards = np.zeros(events.died.shape, np.float32)
    draw = events.game_over & (events.winner < 0)
    rewards[events.died & ~draw[:, None]] = -1.0
    won = np.flatnonzero(events.game_over & (events.winner >= 0))
    rewards[won, events.winner[won]] = 1.0
    return rewards


def food(events: StepEvents) -> np.ndarray:
    """+1 per food eaten, -1 on death."""
    return events.ate.astype(np.float32) - events.died.astype(np.float32)


REWARDS: dict[str, RewardFn] = {"winloss": winloss, "food": food}


def get_reward(name: str) -> RewardFn:
    try:
        return REWARDS[name]
    except KeyError:
        raise ValueError(f"unknown reward {name!r}; choose from {sorted(REWARDS)}") from None
