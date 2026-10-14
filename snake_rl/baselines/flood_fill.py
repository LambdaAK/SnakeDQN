"""Flood-fill heuristic: keep room to move, dodge losing head-to-heads, eat when needed."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from snake_rl.env.rules import DIRECTIONS, GameState, action_mask, blocked_cells


class FloodFillBot:
    name = "flood_fill"

    def __init__(self, seed: int | None = None, hunger_threshold: int = 30):
        self.rng = np.random.default_rng(seed)
        self.hunger_threshold = hunger_threshold

    def __call__(self, state: GameState, snake: int) -> np.ndarray:
        mask = action_mask(state)[:, snake]
        blocked = blocked_cells(state)
        actions = self.rng.integers(0, 4, size=state.num_envs)
        for env in np.flatnonzero(mask.any(axis=1)):
            actions[env] = self._choose(state, snake, env, mask[env], blocked[env])
        return actions

    def _choose(self, state, snake, env, safe, blocked) -> int:
        b = state.board_size
        head = state.body[env, snake, 0].astype(np.int64)
        my_length = int(state.length[env, snake])
        enemies = [k for k in range(state.num_snakes) if k != snake and state.alive[env, k]]
        threats = set()
        for k in enemies:
            if state.length[env, k] >= my_length:
                enemy_head = state.body[env, k, 0].astype(np.int64)
                threats.update((int(r), int(c)) for r, c in enemy_head + DIRECTIONS)

        labels, _ = ndimage.label(~blocked)
        sizes = np.bincount(labels.ravel())
        food_r, food_c = np.nonzero(state.food[env])

        candidates = []
        for action in self.rng.permutation(np.flatnonzero(safe)):
            r, c = (int(v) for v in head + DIRECTIONS[action])
            food_dist = (int(np.min(np.abs(food_r - r) + np.abs(food_c - c)))
                         if food_r.size else 2 * b)
            candidates.append({
                "action": int(action),
                "area": int(sizes[labels[r, c]]),
                "risky": (r, c) in threats,
                "food": food_dist,
            })

        pool = [m for m in candidates if not m["risky"]] or candidates
        roomy = [m for m in pool if m["area"] >= my_length]
        if roomy:
            pool = roomy
        else:
            best = max(m["area"] for m in pool)
            pool = [m for m in pool if m["area"] == best]

        hungry = (state.health[env, snake] <= self.hunger_threshold
                  or any(state.length[env, k] >= my_length for k in enemies))
        if hungry:
            choice = min(pool, key=lambda m: (m["food"], -m["area"]))
        else:
            choice = min(pool, key=lambda m: (-m["area"], m["food"]))
        return choice["action"]
