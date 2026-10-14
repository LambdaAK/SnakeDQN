"""Batched Battlesnake rules in pure NumPy.

Arrays are indexed [env, snake, ...]. Positions are (row, col) with row 0 at the
top. Every function operates on a whole batch of games at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

START_LENGTH = 3
MIN_FOOD = 1
SPAWN_CHANCE = 0.15
MAX_SNAKES = 8
MIN_BOARD_SIZE = 7

# Action index -> (d_row, d_col): up, down, left, right.
DIRECTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int16)
DIAGONALS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.int16)


@dataclass
class GameState:
    body: np.ndarray    # int16 (N, S, B*B + 1, 2); segment 0 is the head
    length: np.ndarray  # int16 (N, S)
    health: np.ndarray  # int16 (N, S)
    alive: np.ndarray   # bool  (N, S)
    food: np.ndarray    # bool  (N, B, B)
    turn: np.ndarray    # int32 (N,)
    over: np.ndarray    # bool  (N,) game has ended and awaits a reset
    max_health: int

    @property
    def num_envs(self) -> int:
        return self.body.shape[0]

    @property
    def num_snakes(self) -> int:
        return self.body.shape[1]

    @property
    def board_size(self) -> int:
        return self.food.shape[1]

    @classmethod
    def empty(cls, num_envs: int, num_snakes: int, board_size: int,
              max_health: int = 100) -> GameState:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if not 1 <= num_snakes <= MAX_SNAKES:
            raise ValueError(f"num_snakes must be in 1..{MAX_SNAKES}, got {num_snakes}")
        if board_size < MIN_BOARD_SIZE:
            raise ValueError(f"board_size must be >= {MIN_BOARD_SIZE}, got {board_size}")
        if max_health < 1:
            raise ValueError(f"max_health must be >= 1, got {max_health}")
        n, s, b = num_envs, num_snakes, board_size
        return cls(
            body=np.zeros((n, s, b * b + 1, 2), np.int16),
            length=np.zeros((n, s), np.int16),
            health=np.zeros((n, s), np.int16),
            alive=np.zeros((n, s), bool),
            food=np.zeros((n, b, b), bool),
            turn=np.zeros(n, np.int32),
            over=np.ones(n, bool),
            max_health=max_health,
        )

    @classmethod
    def from_layout(cls, board_size: int, snakes, food=(), health=None,
                    max_health: int = 100) -> GameState:
        """Build a single game from explicit coordinates (head first) for tests and debugging."""
        state = cls.empty(1, len(snakes), board_size, max_health)
        for k, segments in enumerate(snakes):
            state.body[0, k, :len(segments)] = segments
            state.length[0, k] = len(segments)
            state.health[0, k] = max_health if health is None else health[k]
            state.alive[0, k] = True
        for r, c in food:
            state.food[0, r, c] = True
        state.over[0] = False
        return state

    def take(self, envs) -> GameState:
        """Deep copy of the selected games."""
        envs = np.atleast_1d(np.asarray(envs))
        return GameState(
            body=self.body[envs].copy(),
            length=self.length[envs].copy(),
            health=self.health[envs].copy(),
            alive=self.alive[envs].copy(),
            food=self.food[envs].copy(),
            turn=self.turn[envs].copy(),
            over=self.over[envs].copy(),
            max_health=self.max_health,
        )


def start_points(board_size: int) -> np.ndarray:
    """The 8 symmetric Battlesnake start cells: corners and edge midpoints, one cell in."""
    a, m, z = 1, board_size // 2, board_size - 2
    return np.array(
        [[a, a], [a, m], [a, z], [m, a], [m, z], [z, a], [z, m], [z, z]], dtype=np.int16
    )


def reset_games(state: GameState, mask: np.ndarray, rng: np.random.Generator) -> None:
    """Start new games in place for every env where mask is True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    m, s, b = idx.size, state.num_snakes, state.board_size
    points = start_points(b)
    order = np.argsort(rng.random((m, len(points))), axis=1)[:, :s]
    starts = points[order]  # (m, s, 2)

    state.body[idx] = 0
    state.body[idx, :, :START_LENGTH] = starts[:, :, None, :]
    state.length[idx] = START_LENGTH
    state.health[idx] = state.max_health
    state.alive[idx] = True
    state.turn[idx] = 0
    state.over[idx] = False

    state.food[idx] = False
    centre = b // 2
    state.food[idx, centre, centre] = True
    food_pos = starts + DIAGONALS[rng.integers(0, len(DIAGONALS), size=(m, s))]
    state.food[idx[:, None], food_pos[..., 0], food_pos[..., 1]] = True
