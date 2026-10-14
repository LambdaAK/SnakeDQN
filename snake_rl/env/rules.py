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


@dataclass
class StepEvents:
    ate: np.ndarray           # bool (N, S) snake ate food this turn
    died: np.ndarray          # bool (N, S) snake was eliminated this turn
    alive_before: np.ndarray  # bool (N, S) snake was alive in a running game before the turn
    game_over: np.ndarray     # bool (N,)  game ended this turn
    winner: np.ndarray        # int  (N,)  winning snake, -1 for none / draw / single-snake


def _occupancy(state: GameState, mask: np.ndarray, start: int = 0,
               end_offset: int = 0) -> np.ndarray:
    """Count, per cell, the segments [start, length - end_offset) of snakes where mask is True.

    Off-board segments (a head that just left the board) are ignored.
    Returns int (N, B*B).
    """
    n, b = state.num_envs, state.board_size
    seg = int(state.length.max())
    pos = state.body[:, :, :seg].astype(np.int64)
    r, c = pos[..., 0], pos[..., 1]
    idx = np.arange(seg)
    valid = (
        mask[:, :, None]
        & (idx >= start)
        & (idx < state.length[:, :, None] - end_offset)
        & (r >= 0) & (r < b) & (c >= 0) & (c < b)
    )
    flat = (np.arange(n)[:, None, None] * b + r) * b + c
    return np.bincount(flat[valid], minlength=n * b * b).reshape(n, b * b)


def _validate_actions(actions, num_envs: int, num_snakes: int) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.shape != (num_envs, num_snakes):
        raise ValueError(f"actions must have shape {(num_envs, num_snakes)}, got {actions.shape}")
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError(f"actions must be integers, got dtype {actions.dtype}")
    if ((actions < 0) | (actions > 3)).any():
        raise ValueError("actions must be in 0..3 (up, down, left, right)")
    return actions


def _spawn_food(state: GameState, active: np.ndarray, spawn_u: np.ndarray,
                on_board: np.ndarray) -> None:
    n, b = state.num_envs, state.board_size
    flat_food = state.food.reshape(n, b * b)
    count = flat_food.sum(axis=1)
    need = active & ((count < MIN_FOOD) | (spawn_u[:, 0] < SPAWN_CHANCE))
    idx = np.flatnonzero(need)
    if idx.size == 0:
        return
    occupied = _occupancy(state, on_board & need[:, None])[idx] > 0
    free = ~flat_food[idx] & ~occupied
    n_free = free.sum(axis=1)
    pick = np.floor(spawn_u[idx, 1] * n_free).astype(np.int64)
    cell = np.argmax(np.cumsum(free, axis=1) > pick[:, None], axis=1)
    ok = n_free > 0
    flat_food[idx[ok], cell[ok]] = True


def step_games(state: GameState, actions, spawn_u: np.ndarray) -> StepEvents:
    """Advance every running game by one turn, in place, following Battlesnake order:
    move, reduce health, feed, spawn food, eliminate."""
    n, s, b = state.num_envs, state.num_snakes, state.board_size
    actions = _validate_actions(actions, n, s)
    rows = np.arange(n)[:, None]
    active = ~state.over
    moving = state.alive & active[:, None]
    alive_before = moving.copy()

    # Move: every segment follows the one ahead of it; the old last segment drops off.
    seg = min(int(state.length.max()) + 1, state.body.shape[2])
    new_head = state.body[:, :, 0] + DIRECTIONS[actions]
    state.body[moving, 1:seg] = state.body[moving, 0:seg - 1]
    state.body[moving, 0] = new_head[moving]
    state.health[moving] -= 1

    # Feed: grow by duplicating the tail, so the tail stays put next turn.
    r, c = new_head[..., 0], new_head[..., 1]
    in_bounds = (r >= 0) & (r < b) & (c >= 0) & (c < b)
    rc, cc = np.clip(r, 0, b - 1), np.clip(c, 0, b - 1)
    ate = moving & in_bounds & state.food[rows, rc, cc]
    en, sn = np.nonzero(ate)
    tail = state.length[en, sn]
    state.body[en, sn, tail] = state.body[en, sn, tail - 1]
    state.length[ate] += 1
    state.health[ate] = state.max_health
    state.food[en, rc[en, sn], cc[en, sn]] = False

    _spawn_food(state, active, spawn_u, moving)

    # Eliminate: starved and off-board snakes first, then body and head-to-head
    # collisions among the remaining snakes, applied simultaneously.
    survivors = moving & in_bounds & (state.health > 0)
    occupied = _occupancy(state, survivors, start=1)
    head_flat = rc.astype(np.int64) * b + cc
    body_hit = occupied[rows, head_flat] > 0
    same_cell = head_flat[:, :, None] == head_flat[:, None, :]
    pair = survivors[:, :, None] & survivors[:, None, :] & ~np.eye(s, dtype=bool)
    other_not_shorter = state.length[:, None, :] >= state.length[:, :, None]
    lost_head_to_head = np.any(same_cell & pair & other_not_shorter, axis=2)
    died = moving & ~(survivors & ~body_hit & ~lost_head_to_head)
    state.alive &= ~died

    state.turn[active] += 1
    n_alive = state.alive.sum(axis=1)
    game_over = active & (n_alive <= (0 if s == 1 else 1))
    winner = np.where(game_over & (n_alive == 1) & (s > 1), np.argmax(state.alive, axis=1), -1)
    state.over |= game_over
    return StepEvents(ate=ate, died=died, alive_before=alive_before,
                      game_over=game_over, winner=winner)


def blocked_cells(state: GameState) -> np.ndarray:
    """Cells still occupied next turn: every alive segment except each tail. Returns bool (N, B, B)."""
    n, b = state.num_envs, state.board_size
    running = state.alive & ~state.over[:, None]
    return (_occupancy(state, running, start=0, end_offset=1) > 0).reshape(n, b, b)


def action_mask(state: GameState) -> np.ndarray:
    """Moves that don't immediately hit a wall or a body. Returns bool (N, S, 4)."""
    n, b = state.num_envs, state.board_size
    blocked = blocked_cells(state).reshape(n, b * b)
    nxt = state.body[:, :, 0, None, :].astype(np.int64) + DIRECTIONS
    r, c = nxt[..., 0], nxt[..., 1]
    in_bounds = (r >= 0) & (r < b) & (c >= 0) & (c < b)
    flat = np.clip(r, 0, b - 1) * b + np.clip(c, 0, b - 1)
    free = ~blocked[np.arange(n)[:, None, None], flat]
    running = state.alive & ~state.over[:, None]
    return in_bounds & free & running[:, :, None]
