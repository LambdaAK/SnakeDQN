"""Hamiltonian-cycle bot: never collides and eventually fills the board (single-snake only)."""

from __future__ import annotations

import numpy as np

from snake_rl.env.rules import DIRECTIONS, GameState


def hamiltonian_cycle(board_size: int) -> list[tuple[int, int]]:
    """A cycle through every cell: along row 0, snake through columns 1.., back up column 0."""
    if board_size % 2:
        raise ValueError(f"a Hamiltonian cycle needs an even board size, got {board_size}")
    b = board_size
    path = [(0, c) for c in range(b)]
    for r in range(1, b):
        cols = range(b - 1, 0, -1) if r % 2 == 1 else range(1, b)
        path += [(r, c) for c in cols]
    path += [(r, 0) for r in range(b - 1, 0, -1)]
    return path


def _direction_index(d_row: int, d_col: int) -> int:
    return int(np.flatnonzero((DIRECTIONS == (d_row, d_col)).all(axis=1))[0])


class HamiltonianBot:
    """Needs max_health >= board_size**2 so it never starves between foods."""

    name = "hamiltonian"

    def __init__(self, board_size: int):
        cycle = hamiltonian_cycle(board_size)
        self.board_size = board_size
        self.next_action = np.zeros((board_size, board_size), np.int64)
        for (r, c), (nr, nc) in zip(cycle, cycle[1:] + cycle[:1]):
            self.next_action[r, c] = _direction_index(nr - r, nc - c)

    def __call__(self, state: GameState, snake: int) -> np.ndarray:
        if state.board_size != self.board_size:
            raise ValueError(f"bot built for board {self.board_size}, got {state.board_size}")
        head = state.body[:, snake, 0]
        b = self.board_size
        return self.next_action[np.clip(head[:, 0], 0, b - 1), np.clip(head[:, 1], 0, b - 1)]
