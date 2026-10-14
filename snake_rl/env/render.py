"""Terminal rendering of one game from a GameState."""

from __future__ import annotations

import numpy as np

from snake_rl.env.rules import GameState

RESET = "\033[0m"
FOOD_COLOR = "\033[91m"
SNAKE_COLORS = ["\033[92m", "\033[94m", "\033[93m", "\033[95m",
                "\033[96m", "\033[97m", "\033[90m", "\033[33m"]


def render(state: GameState, env: int = 0, color: bool = True) -> str:
    def paint(text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    b = state.board_size
    grid = [["  "] * b for _ in range(b)]
    for r, c in zip(*np.nonzero(state.food[env])):
        grid[r][c] = paint("◆◆", FOOD_COLOR)
    for k in range(state.num_snakes):
        if not state.alive[env, k]:
            continue
        code = SNAKE_COLORS[k % len(SNAKE_COLORS)]
        for i in range(int(state.length[env, k]) - 1, -1, -1):
            r, c = state.body[env, k, i]
            grid[r][c] = paint("██" if i == 0 else "▓▓", code)

    border = "═" * (2 * b)
    lines = [f"Turn {state.turn[env]}", f"╔{border}╗"]
    lines += [f"║{''.join(row)}║" for row in grid]
    lines.append(f"╚{border}╝")
    for k in range(state.num_snakes):
        status = (f"Snake {k}: length {state.length[env, k]}, health {state.health[env, k]}"
                  + ("" if state.alive[env, k] else " (dead)"))
        lines.append(paint(status, SNAKE_COLORS[k % len(SNAKE_COLORS)]))
    return "\n".join(lines)
