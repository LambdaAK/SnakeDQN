# Foundation: Vectorized Multi-Snake Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-snake DQN code with a tested, fast, batched Battlesnake environment plus scripted baseline bots and an evaluation harness. These are the base for the later benchmark, self-play and browser-demo sub-projects.

**Architecture:** Game state lives in a `GameState` dataclass of NumPy arrays indexed `[env, snake, ...]`. Pure functions in `rules.py` reset and step whole batches. `observations.py` and `rewards.py` turn that state into agent inputs. `SnakeEnv` wraps them with auto-reset. Bots are callables `(state, snake_index) -> actions[num_envs]`. The harness seats any mix of them in a `SnakeEnv` and reports win rates with confidence intervals.

**Tech Stack:** Python ≥ 3.11, NumPy, SciPy (`ndimage.label` for flood fill), pytest, uv, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-14-foundation-design.md`

## Global Constraints

- Python `>=3.11`; dependencies `numpy>=1.26`, `scipy>=1.11`; dev dependency `pytest>=8`. No PyTorch in `snake_rl/` during Foundation.
- Commands run through uv: `uv run pytest ...`, `uv run python scripts/...`.
- Default board 11×11; minimum board size 7; at most 8 snakes per game.
- Rules: start length 3 stacked on one cell, health 100, −1 health per turn, eating resets health to max and grows by duplicating the tail, minimum 1 food, 15% food spawn chance per turn, no hazards.
- Actions: `0=up, 1=down, 2=left, 3=right`; positions are `(row, col)` with row 0 at the top.
- Observation: `float32 (num_envs, num_snakes, 8, H, W)`, no hand-engineered features.
- Rewards: `"winloss"` (default when `num_snakes > 1`) and `"food"` (default when `num_snakes == 1`).
- Performance: ≥ 200,000 env steps/sec at `num_envs=1024, num_snakes=2` on the user's Mac CPU.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (passed as a second `-m`).

## Deviations from the spec

These are decided here so implementers don't have to guess:

1. **Hamiltonian bot:** a Hamiltonian cycle exists only on even-sized boards, and it needs `max_health >= board_size**2` to avoid starving. `SnakeEnv` therefore takes a `max_health` argument (default 100). The bot raises `ValueError` on odd boards.
2. **`SnakeEnv(reward=None)`:** `None` picks the per-mode default above. The spec's signature showed `reward="winloss"`.
3. **`scripts/evaluate.py` seats:** bot names only. Seating a checkpoint arrives in sub-project 2, when a checkpoint format exists.
4. **Body-age channel:** value is `(length - i) / length` for segment `i`, so the head is 1 and the tail is `1/length` (empty cells are 0).

## Review Focus

1. Malformed `actions` (wrong shape, non-integer, value outside 0–3) must raise `ValueError`, not wrap or silently move. Test: Task 3 `test_invalid_actions_raise`.
2. A board with no free cell must skip food spawning without crashing. Test: Task 3 `test_full_board_spawns_nothing`.
3. Actions for dead snakes and finished games must be ignored. Tests: Task 3 `test_dead_snake_action_ignored`, `test_finished_game_does_not_change`.
4. Calling `step()` before `reset()` must raise `RuntimeError`, not index garbage. Test: Task 6 `test_step_before_reset_raises`.
5. Unsupported configurations (board < 7, > 8 snakes, odd board for the Hamiltonian bot, unknown bot or reward name) must raise `ValueError`. Tests: Task 2 `test_empty_validates_arguments`, Task 6 `test_unknown_reward_raises`, Task 7 `test_hamiltonian_rejects_odd_board`, `test_make_bot_unknown_name`.

## File Structure

```
pyproject.toml                       # package + deps + pytest config            (Task 1)
.gitignore                           #                                           (Task 1)
.github/workflows/ci.yml             # uv + pytest on push / PR                  (Task 1)
legacy/                              # old code, moved in Task 1, deleted in Task 11
snake_rl/__init__.py                 # version                                   (Task 1)
snake_rl/env/__init__.py             # public env API re-exports                 (Tasks 2,5,6,10)
snake_rl/env/rules.py                # GameState, reset_games, step_games, masks (Tasks 2-4)
snake_rl/env/observations.py         # build_observations                        (Task 5)
snake_rl/env/rewards.py              # winloss, food, get_reward                 (Task 6)
snake_rl/env/snake_env.py            # SnakeEnv                                  (Task 6)
snake_rl/env/render.py               # render(state) -> str                      (Task 10)
snake_rl/baselines/__init__.py       # make_bot, BOT_NAMES                       (Tasks 7,8)
snake_rl/baselines/simple.py         # RandomBot, SafeRandomBot, GreedyBot       (Task 7)
snake_rl/baselines/hamiltonian.py    # HamiltonianBot                            (Task 7)
snake_rl/baselines/flood_fill.py     # FloodFillBot                              (Task 8)
snake_rl/eval/__init__.py            # re-exports                                (Task 9)
snake_rl/eval/harness.py             # run_matches, MatchResult, CIs             (Task 9)
scripts/benchmark_env.py             #                                           (Task 6)
scripts/evaluate.py                  #                                           (Task 9)
scripts/play.py                      #                                           (Task 10)
results/*.json                       # baseline ladder results                   (Task 9)
tests/...                            # one test module per unit, listed per task
```

---

### Task 1: Project scaffolding and repo cleanup

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `snake_rl/__init__.py`, `snake_rl/env/__init__.py`, `tests/test_package.py`
- Move to `legacy/`: `snake_game.py dqn_agent.py reinforce_agent.py train.py run_trained_model.py monitor_training.py test_dqn.py colors.py reward_function.txt state_representation.txt requirements.txt`
- Untrack (keep on disk): `__pycache__/`, `agents/`, `.DS_Store`

**Interfaces:**
- Produces: importable package `snake_rl` with `snake_rl.__version__ == "0.1.0"`; empty subpackage `snake_rl.env`.

- [ ] **Step 1: Move legacy code and untrack generated files**

```bash
mkdir -p legacy
git mv snake_game.py dqn_agent.py reinforce_agent.py train.py run_trained_model.py monitor_training.py test_dqn.py colors.py reward_function.txt state_representation.txt requirements.txt legacy/
git rm -r --cached --quiet __pycache__ agents .DS_Store
```

Expected: `git status` shows renames into `legacy/` and deletions of `__pycache__/*`, `agents/*.pth`, `.DS_Store`. The `agents/` files still exist on disk (`ls agents | head`).

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.DS_Store
.venv/
*.egg-info/
.pytest_cache/
*.pth
agents/
checkpoints/
runs/
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "snake-rl"
version = "0.1.0"
description = "Multi-agent Snake reinforcement learning: a vectorized Battlesnake environment, baselines and evaluation"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "scipy>=1.11"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
include = ["snake_rl*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Write the failing package test** — `tests/test_package.py`

```python
import snake_rl


def test_version():
    assert snake_rl.__version__ == "0.1.0"
```

- [ ] **Step 5: Run it to verify it fails**

Run: `uv run pytest tests/test_package.py -v`
Expected: FAIL / collection error, `ModuleNotFoundError: No module named 'snake_rl'`.

- [ ] **Step 6: Create the package**

`snake_rl/__init__.py`:

```python
"""Reinforcement learning for multi-agent Snake."""

__version__ = "0.1.0"
```

`snake_rl/env/__init__.py`:

```python
"""Vectorized Battlesnake environment."""
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest tests/test_package.py -v`
Expected: PASS.

- [ ] **Step 8: Add CI** — `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv run pytest -q
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Restructure into snake_rl package and move old code to legacy/" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Game state and reset

**Files:**
- Create: `snake_rl/env/rules.py`
- Modify: `snake_rl/env/__init__.py`
- Test: `tests/test_rules_reset.py`

**Interfaces:**
- Produces (in `snake_rl.env.rules`, re-exported from `snake_rl.env`):
  - Constants `START_LENGTH = 3`, `MIN_FOOD = 1`, `SPAWN_CHANCE = 0.15`, `MAX_SNAKES = 8`, `MIN_BOARD_SIZE = 7`, `DIRECTIONS: int16 (4, 2)` (up, down, left, right), `DIAGONALS: int16 (4, 2)`.
  - `GameState` dataclass, fields `body int16 (N,S,B*B+1,2)`, `length int16 (N,S)`, `health int16 (N,S)`, `alive bool (N,S)`, `food bool (N,B,B)`, `turn int32 (N,)`, `over bool (N,)`, `max_health int`. Properties `num_envs`, `num_snakes`, `board_size`. Classmethods `empty(num_envs, num_snakes, board_size, max_health=100)` and `from_layout(board_size, snakes, food=(), health=None, max_health=100)`. Method `take(envs) -> GameState` (deep copy of the selected games).
  - `start_points(board_size) -> int16 (8, 2)`.
  - `reset_games(state, mask: bool (N,), rng: np.random.Generator) -> None` (in place).

- [ ] **Step 1: Write the failing tests** — `tests/test_rules_reset.py`

```python
import numpy as np
import pytest

from snake_rl.env.rules import (
    DIAGONALS,
    START_LENGTH,
    GameState,
    reset_games,
    start_points,
)


def fresh(num_envs=64, num_snakes=4, board_size=11, seed=0):
    state = GameState.empty(num_envs, num_snakes, board_size)
    reset_games(state, np.ones(num_envs, bool), np.random.default_rng(seed))
    return state


def test_empty_validates_arguments():
    with pytest.raises(ValueError):
        GameState.empty(1, 1, 6)
    with pytest.raises(ValueError):
        GameState.empty(1, 9, 11)
    with pytest.raises(ValueError):
        GameState.empty(1, 0, 11)
    with pytest.raises(ValueError):
        GameState.empty(0, 1, 11)


def test_empty_state_is_over_until_reset():
    state = GameState.empty(3, 2, 11)
    assert state.over.all()
    assert not state.alive.any()
    assert state.body.shape == (3, 2, 11 * 11 + 1, 2)


def test_reset_places_stacked_snakes_on_distinct_start_points():
    state = fresh()
    points = {tuple(p) for p in start_points(11)}
    assert (state.length == START_LENGTH).all()
    assert (state.health == 100).all()
    assert state.alive.all()
    assert not state.over.any()
    assert (state.turn == 0).all()
    for env in range(state.num_envs):
        heads = [tuple(state.body[env, k, 0]) for k in range(4)]
        assert len(set(heads)) == 4
        assert set(heads) <= points
        for k in range(4):
            segments = state.body[env, k, :START_LENGTH]
            assert (segments == segments[0]).all()


def test_reset_places_centre_food_and_one_diagonal_food_per_snake():
    state = fresh()
    centre = 11 // 2
    for env in range(state.num_envs):
        assert state.food[env, centre, centre]
        for k in range(4):
            head = state.body[env, k, 0]
            diagonal_food = [state.food[env, head[0] + dr, head[1] + dc] for dr, dc in DIAGONALS]
            assert any(diagonal_food)
        assert 2 <= state.food[env].sum() <= 5


def test_reset_only_touches_masked_games():
    state = fresh(num_envs=4, num_snakes=2)
    before = state.take([1, 3])
    mask = np.array([True, False, True, False])
    reset_games(state, mask, np.random.default_rng(1))
    after = state.take([1, 3])
    assert np.array_equal(before.body, after.body)
    assert np.array_equal(before.food, after.food)


def test_from_layout_and_take():
    state = GameState.from_layout(
        7, snakes=[[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)], health=[42]
    )
    assert state.num_envs == 1 and state.num_snakes == 1 and state.board_size == 7
    assert state.length[0, 0] == 3 and state.health[0, 0] == 42
    assert state.food[0, 0, 0] and state.food.sum() == 1
    assert not state.over[0]
    copy = state.take([0])
    copy.health[0, 0] = 1
    assert state.health[0, 0] == 42


def test_eight_snakes_fit_on_smallest_board():
    state = fresh(num_envs=8, num_snakes=8, board_size=7)
    for env in range(8):
        heads = {tuple(state.body[env, k, 0]) for k in range(8)}
        assert len(heads) == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rules_reset.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'snake_rl.env.rules'`.

- [ ] **Step 3: Implement** — `snake_rl/env/rules.py`

```python
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
```

`snake_rl/env/__init__.py`:

```python
"""Vectorized Battlesnake environment."""

from snake_rl.env.rules import (
    DIRECTIONS,
    MAX_SNAKES,
    MIN_BOARD_SIZE,
    START_LENGTH,
    GameState,
    reset_games,
    start_points,
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_rules_reset.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/env tests/test_rules_reset.py
git commit -m "Add batched GameState and Battlesnake reset" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Stepping games (movement, feeding, food spawn, elimination)

**Files:**
- Modify: `snake_rl/env/rules.py` (append), `snake_rl/env/__init__.py`
- Test: `tests/test_rules_step.py`

**Interfaces:**
- Consumes: `GameState`, constants from Task 2.
- Produces:
  - `StepEvents` dataclass: `ate bool (N,S)`, `died bool (N,S)`, `alive_before bool (N,S)`, `game_over bool (N,)`, `winner int64 (N,)` (−1 for no winner, a draw, or single-snake mode).
  - `step_games(state, actions: int (N,S), spawn_u: float (N,2)) -> StepEvents`, in place. `spawn_u[:, 0]` is compared with `SPAWN_CHANCE`; `spawn_u[:, 1]` picks the `floor(u * n_free)`-th free cell in row-major order. Finished games (`state.over`) are skipped.
  - Private `_occupancy(state, mask, start=0, end_offset=0) -> int (N, B*B)`, reused by Task 4.

- [ ] **Step 1: Write the failing tests** — `tests/test_rules_step.py`

```python
import numpy as np
import pytest

from snake_rl.env.rules import GameState, step_games

NO_SPAWN = (0.99, 0.0)  # no random spawn; a forced spawn picks the first free cell


def step(state, *actions, spawn=NO_SPAWN):
    return step_games(state, np.array([actions]), np.array([spawn]))


def body(state, k):
    return [tuple(int(v) for v in p) for p in state.body[0, k, :state.length[0, k]]]


def test_move_shifts_body_and_costs_health():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)])
    events = step(state, 0)
    assert body(state, 0) == [(2, 3), (3, 3), (4, 3)]
    assert state.health[0, 0] == 99
    assert state.turn[0] == 1
    assert not events.died.any() and not events.game_over.any()


def test_wall_collision_kills():
    state = GameState.from_layout(7, [[(0, 3), (1, 3), (2, 3)]], food=[(6, 6)])
    events = step(state, 0)
    assert events.died[0, 0] and not state.alive[0, 0]
    assert events.game_over[0] and state.over[0]
    assert events.winner[0] == -1


def test_self_collision_kills():
    state = GameState.from_layout(7, [[(2, 2), (2, 3), (3, 3), (3, 2), (4, 2)]], food=[(6, 6)])
    events = step(state, 1)  # down into (3, 2), still part of the body
    assert events.died[0, 0]


def test_neck_reversal_kills():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)])
    events = step(state, 1)
    assert events.died[0, 0]


def test_moving_into_vacating_tail_is_safe():
    state = GameState.from_layout(7, [[(2, 2), (2, 3), (3, 3), (3, 2)]], food=[(6, 6)])
    events = step(state, 1)
    assert not events.died.any()
    assert body(state, 0) == [(3, 2), (2, 2), (2, 3), (3, 3)]


def test_tail_that_just_ate_is_fatal():
    # A duplicated tail segment is what eating leaves behind.
    state = GameState.from_layout(
        7, [[(2, 2), (2, 3), (3, 3), (3, 2), (3, 2)]], food=[(6, 6)]
    )
    events = step(state, 1)
    assert events.died[0, 0]


def test_eating_grows_duplicates_tail_and_restores_health():
    state = GameState.from_layout(
        7, [[(3, 3), (4, 3), (5, 3)]], food=[(2, 3), (0, 0)], health=[50]
    )
    events = step(state, 0)
    assert events.ate[0, 0]
    assert state.length[0, 0] == 4
    assert state.health[0, 0] == 100
    assert body(state, 0) == [(2, 3), (3, 3), (4, 3), (4, 3)]
    assert not state.food[0, 2, 3]


def test_starvation_kills():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)], health=[1])
    events = step(state, 0)
    assert events.died[0, 0]


def test_eating_on_last_health_point_survives():
    state = GameState.from_layout(
        7, [[(3, 3), (4, 3), (5, 3)]], food=[(2, 3), (0, 0)], health=[1]
    )
    events = step(state, 0)
    assert not events.died.any()
    assert state.health[0, 0] == 100


def test_head_to_head_shorter_snake_dies():
    state = GameState.from_layout(
        7,
        [[(3, 2), (3, 1), (3, 0)], [(3, 4), (3, 5), (3, 6), (2, 6)]],
        food=[(0, 0)],
    )
    events = step(state, 3, 2)
    assert events.died[0].tolist() == [True, False]
    assert events.game_over[0] and events.winner[0] == 1


def test_head_to_head_equal_length_both_die():
    state = GameState.from_layout(
        7, [[(3, 2), (3, 1), (3, 0)], [(3, 4), (3, 5), (3, 6)]], food=[(0, 0)]
    )
    events = step(state, 3, 2)
    assert events.died[0].tolist() == [True, True]
    assert events.game_over[0] and events.winner[0] == -1


def test_moving_into_other_snakes_body_kills():
    state = GameState.from_layout(
        7, [[(2, 3), (1, 3), (0, 3)], [(3, 4), (3, 3), (3, 2)]], food=[(6, 6)]
    )
    events = step(state, 1, 3)
    assert events.died[0].tolist() == [True, False]
    assert events.winner[0] == 1


def test_moving_into_other_snakes_vacating_tail_is_safe():
    state = GameState.from_layout(
        7, [[(2, 3), (1, 3), (0, 3)], [(3, 5), (3, 4), (3, 3)]], food=[(6, 6)]
    )
    events = step(state, 1, 3)
    assert not events.died.any()
    assert not events.game_over[0]


def test_dead_snake_action_ignored():
    state = GameState.from_layout(
        7,
        [[(3, 3), (4, 3), (5, 3)], [(1, 1), (1, 2), (1, 3)], [(5, 5), (5, 6), (6, 6)]],
        food=[(0, 6)],
    )
    state.alive[0, 2] = False
    before = body(state, 2)
    events = step(state, 0, 1, 0)
    assert body(state, 2) == before
    assert not events.alive_before[0, 2] and not events.died[0, 2]


def test_finished_game_does_not_change():
    state = GameState.from_layout(7, [[(0, 3), (1, 3), (2, 3)]], food=[(6, 6)])
    step(state, 0)
    snapshot = state.take([0])
    events = step(state, 0)
    assert state.turn[0] == snapshot.turn[0]
    assert np.array_equal(state.body, snapshot.body)
    assert np.array_equal(state.food, snapshot.food)
    assert not events.game_over[0] and not events.died.any() and not events.alive_before.any()


def test_forced_spawn_when_no_food_left():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(2, 3)])
    step(state, 0, spawn=(0.99, 0.0))
    assert state.food[0].sum() == 1
    assert state.food[0, 0, 0]  # first free cell in row-major order


def test_chance_spawn():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(6, 6)])
    step(state, 0, spawn=(0.2, 0.0))
    assert state.food[0].sum() == 1
    step(state, 0, spawn=(0.1, 0.0))
    assert state.food[0].sum() == 2 and state.food[0, 0, 0]


def test_spawn_fraction_selects_cell():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(6, 6)])
    step(state, 0, spawn=(0.1, 0.999999))
    assert state.food[0, 6, 5]  # last free cell, since (6, 6) already holds food


def test_full_board_spawns_nothing():
    path = []
    for r in range(7):
        cols = range(7) if r % 2 == 0 else range(6, -1, -1)
        path += [(r, c) for c in cols]
    segments = list(reversed(path)) + [path[0]]  # head (6, 6), duplicated tail
    state = GameState.from_layout(7, [segments])
    events = step(state, 1)  # off the board
    assert events.died[0, 0]
    assert state.food[0].sum() == 0


def test_invalid_actions_raise():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)])
    with pytest.raises(ValueError):
        step_games(state, np.array([[4]]), np.array([NO_SPAWN]))
    with pytest.raises(ValueError):
        step_games(state, np.array([[-1]]), np.array([NO_SPAWN]))
    with pytest.raises(ValueError):
        step_games(state, np.array([0]), np.array([NO_SPAWN]))
    with pytest.raises(ValueError):
        step_games(state, np.array([[0.0]]), np.array([NO_SPAWN]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rules_step.py -v`
Expected: FAIL, `ImportError: cannot import name 'step_games'`.

- [ ] **Step 3: Implement** — append to `snake_rl/env/rules.py`

```python
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
    occupied = _occupancy(state, on_board)[idx] > 0
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
```

Add `StepEvents` and `step_games` to the import list in `snake_rl/env/__init__.py`:

```python
"""Vectorized Battlesnake environment."""

from snake_rl.env.rules import (
    DIRECTIONS,
    MAX_SNAKES,
    MIN_BOARD_SIZE,
    START_LENGTH,
    GameState,
    StepEvents,
    reset_games,
    start_points,
    step_games,
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_rules_step.py tests/test_rules_reset.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/env tests/test_rules_step.py
git commit -m "Add batched Battlesnake step: movement, feeding, food spawn, elimination" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Action mask, reference oracle, batch consistency

**Files:**
- Modify: `snake_rl/env/rules.py` (append), `snake_rl/env/__init__.py`
- Create: `tests/reference_rules.py` (slow oracle, not a test module)
- Test: `tests/test_action_mask.py`, `tests/test_rules_reference.py`

**Interfaces:**
- Consumes: `GameState`, `step_games`, `reset_games`, `_occupancy`, `DIRECTIONS`.
- Produces:
  - `blocked_cells(state) -> bool (N, B, B)`: cells that will still hold a snake segment next turn. That is every segment of every alive snake except each tail; a duplicated tail stays blocked.
  - `action_mask(state) -> bool (N, S, 4)`: moves that don't hit a wall or a blocked cell. All False for dead snakes and finished games.

- [ ] **Step 1: Write the failing mask tests** — `tests/test_action_mask.py`

```python
from snake_rl.env.rules import GameState, action_mask, blocked_cells


def test_walls_and_neck_are_masked():
    state = GameState.from_layout(7, [[(0, 0), (1, 0), (2, 0)]])
    assert action_mask(state)[0, 0].tolist() == [False, False, False, True]


def test_vacating_tail_is_allowed():
    state = GameState.from_layout(7, [[(2, 2), (2, 3), (3, 3), (3, 2)]])
    assert action_mask(state)[0, 0, 1]  # down into the tail


def test_duplicated_tail_is_blocked():
    state = GameState.from_layout(7, [[(2, 2), (2, 3), (3, 3), (3, 2), (3, 2)]])
    assert not action_mask(state)[0, 0, 1]
    assert blocked_cells(state)[0, 3, 2]


def test_other_snakes_body_is_blocked_but_its_tail_is_not():
    state = GameState.from_layout(
        7, [[(2, 3), (1, 3), (0, 3)], [(3, 4), (3, 3), (3, 2)]]
    )
    mask = action_mask(state)
    assert not mask[0, 0, 1]  # (3, 3) is snake 1's body
    blocked = blocked_cells(state)
    assert blocked[0, 3, 4] and blocked[0, 3, 3] and not blocked[0, 3, 2]


def test_dead_snakes_and_finished_games_have_empty_masks():
    state = GameState.from_layout(
        7, [[(3, 3), (4, 3), (5, 3)], [(1, 1), (1, 2), (1, 3)]]
    )
    state.alive[0, 1] = False
    mask = action_mask(state)
    assert not mask[0, 1].any() and mask[0, 0].any()
    state.over[0] = True
    assert not action_mask(state).any()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_action_mask.py -v`
Expected: FAIL, `ImportError: cannot import name 'action_mask'`.

- [ ] **Step 3: Implement** — append to `snake_rl/env/rules.py`

```python
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
```

Replace `snake_rl/env/__init__.py` with:

```python
"""Vectorized Battlesnake environment."""

from snake_rl.env.rules import (
    DIRECTIONS,
    MAX_SNAKES,
    MIN_BOARD_SIZE,
    START_LENGTH,
    GameState,
    StepEvents,
    action_mask,
    blocked_cells,
    reset_games,
    start_points,
    step_games,
)
```

- [ ] **Step 4: Run the mask tests to verify they pass**

Run: `uv run pytest tests/test_action_mask.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the reference oracle** — `tests/reference_rules.py`

```python
"""Slow, readable Battlesnake rules for a single game, used to check the vectorized rules."""

from collections import Counter
from dataclasses import dataclass

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIN_FOOD = 1
SPAWN_CHANCE = 0.15


@dataclass
class RefSnake:
    body: list
    health: int
    alive: bool


@dataclass
class RefGame:
    board_size: int
    snakes: list
    food: set
    max_health: int
    turn: int
    over: bool


def from_state(state, env):
    snakes = []
    for k in range(state.num_snakes):
        length = int(state.length[env, k])
        segments = [tuple(int(v) for v in p) for p in state.body[env, k, :length]]
        snakes.append(RefSnake(segments, int(state.health[env, k]), bool(state.alive[env, k])))
    food = {(int(r), int(c)) for r, c in zip(*state.food[env].nonzero())}
    return RefGame(state.board_size, snakes, food, state.max_health,
                   int(state.turn[env]), bool(state.over[env]))


def ref_step(game, actions, spawn_u):
    if game.over:
        return
    b = game.board_size
    movers = [i for i, snake in enumerate(game.snakes) if snake.alive]

    for i in movers:
        snake = game.snakes[i]
        dr, dc = MOVES[int(actions[i])]
        head = (snake.body[0][0] + dr, snake.body[0][1] + dc)
        snake.body = [head] + snake.body[:-1]
        snake.health -= 1

    eaten = set()
    for i in movers:
        snake = game.snakes[i]
        if snake.body[0] in game.food:
            snake.health = game.max_health
            snake.body.append(snake.body[-1])
            eaten.add(snake.body[0])
    game.food -= eaten

    if len(game.food) < MIN_FOOD or spawn_u[0] < SPAWN_CHANCE:
        occupied = {seg for i in movers for seg in game.snakes[i].body}
        free = [(r, c) for r in range(b) for c in range(b)
                if (r, c) not in game.food and (r, c) not in occupied]
        if free:
            game.food.add(free[int(spawn_u[1] * len(free))])

    def on_board(p):
        return 0 <= p[0] < b and 0 <= p[1] < b

    survivors = [i for i in movers
                 if game.snakes[i].health > 0 and on_board(game.snakes[i].body[0])]
    eliminated = {i for i in movers if i not in survivors}
    body_cells = Counter(seg for i in survivors for seg in game.snakes[i].body[1:])
    for i in survivors:
        head = game.snakes[i].body[0]
        if head in body_cells:
            eliminated.add(i)
        for k in survivors:
            other = game.snakes[k]
            if k != i and other.body[0] == head and len(other.body) >= len(game.snakes[i].body):
                eliminated.add(i)
    for i in eliminated:
        game.snakes[i].alive = False

    game.turn += 1
    n_alive = sum(snake.alive for snake in game.snakes)
    game.over = n_alive == 0 if len(game.snakes) == 1 else n_alive <= 1
```

- [ ] **Step 6: Write the oracle and batch-consistency tests** — `tests/test_rules_reference.py`

```python
import numpy as np
import pytest

from reference_rules import from_state, ref_step
from snake_rl.env.rules import GameState, action_mask, reset_games, step_games


def assert_game_equal(state, env, game):
    assert int(state.turn[env]) == game.turn
    assert bool(state.over[env]) == game.over
    actual_food = {(int(r), int(c)) for r, c in zip(*state.food[env].nonzero())}
    assert actual_food == game.food
    for k, snake in enumerate(game.snakes):
        length = int(state.length[env, k])
        segments = [tuple(int(v) for v in p) for p in state.body[env, k, :length]]
        assert segments == snake.body, f"env {env} snake {k}"
        assert int(state.health[env, k]) == snake.health
        assert bool(state.alive[env, k]) == snake.alive


def random_actions(state, rng, safe_prob=0.9):
    mask = action_mask(state)
    safe = np.argmax(rng.random(mask.shape) + mask, axis=-1)
    wild = rng.integers(0, 4, size=safe.shape)
    return np.where(rng.random(safe.shape) < safe_prob, safe, wild)


@pytest.mark.parametrize("board_size,num_snakes", [(7, 1), (11, 2), (11, 4), (8, 3)])
def test_vectorized_rules_match_reference(board_size, num_snakes):
    rng = np.random.default_rng(board_size * 10 + num_snakes)
    state = GameState.empty(32, num_snakes, board_size)
    for _ in range(150):
        reset_games(state, state.over, rng)
        games = [from_state(state, env) for env in range(state.num_envs)]
        actions = random_actions(state, rng)
        spawn_u = rng.random((state.num_envs, 2))
        step_games(state, actions, spawn_u)
        for env, game in enumerate(games):
            ref_step(game, actions[env], spawn_u[env])
            assert_game_equal(state, env, game)


def test_batched_step_equals_individual_steps():
    rng = np.random.default_rng(7)
    state = GameState.empty(256, 2, 11)
    for _ in range(60):
        reset_games(state, state.over, rng)
        singles = [state.take([env]) for env in range(state.num_envs)]
        actions = random_actions(state, rng)
        spawn_u = rng.random((state.num_envs, 2))
        step_games(state, actions, spawn_u)
        for env, single in enumerate(singles):
            step_games(single, actions[env:env + 1], spawn_u[env:env + 1])
            game = from_state(single, 0)
            assert_game_equal(state, env, game)
```

`reference_rules` is importable because `tests/` has no `__init__.py`. pytest's default import mode then puts each test file's directory on `sys.path`. Do not add an `__init__.py` to `tests/`.

- [ ] **Step 7: Run all rules tests**

Run: `uv run pytest tests/test_rules_reference.py tests/test_action_mask.py tests/test_rules_step.py -v`
Expected: all PASS. If an oracle comparison fails, the assertion message names the env and snake. Find which rule differs by checking the step order in `step_games`. Do not change the oracle to match.

- [ ] **Step 8: Commit**

```bash
git add snake_rl/env tests/test_action_mask.py tests/reference_rules.py tests/test_rules_reference.py
git commit -m "Add action mask and check batched rules against a reference implementation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Observations

**Files:**
- Create: `snake_rl/env/observations.py`
- Modify: `snake_rl/env/__init__.py`
- Test: `tests/test_observations.py`

**Interfaces:**
- Consumes: `GameState`, `GameState.take`.
- Produces: `NUM_CHANNELS = 8`; channel index constants `OWN_HEAD, OWN_BODY, OWN_AGE, ENEMY_HEADS, ENEMY_BODIES, DANGEROUS_HEADS, FOOD, HEALTH` (0..7); `build_observations(state, envs=None) -> float32 (len(envs) or N, S, 8, B, B)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_observations.py`

```python
import numpy as np

from snake_rl.env.observations import (
    DANGEROUS_HEADS,
    ENEMY_BODIES,
    ENEMY_HEADS,
    FOOD,
    HEALTH,
    NUM_CHANNELS,
    OWN_AGE,
    OWN_BODY,
    OWN_HEAD,
    build_observations,
)
from snake_rl.env.rules import GameState, reset_games

SNAKE_A = [(3, 3), (4, 3), (5, 3)]
SNAKE_B = [(1, 1), (1, 2), (1, 3), (1, 4)]


def cells(plane):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(plane))}


def two_snake_state():
    return GameState.from_layout(7, [SNAKE_A, SNAKE_B], food=[(0, 6)], health=[50, 100])


def test_shape_and_dtype():
    state = GameState.empty(5, 3, 11)
    reset_games(state, np.ones(5, bool), np.random.default_rng(0))
    obs = build_observations(state)
    assert obs.shape == (5, 3, NUM_CHANNELS, 11, 11)
    assert obs.dtype == np.float32


def test_channels_from_first_snakes_view():
    obs = build_observations(two_snake_state())[0, 0]
    assert cells(obs[OWN_HEAD]) == {(3, 3)}
    assert cells(obs[OWN_BODY]) == {(4, 3), (5, 3)}
    assert obs[OWN_AGE, 3, 3] == 1.0
    assert np.isclose(obs[OWN_AGE, 4, 3], 2 / 3)
    assert np.isclose(obs[OWN_AGE, 5, 3], 1 / 3)
    assert cells(obs[ENEMY_HEADS]) == {(1, 1)}
    assert cells(obs[ENEMY_BODIES]) == {(1, 2), (1, 3), (1, 4)}
    assert cells(obs[DANGEROUS_HEADS]) == {(1, 1)}  # enemy is longer
    assert cells(obs[FOOD]) == {(0, 6)}
    assert np.all(obs[HEALTH] == 0.5)


def test_shorter_enemy_head_is_not_dangerous():
    obs = build_observations(two_snake_state())[0, 1]
    assert cells(obs[ENEMY_HEADS]) == {(3, 3)}
    assert not obs[DANGEROUS_HEADS].any()
    assert np.all(obs[HEALTH] == 1.0)


def test_observations_are_egocentric():
    swapped = GameState.from_layout(7, [SNAKE_B, SNAKE_A], food=[(0, 6)], health=[100, 50])
    original = build_observations(two_snake_state())
    flipped = build_observations(swapped)
    assert np.array_equal(original[0, 0], flipped[0, 1])
    assert np.array_equal(original[0, 1], flipped[0, 0])


def test_dead_snakes_are_invisible():
    state = two_snake_state()
    state.alive[0, 1] = False
    obs = build_observations(state)
    assert not obs[0, 0, ENEMY_HEADS].any() and not obs[0, 0, ENEMY_BODIES].any()
    assert not obs[0, 1, OWN_HEAD].any() and not obs[0, 1, HEALTH].any()


def test_subset_of_envs():
    state = GameState.empty(8, 2, 11)
    reset_games(state, np.ones(8, bool), np.random.default_rng(3))
    full = build_observations(state)
    assert np.array_equal(build_observations(state, np.array([2, 5])), full[[2, 5]])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_observations.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'snake_rl.env.observations'`.

- [ ] **Step 3: Implement** — `snake_rl/env/observations.py`

```python
"""Egocentric grid observations: each snake sees the board as 'self' versus 'enemies'."""

from __future__ import annotations

import numpy as np

from snake_rl.env.rules import GameState

NUM_CHANNELS = 8
(OWN_HEAD, OWN_BODY, OWN_AGE, ENEMY_HEADS, ENEMY_BODIES,
 DANGEROUS_HEADS, FOOD, HEALTH) = range(NUM_CHANNELS)


def _snake_layers(state: GameState):
    """Per-snake head, body and age planes, each float32 (N, S, B, B). Dead snakes are blank."""
    n, s, b = state.num_envs, state.num_snakes, state.board_size
    head = np.zeros((n, s, b, b), np.float32)
    body = np.zeros((n, s, b, b), np.float32)
    age = np.zeros((n, s, b, b), np.float32)
    seg = int(state.length.max())
    if seg == 0:
        return head, body, age
    valid = state.alive[:, :, None] & (np.arange(seg) < state.length[:, :, None])
    # Visit segments tail-first so that where segments share a cell (the stacked
    # start, a duplicated tail) the one nearest the head is written last.
    ns, ss, rev = np.nonzero(valid[:, :, ::-1])
    ii = seg - 1 - rev
    r, c = state.body[ns, ss, ii, 0], state.body[ns, ss, ii, 1]
    length = state.length[ns, ss].astype(np.float32)
    age[ns, ss, r, c] = (length - ii) / length
    is_head = ii == 0
    head[ns[is_head], ss[is_head], r[is_head], c[is_head]] = 1.0
    is_body = ~is_head
    body[ns[is_body], ss[is_body], r[is_body], c[is_body]] = 1.0
    return head, body, age


def build_observations(state: GameState, envs=None) -> np.ndarray:
    """Observation for every snake: float32 (N, S, NUM_CHANNELS, B, B)."""
    if envs is not None:
        state = state.take(envs)
    n, s, b = state.num_envs, state.num_snakes, state.board_size
    head, body, age = _snake_layers(state)
    enemy_heads = head.sum(axis=1, keepdims=True) - head
    enemy_bodies = body.sum(axis=1, keepdims=True) - body
    threat = (
        (state.length[:, None, :] >= state.length[:, :, None])
        & state.alive[:, None, :]
        & ~np.eye(s, dtype=bool)
    )  # (N, viewer, other)
    dangerous = np.einsum("nvk,nkhw->nvhw", threat.astype(np.float32), head)
    food = np.broadcast_to(state.food[:, None].astype(np.float32), (n, s, b, b))
    health = (state.health * state.alive / state.max_health).astype(np.float32)
    health = np.broadcast_to(health[:, :, None, None], (n, s, b, b))
    return np.stack(
        [head, body, age, enemy_heads, enemy_bodies, dangerous, food, health], axis=2
    )
```

Add to `snake_rl/env/__init__.py`, below the rules import:

```python
from snake_rl.env.observations import NUM_CHANNELS, build_observations
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_observations.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/env tests/test_observations.py
git commit -m "Add egocentric grid observations" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Rewards, SnakeEnv and the speed benchmark

**Files:**
- Create: `snake_rl/env/rewards.py`, `snake_rl/env/snake_env.py`, `scripts/benchmark_env.py`
- Modify: `snake_rl/env/__init__.py`
- Test: `tests/test_rewards.py`, `tests/test_env.py`

**Interfaces:**
- Consumes: `GameState`, `StepEvents`, `reset_games`, `step_games`, `action_mask`, `build_observations`.
- Produces:
  - `rewards.winloss(events) -> float32 (N,S)`, `rewards.food(events) -> float32 (N,S)`, `get_reward(name) -> callable`; `REWARDS` dict.
  - `SnakeEnv(num_envs, num_snakes, board_size=11, reward=None, max_health=100, seed=None)` with attributes `state: GameState`, properties `num_envs`, `num_snakes`, `board_size`, and methods:
    - `reset() -> (obs, info)`
    - `step(actions) -> (obs, rewards, dones, info)`
    - `info` keys after both: `action_mask (N,S,4)`, `length (N,S)`, `health (N,S)`, `alive (N,S)`, `turn (N,)`.
    - `step` also adds, describing the step *before* auto-reset: `game_over (N,)`, `winner (N,)`, `final_length (N,S)`, `final_turn (N,)`, `final_obs (N,S,C,B,B)`.

- [ ] **Step 1: Write the failing reward tests** — `tests/test_rewards.py`

```python
import numpy as np
import pytest

from snake_rl.env.rewards import food, get_reward, winloss
from snake_rl.env.rules import StepEvents


def events(ate, died, game_over, winner):
    died = np.array(died)
    return StepEvents(
        ate=np.array(ate), died=died, alive_before=np.ones_like(died),
        game_over=np.array(game_over), winner=np.array(winner),
    )


def test_winloss_win_and_loss():
    e = events(ate=[[False, False]], died=[[True, False]], game_over=[True], winner=[1])
    assert winloss(e).tolist() == [[-1.0, 1.0]]


def test_winloss_draw_gives_zero():
    e = events(ate=[[False, False]], died=[[True, True]], game_over=[True], winner=[-1])
    assert winloss(e).tolist() == [[0.0, 0.0]]


def test_winloss_mid_game_death():
    e = events(ate=[[False, False, False]], died=[[False, True, False]],
               game_over=[False], winner=[-1])
    assert winloss(e).tolist() == [[0.0, -1.0, 0.0]]


def test_food_reward():
    e = events(ate=[[True], [False], [True]], died=[[False], [True], [True]],
               game_over=[False, True, True], winner=[-1, -1, -1])
    assert food(e).tolist() == [[1.0], [-1.0], [0.0]]


def test_get_reward():
    assert get_reward("food") is food
    with pytest.raises(ValueError):
        get_reward("nope")
```

- [ ] **Step 2: Write the failing env tests** — `tests/test_env.py`

```python
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from snake_rl.env import SnakeEnv, build_observations
from snake_rl.env.observations import NUM_CHANNELS, OWN_HEAD
from snake_rl.env.rewards import food, winloss

ROOT = Path(__file__).resolve().parents[1]


def test_reset_returns_obs_and_info():
    env = SnakeEnv(4, 2, seed=0)
    obs, info = env.reset()
    assert obs.shape == (4, 2, NUM_CHANNELS, 11, 11)
    assert info["action_mask"].shape == (4, 2, 4)
    assert set(info) >= {"action_mask", "length", "health", "alive", "turn"}


def test_step_before_reset_raises():
    env = SnakeEnv(2, 1, seed=0)
    with pytest.raises(RuntimeError):
        env.step(np.zeros((2, 1), int))


def test_default_reward_depends_on_snake_count():
    assert SnakeEnv(1, 1).reward_fn is food
    assert SnakeEnv(1, 2).reward_fn is winloss


def test_unknown_reward_raises():
    with pytest.raises(ValueError):
        SnakeEnv(1, 1, reward="bogus")


def test_auto_reset_after_game_over():
    env = SnakeEnv(8, 1, seed=0)
    env.reset()
    seen = np.zeros(8, bool)
    for _ in range(12):  # moving up hits the top wall within 10 turns on 11x11
        obs, rewards, dones, info = env.step(np.zeros((8, 1), int))
        for e in np.flatnonzero(info["game_over"] & ~seen):
            assert dones[e, 0] and rewards[e, 0] == -1.0
            assert info["final_turn"][e] > 0 and info["turn"][e] == 0
            assert info["final_obs"][e, 0, OWN_HEAD].sum() == 0  # dead in the final frame
            assert obs[e, 0, OWN_HEAD].sum() == 1                # alive in the new game
            assert np.array_equal(obs[e], build_observations(env.state, [e])[0])
        seen |= info["game_over"]
    assert seen.all()


def test_same_seed_same_trajectory():
    def run():
        env = SnakeEnv(16, 2, seed=123)
        obs, info = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs, _, _, info = env.step(rng.integers(0, 4, size=(16, 2)))
        return obs

    assert np.array_equal(run(), run())


def test_two_snake_game_reports_winner():
    env = SnakeEnv(64, 2, seed=1)
    env.reset()
    rng = np.random.default_rng(1)
    winners = []
    for _ in range(200):
        _, rewards, _, info = env.step(rng.integers(0, 4, size=(64, 2)))
        for e in np.flatnonzero(info["game_over"]):
            w = info["winner"][e]
            winners.append(w)
            if w >= 0:
                assert rewards[e, w] == 1.0
    assert winners and set(winners) <= {-1, 0, 1}


def test_benchmark_script_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark_env.py"),
         "--num-envs", "8", "--steps", "5"],
        capture_output=True, text=True, check=True,
    )
    assert "env steps/sec" in result.stdout
```

- [ ] **Step 3: Run both test files to verify they fail**

Run: `uv run pytest tests/test_rewards.py tests/test_env.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'snake_rl.env.rewards'` and `ImportError: cannot import name 'SnakeEnv'`.

- [ ] **Step 4: Implement rewards** — `snake_rl/env/rewards.py`

```python
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
```

- [ ] **Step 5: Implement the env** — `snake_rl/env/snake_env.py`

```python
"""SnakeEnv: many Battlesnake games stepped together, with auto-reset."""

from __future__ import annotations

import numpy as np

from snake_rl.env.observations import build_observations
from snake_rl.env.rewards import get_reward
from snake_rl.env.rules import GameState, action_mask, reset_games, step_games


class SnakeEnv:
    def __init__(self, num_envs: int, num_snakes: int, board_size: int = 11,
                 reward: str | None = None, max_health: int = 100,
                 seed: int | None = None):
        self.state = GameState.empty(num_envs, num_snakes, board_size, max_health)
        self.reward_fn = get_reward(reward or ("winloss" if num_snakes > 1 else "food"))
        self.rng = np.random.default_rng(seed)
        self._needs_reset = True

    @property
    def num_envs(self) -> int:
        return self.state.num_envs

    @property
    def num_snakes(self) -> int:
        return self.state.num_snakes

    @property
    def board_size(self) -> int:
        return self.state.board_size

    def reset(self):
        reset_games(self.state, np.ones(self.num_envs, bool), self.rng)
        self._needs_reset = False
        return build_observations(self.state), self._info()

    def step(self, actions):
        """Advance every game one turn. Games that end are reset immediately; `info`
        keys prefixed `final_` (and `game_over`, `winner`) describe them before the reset.
        When no game ended, `obs` and `info["final_obs"]` are the same array."""
        if self._needs_reset:
            raise RuntimeError("call reset() before step()")
        spawn_u = self.rng.random((self.num_envs, 2))
        events = step_games(self.state, actions, spawn_u)
        rewards = self.reward_fn(events)
        dones = events.died | (events.game_over[:, None] & events.alive_before)
        obs = build_observations(self.state)
        info = {
            "game_over": events.game_over,
            "winner": events.winner,
            "final_length": self.state.length.copy(),
            "final_turn": self.state.turn.copy(),
            "final_obs": obs,
        }
        ended = np.flatnonzero(events.game_over)
        if ended.size:
            reset_games(self.state, events.game_over, self.rng)
            obs = obs.copy()
            obs[ended] = build_observations(self.state, ended)
        info.update(self._info())
        return obs, rewards, dones, info

    def _info(self) -> dict:
        return {
            "action_mask": action_mask(self.state),
            "length": self.state.length.copy(),
            "health": self.state.health.copy(),
            "alive": self.state.alive.copy(),
            "turn": self.state.turn.copy(),
        }
```

Add to `snake_rl/env/__init__.py`, below the observations import:

```python
from snake_rl.env.rewards import get_reward
from snake_rl.env.snake_env import SnakeEnv
```

- [ ] **Step 6: Write the benchmark script** — `scripts/benchmark_env.py`

```python
"""Measure SnakeEnv throughput (one env step = one turn of one game)."""

import argparse
import time

import numpy as np

from snake_rl.env import SnakeEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-snakes", type=int, default=2)
    parser.add_argument("--board", type=int, default=11)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    env = SnakeEnv(args.num_envs, args.num_snakes, args.board, seed=args.seed)
    _, info = env.reset()
    rng = np.random.default_rng(args.seed)

    def safe_random(mask):
        return np.argmax(rng.random(mask.shape) + mask, axis=-1)

    for _ in range(min(20, args.steps)):  # warm-up
        _, _, _, info = env.step(safe_random(info["action_mask"]))
    start = time.perf_counter()
    for _ in range(args.steps):
        _, _, _, info = env.step(safe_random(info["action_mask"]))
    elapsed = time.perf_counter() - start
    rate = args.steps * args.num_envs / elapsed
    print(f"{rate:,.0f} env steps/sec "
          f"(num_envs={args.num_envs}, num_snakes={args.num_snakes}, board={args.board})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_rewards.py tests/test_env.py -v`
Expected: all PASS.

- [ ] **Step 8: Check the speed target**

Run: `uv run python scripts/benchmark_env.py`
Expected: at least `200,000 env steps/sec`. If it is lower, profile:

```bash
uv run python -m cProfile -s cumtime scripts/benchmark_env.py --steps 100 | head -40
```

Optimize the hottest functions without changing behaviour; the full test suite must still pass. Likely candidates:
- reuse one `_occupancy` result where two calls see the same mask;
- in `build_observations`, write channels into a preallocated array instead of `np.stack`.

Record the final number; it goes in the README in Task 11.

- [ ] **Step 9: Commit**

```bash
git add snake_rl/env scripts/benchmark_env.py tests/test_rewards.py tests/test_env.py
git commit -m "Add SnakeEnv with auto-reset, reward functions and speed benchmark" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Simple baseline bots and the Hamiltonian bot

**Files:**
- Create: `snake_rl/baselines/__init__.py`, `snake_rl/baselines/simple.py`, `snake_rl/baselines/hamiltonian.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `GameState`, `action_mask`, `DIRECTIONS`, `SnakeEnv`.
- Produces: bot classes, each a callable `bot(state: GameState, snake: int) -> int64 (N,)` with a `name` attribute:
  - `RandomBot(seed=None)`, `SafeRandomBot(seed=None)`, `GreedyBot(seed=None)`, `HamiltonianBot(board_size)`.
  - `hamiltonian_cycle(board_size) -> list[tuple[int, int]]`.
  - `make_bot(name, seed=None, board_size=11)` and `BOT_NAMES` (tuple of names usable as seats in two-player games).

- [ ] **Step 1: Write the failing tests** — `tests/test_baselines.py`

```python
import numpy as np
import pytest

from snake_rl.baselines import (
    BOT_NAMES,
    GreedyBot,
    HamiltonianBot,
    RandomBot,
    SafeRandomBot,
    hamiltonian_cycle,
    make_bot,
)
from snake_rl.env import SnakeEnv
from snake_rl.env.rules import GameState, action_mask


def test_random_bot_returns_valid_actions():
    env = SnakeEnv(32, 2, seed=0)
    env.reset()
    actions = RandomBot(0)(env.state, 1)
    assert actions.shape == (32,)
    assert ((actions >= 0) & (actions <= 3)).all()


def test_safe_random_bot_respects_mask():
    env = SnakeEnv(32, 2, seed=0)
    env.reset()
    bot = SafeRandomBot(0)
    for _ in range(50):
        mask = action_mask(env.state)[:, 0]
        actions = bot(env.state, 0)
        has_safe = mask.any(axis=1)
        assert mask[np.arange(32), actions][has_safe].all()
        env.step(np.stack([actions, bot(env.state, 1)], axis=1))


def test_greedy_bot_heads_to_food():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 3)])
    assert GreedyBot(0)(state, 0)[0] == 0
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(3, 6)])
    assert GreedyBot(0)(state, 0)[0] == 3


def test_greedy_bot_avoids_immediate_death():
    # Food straight behind the neck: the only way to it is a fatal reversal.
    state = GameState.from_layout(7, [[(0, 3), (1, 3), (2, 3)]], food=[(6, 3)])
    action = GreedyBot(0)(state, 0)[0]
    assert action in (2, 3)


def test_hamiltonian_cycle_covers_board():
    cycle = hamiltonian_cycle(8)
    assert len(cycle) == 64 and len(set(cycle)) == 64
    for (r1, c1), (r2, c2) in zip(cycle, cycle[1:] + cycle[:1]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_hamiltonian_rejects_odd_board():
    with pytest.raises(ValueError):
        HamiltonianBot(11)


def test_hamiltonian_bot_fills_the_board():
    env = SnakeEnv(2, 1, board_size=8, max_health=64, seed=0)
    env.reset()
    bot = HamiltonianBot(8)
    final = np.zeros(2, int)
    done = np.zeros(2, bool)
    while not done.all():
        _, _, _, info = env.step(bot(env.state, 0)[:, None])
        new = info["game_over"] & ~done
        final[new] = info["final_length"][new, 0]
        done |= new
    assert (final == 64).all()


def test_make_bot():
    for name in BOT_NAMES:
        assert make_bot(name, seed=0).name == name
    assert make_bot("hamiltonian", board_size=8).name == "hamiltonian"


def test_make_bot_unknown_name():
    with pytest.raises(ValueError):
        make_bot("nope")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'snake_rl.baselines'`.

- [ ] **Step 3: Implement** — `snake_rl/baselines/simple.py`

```python
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
```

`snake_rl/baselines/hamiltonian.py`:

```python
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
```

`snake_rl/baselines/__init__.py`:

```python
"""Scripted baseline bots and a factory to build them by name."""

from snake_rl.baselines.hamiltonian import HamiltonianBot, hamiltonian_cycle
from snake_rl.baselines.simple import GreedyBot, RandomBot, SafeRandomBot

_BOTS = {
    "random": RandomBot,
    "safe_random": SafeRandomBot,
    "greedy": GreedyBot,
}
BOT_NAMES = tuple(_BOTS)


def make_bot(name: str, seed: int | None = None, board_size: int = 11):
    if name == "hamiltonian":
        return HamiltonianBot(board_size)
    if name not in _BOTS:
        raise ValueError(f"unknown bot {name!r}; choose from {sorted(_BOTS) + ['hamiltonian']}")
    return _BOTS[name](seed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/baselines tests/test_baselines.py
git commit -m "Add random, safe-random, greedy and Hamiltonian baseline bots" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Flood-fill heuristic bot

**Files:**
- Create: `snake_rl/baselines/flood_fill.py`
- Modify: `snake_rl/baselines/__init__.py`
- Test: `tests/test_flood_fill.py`

**Interfaces:**
- Consumes: `GameState`, `action_mask`, `blocked_cells`, `DIRECTIONS`, `SnakeEnv`, `make_bot` registry.
- Produces: `FloodFillBot(seed=None, hunger_threshold=30)`, `name = "flood_fill"`. Registered in `make_bot` and `BOT_NAMES`.

Decision rule per game, applied to the safe moves (from `action_mask`):
1. Drop moves into a cell that an equal-or-longer enemy head could also reach next turn, unless that leaves nothing.
2. Area = size of the free region (from `scipy.ndimage.label` on `~blocked_cells`) that the move enters. Keep moves whose area ≥ own length; if none, keep the moves with the largest area.
3. The snake is hungry if health ≤ `hunger_threshold`, or if any living enemy is at least as long. If hungry, pick the minimum (food distance, −area). Otherwise pick the minimum (−area, food distance).
4. Ties are broken randomly (candidates are shuffled before `min`).

- [ ] **Step 1: Write the failing tests** — `tests/test_flood_fill.py`

```python
import numpy as np

from snake_rl.baselines import BOT_NAMES, FloodFillBot, make_bot
from snake_rl.env import SnakeEnv
from snake_rl.env.rules import GameState, action_mask


def test_refuses_to_enter_a_dead_end_pocket_even_for_food():
    # Moving left enters a 2-cell pocket (0,0)-(0,1) holding food; right is open.
    state = GameState.from_layout(
        7, [[(0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0)]], food=[(0, 0)], health=[10]
    )
    assert FloodFillBot(0)(state, 0)[0] == 3


def test_avoids_head_to_head_with_a_longer_snake():
    state = GameState.from_layout(
        7,
        [[(3, 3), (4, 3), (5, 3)], [(1, 3), (1, 4), (1, 5), (1, 6)]],
        food=[(2, 3)],
        health=[10, 100],
    )
    for seed in range(5):
        assert FloodFillBot(seed)(state, 0)[0] in (2, 3)


def test_only_picks_safe_moves_when_available():
    env = SnakeEnv(16, 2, seed=0)
    env.reset()
    bot = FloodFillBot(0)
    for _ in range(40):
        mask = action_mask(env.state)
        actions = np.stack([bot(env.state, 0), bot(env.state, 1)], axis=1)
        for k in range(2):
            has_safe = mask[:, k].any(axis=1)
            assert mask[np.arange(16), k, actions[:, k]][has_safe].all()
        env.step(actions)


def test_registered():
    assert "flood_fill" in BOT_NAMES
    assert make_bot("flood_fill", seed=0).name == "flood_fill"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_flood_fill.py -v`
Expected: FAIL, `ImportError: cannot import name 'FloodFillBot'`.

- [ ] **Step 3: Implement** — `snake_rl/baselines/flood_fill.py`

```python
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
```

Replace `snake_rl/baselines/__init__.py` with:

```python
"""Scripted baseline bots and a factory to build them by name."""

from snake_rl.baselines.flood_fill import FloodFillBot
from snake_rl.baselines.hamiltonian import HamiltonianBot, hamiltonian_cycle
from snake_rl.baselines.simple import GreedyBot, RandomBot, SafeRandomBot

_BOTS = {
    "random": RandomBot,
    "safe_random": SafeRandomBot,
    "greedy": GreedyBot,
    "flood_fill": FloodFillBot,
}
BOT_NAMES = tuple(_BOTS)


def make_bot(name: str, seed: int | None = None, board_size: int = 11):
    if name == "hamiltonian":
        return HamiltonianBot(board_size)
    if name not in _BOTS:
        raise ValueError(f"unknown bot {name!r}; choose from {sorted(_BOTS) + ['hamiltonian']}")
    return _BOTS[name](seed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_flood_fill.py tests/test_baselines.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/baselines tests/test_flood_fill.py
git commit -m "Add flood-fill heuristic bot" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Evaluation harness, CLI, and the baseline ladder

**Files:**
- Create: `snake_rl/eval/__init__.py`, `snake_rl/eval/harness.py`, `scripts/evaluate.py`, `results/` JSON files
- Test: `tests/test_harness.py`, `tests/test_bot_ranking.py`

**Interfaces:**
- Consumes: `SnakeEnv`, `GameState`, `START_LENGTH`, `make_bot`, `BOT_NAMES`.
- Produces:
  - `Policy = Callable[[GameState, int], np.ndarray]`.
  - `wilson_interval(successes, n, z=1.96) -> tuple[float, float]`; `bootstrap_mean_ci(values, rng, n_resamples=2000) -> tuple[float, float]`.
  - `SeatResult(name, wins, draws, losses, win_rate, win_rate_ci, mean_length)`, `ScoreResult(mean, median, max, ci)`, `MatchResult(n_games, board_size, seed, mean_turns, seats, score)` with `to_dict()`, `to_json(path)`, `format_table()`.
  - `run_matches(policies, n_games, seed, board_size=11, max_health=100, names=None) -> MatchResult`.
  - Single-snake runs (`len(policies) == 1`) fill `score` (length − 3 = food eaten) and leave wins/draws/losses at 0.
  - In multi-snake runs, a seat gets a *draw* when the game ends with no winner and that seat died on the final turn. Seats that died earlier get a loss.

- [ ] **Step 1: Write the failing harness tests** — `tests/test_harness.py`

```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from snake_rl.baselines import GreedyBot, RandomBot
from snake_rl.eval import bootstrap_mean_ci, run_matches, wilson_interval

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_bootstrap_ci_brackets_mean():
    values = np.arange(100)
    lo, hi = bootstrap_mean_ci(values, np.random.default_rng(0))
    assert lo < values.mean() < hi


def test_two_player_counts_add_up_and_are_reproducible():
    def run():
        return run_matches([RandomBot(1), RandomBot(2)], n_games=50, seed=0)

    result = run()
    for seat in result.seats:
        assert seat.wins + seat.draws + seat.losses == 50
    assert result.seats[0].wins + result.seats[1].wins + result.seats[0].draws <= 50
    assert result.seats[0].draws == result.seats[1].draws
    assert result.score is None
    assert run().to_dict() == result.to_dict()


def test_single_snake_reports_score():
    result = run_matches([GreedyBot(0)], n_games=20, seed=0)
    score = result.score
    assert score is not None
    assert score.max >= score.median >= 0
    assert score.ci[0] <= score.mean <= score.ci[1]


def test_json_and_table(tmp_path):
    result = run_matches([GreedyBot(0), RandomBot(0)], n_games=10, seed=0)
    path = tmp_path / "result.json"
    result.to_json(path)
    data = json.loads(path.read_text())
    assert data["n_games"] == 10 and len(data["seats"]) == 2
    table = result.format_table()
    assert "greedy" in table and "random" in table


def test_evaluate_cli(tmp_path):
    out = tmp_path / "cli.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "greedy", "random",
         "--games", "10", "--seed", "0", "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(out.read_text())["seats"][0]["name"] == "greedy"


def test_evaluate_cli_rejects_unknown_bot():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "nope", "random"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "invalid choice" in result.stderr
```

- [ ] **Step 2: Write the failing ranking tests** — `tests/test_bot_ranking.py`

```python
import pytest

from snake_rl.baselines import make_bot
from snake_rl.eval import run_matches


@pytest.mark.parametrize(
    "stronger,weaker",
    [("safe_random", "random"), ("greedy", "safe_random"), ("flood_fill", "greedy")],
)
def test_baseline_ladder(stronger, weaker):
    result = run_matches(
        [make_bot(stronger, seed=1), make_bot(weaker, seed=2)], n_games=500, seed=0
    )
    assert result.seats[0].win_rate > 0.6, result.format_table()
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_harness.py tests/test_bot_ranking.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'snake_rl.eval'`.

- [ ] **Step 4: Implement** — `snake_rl/eval/harness.py`

```python
"""Play matches between policies and summarize results with confidence intervals."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from snake_rl.env import SnakeEnv
from snake_rl.env.rules import START_LENGTH, GameState

Policy = Callable[[GameState, int], np.ndarray]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values, rng: np.random.Generator,
                      n_resamples: int = 2000) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean."""
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(n_resamples, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


@dataclass
class SeatResult:
    name: str
    wins: int
    draws: int
    losses: int
    win_rate: float
    win_rate_ci: tuple[float, float]
    mean_length: float


@dataclass
class ScoreResult:
    mean: float
    median: float
    max: int
    ci: tuple[float, float]


@dataclass
class MatchResult:
    n_games: int
    board_size: int
    seed: int
    mean_turns: float
    seats: list[SeatResult]
    score: ScoreResult | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def format_table(self) -> str:
        lines = [f"{self.n_games} games on {self.board_size}x{self.board_size}, "
                 f"seed {self.seed}, mean game length {self.mean_turns:.1f} turns"]
        if self.score is not None:
            lo, hi = self.score.ci
            lines.append(f"{self.seats[0].name}: score mean {self.score.mean:.2f} "
                         f"(95% CI {lo:.2f}-{hi:.2f}), median {self.score.median:.1f}, "
                         f"max {self.score.max}")
            return "\n".join(lines)
        lines.append(f"{'seat':<14}{'wins':>6}{'draws':>7}{'losses':>8}"
                     f"{'win rate':>10}{'95% CI':>16}{'mean len':>10}")
        for seat in self.seats:
            lo, hi = seat.win_rate_ci
            ci = f"{lo:.3f}-{hi:.3f}"
            lines.append(f"{seat.name:<14}{seat.wins:>6}{seat.draws:>7}{seat.losses:>8}"
                         f"{seat.win_rate:>10.3f}{ci:>16}{seat.mean_length:>10.1f}")
        return "\n".join(lines)


def run_matches(policies: Sequence[Policy], n_games: int, seed: int,
                board_size: int = 11, max_health: int = 100,
                names: Sequence[str] | None = None) -> MatchResult:
    """Play n_games games, one per env, with policies[i] controlling snake i."""
    if not policies:
        raise ValueError("need at least one policy")
    if n_games < 1:
        raise ValueError(f"n_games must be >= 1, got {n_games}")
    s = len(policies)
    names = list(names) if names else [getattr(p, "name", f"seat{i}") for i, p in enumerate(policies)]
    env = SnakeEnv(n_games, s, board_size, max_health=max_health, seed=seed)
    env.reset()

    recorded = np.zeros(n_games, bool)
    winner = np.full(n_games, -1)
    drew = np.zeros((n_games, s), bool)
    final_length = np.zeros((n_games, s))
    turns = np.zeros(n_games)
    while not recorded.all():
        actions = np.stack([policy(env.state, i) for i, policy in enumerate(policies)], axis=1)
        _, _, dones, info = env.step(actions)
        new = info["game_over"] & ~recorded
        if new.any():
            winner[new] = info["winner"][new]
            drew[new] = dones[new] & (info["winner"][new] < 0)[:, None]
            final_length[new] = info["final_length"][new]
            turns[new] = info["final_turn"][new]
            recorded |= new

    seats = []
    for i, name in enumerate(names):
        if s == 1:
            wins = draws = losses = 0
        else:
            wins = int((winner == i).sum())
            draws = int(drew[:, i].sum())
            losses = n_games - wins - draws
        seats.append(SeatResult(name, wins, draws, losses, wins / n_games,
                                wilson_interval(wins, n_games),
                                float(final_length[:, i].mean())))

    score = None
    if s == 1:
        scores = final_length[:, 0] - START_LENGTH
        score = ScoreResult(float(scores.mean()), float(np.median(scores)), int(scores.max()),
                            bootstrap_mean_ci(scores, np.random.default_rng(seed)))
    return MatchResult(n_games, board_size, seed, float(turns.mean()), seats, score)
```

`snake_rl/eval/__init__.py`:

```python
"""Evaluation harness."""

from snake_rl.eval.harness import (
    MatchResult,
    Policy,
    ScoreResult,
    SeatResult,
    bootstrap_mean_ci,
    run_matches,
    wilson_interval,
)
```

`scripts/evaluate.py`:

```python
"""Play matches between baseline bots and report win rates or scores.

Examples:
  uv run python scripts/evaluate.py flood_fill greedy --games 500
  uv run python scripts/evaluate.py hamiltonian --board 8 --max-health 64 --games 20
"""

import argparse

from snake_rl.baselines import BOT_NAMES, make_bot
from snake_rl.eval import run_matches


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate baseline bots against each other.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("seats", nargs="+", choices=list(BOT_NAMES) + ["hamiltonian"],
                        help="one bot per snake; give one name for single-snake mode")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--board", type=int, default=11)
    parser.add_argument("--max-health", type=int, default=100)
    parser.add_argument("--out", help="write the result as JSON to this path")
    args = parser.parse_args(argv)

    policies = [make_bot(name, seed=args.seed + i + 1, board_size=args.board)
                for i, name in enumerate(args.seats)]
    result = run_matches(policies, args.games, args.seed, board_size=args.board,
                         max_health=args.max_health)
    print(result.format_table())
    if args.out:
        result.to_json(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the harness tests to verify they pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all PASS.

- [ ] **Step 6: Run the ranking tests**

Run: `uv run pytest tests/test_bot_ranking.py -v`
Expected: all three PASS (the flood-fill matchup may take ~30 s).

If one fails, read the printed table. Fix the *stronger bot's logic*, e.g. FloodFillBot's `hunger_threshold`, or a bug in its area or threat calculation. Do not lower the 0.6 threshold or change seeds to get a pass. If a fix isn't clear, stop and report the table.

- [ ] **Step 7: Save the baseline ladder results**

```bash
uv run python scripts/evaluate.py safe_random random --games 500 --out results/safe_random_vs_random.json
uv run python scripts/evaluate.py greedy safe_random --games 500 --out results/greedy_vs_safe_random.json
uv run python scripts/evaluate.py flood_fill greedy --games 500 --out results/flood_fill_vs_greedy.json
uv run python scripts/evaluate.py hamiltonian --board 8 --max-health 64 --games 20 --out results/hamiltonian_8x8.json
uv run python scripts/evaluate.py greedy --games 500 --out results/greedy_single.json
```

Expected: each prints a table; the first seat's win rate is above 0.6 in the three matchups, and the Hamiltonian score mean is 61 (64 cells − 3).

- [ ] **Step 8: Commit**

```bash
git add snake_rl/eval scripts/evaluate.py tests/test_harness.py tests/test_bot_ranking.py results/
git commit -m "Add evaluation harness, evaluate CLI and baseline ladder results" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Terminal renderer and human play

**Files:**
- Create: `snake_rl/env/render.py`, `scripts/play.py`
- Modify: `snake_rl/env/__init__.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `GameState`, `SnakeEnv`, `make_bot`, `BOT_NAMES`, `START_LENGTH`.
- Produces: `render(state, env=0, color=True) -> str`. The output has `1 + 2 + board_size + num_snakes` lines: a turn line, top border, rows, bottom border, then one status line per snake.

- [ ] **Step 1: Write the failing tests** — `tests/test_render.py`

```python
from snake_rl.env.render import render
from snake_rl.env.rules import GameState


def layout():
    return GameState.from_layout(
        7, [[(3, 3), (4, 3), (5, 3)], [(1, 1), (1, 2), (1, 3)]], food=[(0, 6)]
    )


def test_plain_render_layout():
    text = render(layout(), color=False)
    lines = text.splitlines()
    assert len(lines) == 1 + 2 + 7 + 2
    assert lines[0] == "Turn 0"
    assert "\033" not in text
    assert text.count("██") == 2 and text.count("▓▓") == 4 and text.count("◆◆") == 1
    assert "Snake 1: length 3, health 100" in text


def test_dead_snake_not_drawn():
    state = layout()
    state.alive[0, 1] = False
    text = render(state, color=False)
    assert text.count("██") == 1
    assert "(dead)" in text


def test_color_render_has_escape_codes():
    assert "\033[" in render(layout())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'snake_rl.env.render'`.

- [ ] **Step 3: Implement** — `snake_rl/env/render.py`

```python
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
```

Add to `snake_rl/env/__init__.py`:

```python
from snake_rl.env.render import render
```

`scripts/play.py`:

```python
"""Play Snake in the terminal with WASD, alone or against a bot.

  uv run python scripts/play.py
  uv run python scripts/play.py --opponent flood_fill
"""

import argparse
import sys
import termios
import tty

import numpy as np

from snake_rl.baselines import BOT_NAMES, make_bot
from snake_rl.env import START_LENGTH, SnakeEnv, render

KEYS = {"w": 0, "s": 1, "a": 2, "d": 3}


def read_key() -> str:
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:  # not a terminal
        return (input("move (w/a/s/d, q to quit): ").strip().lower() or "q")[0]
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play Snake with WASD.")
    parser.add_argument("--opponent", choices=BOT_NAMES, help="bot to play against")
    parser.add_argument("--board", type=int, default=11)
    parser.add_argument("--max-health", type=int, default=100)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    num_snakes = 2 if args.opponent else 1
    env = SnakeEnv(1, num_snakes, args.board, max_health=args.max_health, seed=args.seed)
    env.reset()
    bot = make_bot(args.opponent, seed=args.seed) if args.opponent else None
    action = 3
    while True:
        print("\033[2J\033[H" + render(env.state), flush=True)
        print("You are snake 0. WASD to move, Q to quit.")
        key = read_key()
        if key == "q":
            return
        action = KEYS.get(key, action)
        row = [action] + ([int(bot(env.state, 1)[0])] if bot else [])
        _, _, _, info = env.step(np.array([row]))
        if info["game_over"][0]:
            if bot is None:
                print(f"Game over. Score: {info['final_length'][0, 0] - START_LENGTH}")
            else:
                winner = info["winner"][0]
                print({0: "You win!", 1: f"{args.opponent} wins."}.get(winner, "Draw."))
            return


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_render.py -v`
Expected: all PASS.

- [ ] **Step 5: Manual check of human play**

Run in a real terminal: `uv run python scripts/play.py --opponent greedy`
Expected: a coloured 11×11 board with two snakes. Each WASD keypress advances one turn, Q quits, and the game prints a result line when it ends. Also run `uv run python scripts/play.py` for solo mode.

- [ ] **Step 6: Commit**

```bash
git add snake_rl/env scripts/play.py tests/test_render.py
git commit -m "Add terminal renderer and human play script" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: README, legacy removal, final verification

**Files:**
- Delete: `legacy/`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above; numbers from `results/*.json` and the Task 6 benchmark.

- [ ] **Step 1: Confirm nothing imports legacy code**

Run: `grep -rn "legacy\|snake_game\|dqn_agent\|colors import" snake_rl scripts tests`
Expected: no output.

- [ ] **Step 2: Delete legacy code**

```bash
git rm -r --quiet legacy
```

- [ ] **Step 3: Rewrite `README.md`**

Replace the whole file with the text below. Then substitute the four bracketed values with real numbers:
- `[RATE]` comes from `uv run python scripts/benchmark_env.py`;
- each `[WR …]` is the first seat's `win_rate` in the matching `results/*.json`, formatted as a percentage.

````markdown
# Snake RL

Multi-agent Snake as a reinforcement-learning testbed. The end goal is agents trained by
**self-play** against past versions of themselves, ranked by Elo and playable in the
browser. This repository is being built in stages:

1. **Foundation** (done): a fast, tested, batched Battlesnake environment, scripted
   baseline bots and an evaluation harness.
2. Algorithm benchmark: DQN through Rainbow components, and PPO, compared across seeds.
3. Self-play league with Elo tracking.
4. Browser demo with the trained agents.
5. Write-up of the results.

## Quick start

```bash
uv run pytest                                          # full test suite
uv run python scripts/play.py --opponent flood_fill    # play against a bot (WASD)
uv run python scripts/evaluate.py flood_fill greedy    # bot vs bot, 500 games
uv run python scripts/benchmark_env.py                 # environment throughput
```

## Environment

`SnakeEnv` steps many games at once, with every game's state held in NumPy arrays:

```python
from snake_rl.env import SnakeEnv

env = SnakeEnv(num_envs=1024, num_snakes=2)       # standard 11x11 Battlesnake
obs, info = env.reset()                           # obs: (1024, 2, 8, 11, 11)
obs, rewards, dones, info = env.step(actions)     # actions: (1024, 2) in {up, down, left, right}
```

- **Rules:** standard Battlesnake: simultaneous moves, health 100 that ticks down each turn,
  growth on eating, head-to-head collisions won by the longer snake, food spawning.
- **Observations:** egocentric 8-channel grids. These show own head, body and body age,
  enemy heads and bodies, dangerous enemy heads, food, and health. No hand-made features.
- **Speed:** about [RATE] env steps/sec on an Apple Silicon CPU (1024 games, 2 snakes).
- **Correctness:** each rule has a unit test; the batched rules are checked step-by-step
  against a slow reference implementation, and batched results match one-game-at-a-time
  results exactly.

## Baselines (500 games each, 11x11)

| Matchup | Win rate of first bot |
|---|---|
| flood_fill vs greedy | [WR flood_fill_vs_greedy] |
| greedy vs safe_random | [WR greedy_vs_safe_random] |
| safe_random vs random | [WR safe_random_vs_random] |

The Hamiltonian-cycle bot fills an 8x8 board every game, a reference point for perfect
single-snake play. Raw results, with 95% confidence intervals, are in `results/`.

## Layout

```
snake_rl/env/        rules, observations, rewards, SnakeEnv, renderer
snake_rl/baselines/  random, safe_random, greedy, flood_fill, hamiltonian bots
snake_rl/eval/       match runner with confidence intervals
scripts/             play, evaluate, benchmark
tests/               pytest suite (run in CI)
```
````

- [ ] **Step 4: Run the full verification**

```bash
uv run pytest -q
uv run python scripts/benchmark_env.py
grep -n "\[RATE\]\|\[WR" README.md
```

Expected:
- every test passes;
- the benchmark prints ≥ 200,000 env steps/sec;
- the grep prints nothing (all placeholders filled).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Rewrite README for the new direction and remove legacy code" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Check against the spec's Definition of done**

Confirm each item:
1. Tests pass locally; CI passes once pushed (pushing is the user's call).
2. Benchmark ≥ 200k steps/sec.
3. `results/` holds the ladder JSON.
4. `play.py` works solo and against a bot.
5. Package layout, `.gitignore` and README are done.
6. `legacy/` is gone.
