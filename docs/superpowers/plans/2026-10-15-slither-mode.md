# Slither Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a continuous, slither.io-style game mode to SnakeDQN, alongside the existing grid mode. It needs:

- a verified batched NumPy simulator;
- PPO self-play training;
- scripted bots and evaluation with confidence intervals;
- a browser component where visitors watch the trained network or play against it.

**Architecture:**

- **`snake_rl/slither/`:** a new package. Its core is a float64 NumPy engine whose snake
  bodies are ring buffers of points spaced 1 unit apart. On top of the engine sit a
  sector observation (263 floats), an 18-action interface, `SlitherEnv`, bots, an MLP and
  evaluation.
- **`tests/slither_reference.py`:** a slow, plain-Python reference that works on snapshot
  dicts. `web/slither/rules.js` is a line-by-line port of it. Parity tests tie all three
  implementations together.
- **Training:** `scripts/train_slither.py` follows the shape of `scripts/train_selfplay.py`.
- **Browser:** `web/slither/` mirrors the grid demo's component pattern
  (`mountSnakeDemo` → `mountSlitherDemo`).

**Tech Stack:** Python ≥3.11, NumPy (float64), PyTorch (MPS), ONNX / onnxruntime, Node 20 (parity tests), Canvas 2D, ONNX Runtime Web 1.20.1.

**Spec:** `docs/superpowers/specs/2026-10-15-slither-mode-design.md`. Read it before starting any task. Section references (§2, §3, …) below point into it.

## Global Constraints

- **Purely additive:** do not delete, rename or rewrite anything that already exists.
  Existing files may only gain:
  - one "Slither →" link in `web/index.html`;
  - a new README section.
  No existing test may be modified.
- **Dependencies:** core dependencies stay `numpy`, `scipy`. Training-only imports
  (`torch`, `onnx`, `onnxruntime`) live in scripts and in `snake_rl/slither/model.py`.
  Tests that need torch use `pytest.importorskip("torch")`.
- **Float64 only:** all rule and observation maths is float64. Observations are cast to
  float32 only on return. Never use `np.float32` inside the rules.
- **Operation order:** the batched engine, the reference and the JS port must use the
  same arithmetic expressions in the same order, e.g. `dx * dx + dy * dy < t * t` with
  `dx = point - head`, and `2.0 * pi * u`.
- **v1 constants:** exactly the `SlitherConfig` defaults in Task 1, which match spec §2.
  `ticks_per_decision = 3`, `max_snakes` = slots per arena (12 in training, 8 in
  evaluation).
- **Observation:** 263 floats = 32 sectors × 8 channels + 7 scalars. **Actions:** 18 =
  9 steering bins × {cruise, boost}.
- **Commands:** run tests with `uv run pytest …` and training scripts with
  `uv run --group train python scripts/…`.
- **Commits:** commit only files you changed. End every commit message with the line
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Maximum mass.** A snake at mass 500 whose body fills to 1,020 points and boosts must
   not overflow its 1,024-slot ring buffer or corrupt its body.
   (Test: Task 3, `test_max_mass_boosting_snake_stays_within_capacity`.)
2. **No live snakes.** When every snake is dead and one respawns, there are no live heads
   to measure clearance against. Respawn must pick candidate 0 without NaN or inf errors.
   (Test: Task 3, `test_respawn_with_no_live_heads_takes_first_candidate`.)
3. **Long lives.** Headings are never wrapped, so a long life can reach headings in the
   thousands of radians. Physics and observations must behave the same as at the
   equivalent small heading.
   (Test: Task 5, `test_observation_invariant_to_full_turns_of_heading`.)
4. **Config that breaks accounting.** A `respawn_delay` ≤ `ticks_per_decision` would let a
   snake die and respawn within one decision, which breaks the reward and `active`
   accounting. `SlitherEnv` must reject such a config.
   (Test: Task 6, `test_env_rejects_respawn_faster_than_a_decision`.)
5. **Bad actions.** Non-finite turns and out-of-range actions must raise a `ValueError`,
   never corrupt state.
   (Tests: Task 2, `test_step_tick_rejects_bad_inputs`; Task 6,
   `test_env_rejects_bad_actions`.)

---

## File map

| File | Responsibility | Task |
|---|---|---|
| `snake_rl/slither/__init__.py` | Package exports | 1 (grows later) |
| `snake_rl/slither/config.py` | `SlitherConfig`, `u_layout` | 1 |
| `snake_rl/slither/rules.py` | `SlitherState`, snapshots, `step_tick`, `reset_arenas`, `near_pairs` | 1–3, 7 |
| `tests/slither_helpers.py` | Scene builders for tests | 1 |
| `tests/test_slither_state.py` | State, snapshots, layout | 1 |
| `tests/test_slither_rules.py` | One test per rule | 2–3 |
| `tests/slither_reference.py` | Reference `step_tick` and `observe` on snapshots | 4–5 |
| `tests/test_slither_reference.py` | Batched == reference, batched == single | 4 |
| `snake_rl/slither/actions.py` | `NUM_ACTIONS`, `decode_actions`, `action_mask`, `steer_bin` | 5 |
| `snake_rl/slither/observations.py` | `build_observations`, `obs_size` | 5 |
| `tests/test_slither_observations.py` | Observation scenes and reference match | 5 |
| `snake_rl/slither/env.py` | `SlitherEnv` | 6 |
| `tests/test_slither_env.py` | Env API, rewards, masks | 6 |
| `scripts/benchmark_slither.py` | Decisions/sec | 7 |
| `tests/test_slither_pairs.py` | Grid pair search == dense (only if Task 7 needs it) | 7 |
| `snake_rl/slither/bots.py` | wander / forager / hunter | 8 |
| `tests/test_slither_bots.py` | Bot behaviour | 8 |
| `snake_rl/slither/model.py` | `SlitherNet`, `SlitherPolicy`, `masked_logits`, `load_slither_policy` | 9 |
| `snake_rl/slither/eval.py` | `evaluate_vs_bot`, `SlitherEvalResult` | 9 |
| `tests/test_slither_model.py`, `tests/test_slither_eval.py` | Model and evaluation tests | 9 |
| `scripts/train_slither.py`, `tests/test_train_slither.py` | PPO self-play and smoke test | 10 |
| `scripts/eval_slither.py`, `scripts/build_slither_assets.py`, `tests/test_slither_assets.py` | Final eval and web export | 11 |
| `web/slither/rules.js`, `web/slither/observe.js`, `web/slither/test/parity.mjs`, `tests/test_slither_js_parity.py` | JS port and parity | 12 |
| `web/slither/render.js`, `slither-demo.js`, `slither-demo.css`, `index.html`, `page.js`; one link in `web/index.html` | Browser component and page | 13 |
| `README.md` (new section), `results/slither_eval.json`, `web/slither/model.onnx`, `web/slither/stats.json` | Trained results | 14 |

---

### Task 1: Config, batched state and snapshots

**Files:**
- Create: `snake_rl/slither/__init__.py`, `snake_rl/slither/config.py`, `snake_rl/slither/rules.py`
- Create: `tests/slither_helpers.py`, `tests/test_slither_state.py`

**Interfaces:**
- Produces:
  - `SlitherConfig` (frozen dataclass, defaults below) with `.to_dict()`.
  - `u_layout(num_slots, cfg) -> {"respawn", "drops", "food", "size"}`.
  - `target_length(mass, cfg) -> int32 array`.
  - `body_radius(mass, cfg) -> float64 array`.
  - `SlitherState` with fields `cfg, body (N,S,CAP,2) f64, head (N,S) i32, length (N,S) i32,
    heading (N,S) f64, mass (N,S) f64, alive/enabled/boosting (N,S) bool,
    boost_ticks/respawn (N,S) i32, food_pos (N,F,2) f64, food_value (N,F) f64,
    food_active/food_natural (N,F) bool, tick (N,) i64`, plus the properties `num_envs`
    and `num_snakes`, and `SlitherState.empty(num_envs, num_snakes, cfg=None)`.
  - `ring_index(state) -> (N,S,CAP) int`.
  - `head_points(state) -> (N,S,2)`.
  - `logical_body(state, env, slot) -> (L,2)`.
  - `snapshot(state, env) -> dict`.
  - `from_snapshots(snaps, cfg=None) -> SlitherState`.
  - The test helpers `straight`, `snake`, `scene`, `state_of` and `QUIET`.
- **Snapshot format.** This is shared with the reference and the JS port:
  `{"slots": S, "tick": int, "snakes": [{"body": [[x, y], …] head first, "heading": float,
  "mass": float, "alive": bool, "enabled": bool, "boosting": bool, "boost_ticks": int,
  "respawn": int}], "food": [None | [x, y, value, natural], …] (length = food_capacity)}`.

- [ ] **Step 1: Write the failing tests**

`tests/slither_helpers.py`:

```python
"""Scene builders shared by the slither tests (snapshot dicts, see rules.snapshot)."""

import math
from dataclasses import replace

from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.rules import from_snapshots

# Unit-test config: no natural food, so scenes contain only the pellets a test places.
QUIET = replace(SlitherConfig(), natural_food_target=0)


def straight(x, y, heading, n):
    """n body points, head first, spaced 1 unit apart behind a head at (x, y)."""
    return [[x - i * math.cos(heading), y - i * math.sin(heading)] for i in range(n)]


def snake(body, heading=0.0, mass=10.0, alive=True, enabled=True, boosting=False,
          boost_ticks=0, respawn=0):
    return {"body": [list(p) for p in body], "heading": heading, "mass": mass, "alive": alive,
            "enabled": enabled, "boosting": boosting, "boost_ticks": boost_ticks,
            "respawn": respawn}


def dead(respawn=0, enabled=True, mass=10.0):
    return snake([], alive=False, enabled=enabled, respawn=respawn, mass=mass)


def scene(snakes, food=(), cfg=QUIET, tick=0):
    """food: (x, y, value) or (x, y, value, natural) tuples, placed in slots 0, 1, ..."""
    pellets = [None] * cfg.food_capacity
    for i, item in enumerate(food):
        x, y, v, *nat = item
        pellets[i] = [x, y, v, bool(nat[0]) if nat else False]
    return {"slots": len(snakes), "tick": tick, "snakes": snakes, "food": pellets}


def state_of(*scenes, cfg=QUIET):
    return from_snapshots(list(scenes), cfg)
```

`tests/test_slither_state.py`:

```python
import math

import numpy as np
import pytest

from slither_helpers import QUIET, scene, snake, state_of, straight
from snake_rl.slither.config import SlitherConfig, u_layout
from snake_rl.slither.rules import (SlitherState, body_radius, head_points, logical_body,
                                    ring_index, snapshot, target_length)


def test_empty_state_shapes_and_dtypes():
    cfg = SlitherConfig()
    s = SlitherState.empty(3, 12, cfg)
    assert s.body.shape == (3, 12, cfg.body_capacity, 2) and s.body.dtype == np.float64
    assert s.food_pos.shape == (3, cfg.food_capacity, 2)
    assert s.num_envs == 3 and s.num_snakes == 12
    assert not s.alive.any() and not s.enabled.any() and not s.food_active.any()


@pytest.mark.parametrize("n,k", [(0, 4), (2, 0), (2, 17)])
def test_empty_rejects_bad_sizes(n, k):
    with pytest.raises(ValueError):
        SlitherState.empty(n, k)


def test_length_and_radius_formulas():
    cfg = SlitherConfig()
    assert int(target_length(10.0, cfg)) == 40
    assert int(target_length(500.0, cfg)) == 1020 <= cfg.body_capacity - 2
    assert float(body_radius(16.0, cfg)) == 4.0


def test_u_layout_for_twelve_slots():
    assert u_layout(12, SlitherConfig()) == {"respawn": 0, "drops": 396, "food": 1932, "size": 1936}


def test_snapshot_round_trip():
    a = scene([snake(straight(5.0, -3.0, 0.3, 40), heading=0.3, mass=12.5, boosting=True,
                     boost_ticks=2),
               snake([], alive=False, respawn=7)],
              food=[(1.0, 2.0, 1.0, True), (3.0, 4.0, 0.8)])
    b = scene([snake(straight(-50.0, 10.0, 2.0, 41), heading=2.0, mass=10.4),
               snake([], alive=False, enabled=False)])
    state = state_of(a, b)
    assert snapshot(state, 0) == a
    assert snapshot(state, 1) == b


def test_logical_body_wraps_the_ring():
    cfg = QUIET
    state = state_of(scene([snake(straight(0.0, 0.0, 0.0, 5))]))
    pts = logical_body(state, 0, 0).copy()
    # Move the ring so the head sits at index 1: logical points 2.. wrap to the end.
    rolled = np.roll(state.body[0, 0], 1 - int(state.head[0, 0]), axis=0)
    state.body[0, 0] = rolled
    state.head[0, 0] = 1
    np.testing.assert_array_equal(logical_body(state, 0, 0), pts)
    lidx = ring_index(state)[0, 0]
    assert lidx[1] == 0 and lidx[0] == 1 and lidx[cfg.body_capacity - 1] == 2
    np.testing.assert_array_equal(head_points(state)[0, 0], [0.0, 0.0])


def test_from_snapshots_rejects_wrong_food_length():
    bad = scene([snake(straight(0, 0, 0, 3))])
    bad["food"] = bad["food"][:-1]
    with pytest.raises(ValueError):
        state_of(bad)
```

`tests/` has no `__init__.py`, and `pyproject.toml` sets `testpaths = ["tests"]`, so
pytest's rootdir-based `sys.path` insertion makes `slither_helpers` importable. The
existing `tests/reference_rules.py` relies on the same thing.

- [ ] **Step 2: Run the tests to check they fail**

Run: `uv run pytest tests/test_slither_state.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'snake_rl.slither'`

- [ ] **Step 3: Implement**

`snake_rl/slither/__init__.py`:

```python
"""Slither mode: a continuous, slither.io-style arena (spec 2026-10-15-slither-mode-design)."""

from snake_rl.slither.config import SlitherConfig, u_layout

__all__ = ["SlitherConfig", "u_layout"]
```

`snake_rl/slither/config.py`:

```python
"""Every constant of the slither game in one place. Units: world units and physics ticks."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SlitherConfig:
    arena_radius: float = 300.0
    tick_hz: int = 30
    ticks_per_decision: int = 3
    max_turn: float = 0.15            # radians per tick
    start_mass: float = 10.0
    max_mass: float = 500.0
    base_length: float = 20.0         # body points = floor(base_length + length_per_mass * mass)
    length_per_mass: float = 2.0
    base_radius: float = 2.0          # radius = base_radius + radius_per_sqrt_mass * sqrt(mass)
    radius_per_sqrt_mass: float = 0.5
    body_capacity: int = 1024         # ring buffer size; must exceed the longest body + 2
    boost_min_mass: float = 15.0
    boost_cost: float = 0.2           # mass per boosting tick
    boost_drop_every: int = 5         # ticks of continuous boost per dropped pellet
    boost_drop_value: float = 0.8
    eat_margin: float = 3.0
    food_capacity: int = 1024
    natural_food_target: int = 200
    natural_food_per_tick: int = 2
    natural_food_value: float = 1.0
    food_spawn_margin: float = 10.0
    death_drop_fraction: float = 0.8
    death_drop_max_pellets: int = 64
    death_drop_jitter: float = 2.0
    respawn_delay: int = 30
    respawn_candidates: int = 16
    respawn_margin: float = 60.0
    respawn_clearance: float = 40.0
    vision: float = 60.0
    sectors: int = 32
    neck: int = 10                    # own-body points closer to the head than this are not observed

    def to_dict(self) -> dict:
        return asdict(self)


def u_layout(num_slots: int, cfg: SlitherConfig) -> dict[str, int]:
    """Offsets into one arena's row of the per-tick uniform array `u`:
    [respawn: per slot, `respawn_candidates` (radius, angle) pairs then a heading]
    [drops:   per slot, `death_drop_max_pellets` (x, y) jitter pairs]
    [food:    `natural_food_per_tick` (radius, angle) pairs]"""
    drops = num_slots * (2 * cfg.respawn_candidates + 1)
    food = drops + num_slots * cfg.death_drop_max_pellets * 2
    return {"respawn": 0, "drops": drops, "food": food,
            "size": food + 2 * cfg.natural_food_per_tick}
```

`snake_rl/slither/rules.py` (state part; Tasks 2–3 add the tick):

```python
"""Batched slither rules in pure NumPy float64 (spec §2).

Arrays are indexed [arena, slot, ...]. Each body is a ring buffer of points spaced one
cruise step (1 unit) apart: `head` is the ring index of the head point, and logical point
i (0 = head) lives at ring index (head - i) % body_capacity. tests/slither_reference.py
is the slow, obviously-correct version of step_tick; web/slither/rules.js ports it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from snake_rl.slither.config import SlitherConfig, u_layout

MAX_SLOTS = 16
CHUNK = 1 << 17  # rows per block in point-vs-head passes, to bound temporary memory


def target_length(mass, cfg: SlitherConfig) -> np.ndarray:
    return np.floor(cfg.base_length + cfg.length_per_mass * np.asarray(mass, np.float64)).astype(np.int32)


def body_radius(mass, cfg: SlitherConfig) -> np.ndarray:
    return cfg.base_radius + cfg.radius_per_sqrt_mass * np.sqrt(np.asarray(mass, np.float64))


@dataclass
class SlitherState:
    cfg: SlitherConfig
    body: np.ndarray          # float64 (N, S, CAP, 2) ring buffer of body points
    head: np.ndarray          # int32 (N, S) ring index of the head point
    length: np.ndarray        # int32 (N, S) points in the body; 0 while dead
    heading: np.ndarray       # float64 (N, S) radians, never wrapped
    mass: np.ndarray          # float64 (N, S); kept while dead, reset on respawn
    alive: np.ndarray         # bool (N, S)
    enabled: np.ndarray       # bool (N, S) slot takes part in its arena
    boosting: np.ndarray      # bool (N, S) boosted on the last tick
    boost_ticks: np.ndarray   # int32 (N, S) consecutive boosting ticks
    respawn: np.ndarray       # int32 (N, S) ticks until a dead enabled slot respawns
    food_pos: np.ndarray      # float64 (N, F, 2)
    food_value: np.ndarray    # float64 (N, F)
    food_active: np.ndarray   # bool (N, F)
    food_natural: np.ndarray  # bool (N, F) natural pellet (counts toward the food target)
    tick: np.ndarray          # int64 (N,)

    @property
    def num_envs(self) -> int:
        return self.body.shape[0]

    @property
    def num_snakes(self) -> int:
        return self.body.shape[1]

    @classmethod
    def empty(cls, num_envs: int, num_snakes: int, cfg: SlitherConfig | None = None) -> SlitherState:
        cfg = cfg or SlitherConfig()
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if not 1 <= num_snakes <= MAX_SLOTS:
            raise ValueError(f"num_snakes must be in 1..{MAX_SLOTS}, got {num_snakes}")
        if int(target_length(cfg.max_mass, cfg)) + 2 > cfg.body_capacity:
            raise ValueError("body_capacity is too small for max_mass")
        n, s, cap, f = num_envs, num_snakes, cfg.body_capacity, cfg.food_capacity
        return cls(
            cfg=cfg,
            body=np.zeros((n, s, cap, 2)), head=np.zeros((n, s), np.int32),
            length=np.zeros((n, s), np.int32), heading=np.zeros((n, s)), mass=np.zeros((n, s)),
            alive=np.zeros((n, s), bool), enabled=np.zeros((n, s), bool),
            boosting=np.zeros((n, s), bool), boost_ticks=np.zeros((n, s), np.int32),
            respawn=np.zeros((n, s), np.int32),
            food_pos=np.zeros((n, f, 2)), food_value=np.zeros((n, f)),
            food_active=np.zeros((n, f), bool), food_natural=np.zeros((n, f), bool),
            tick=np.zeros(n, np.int64),
        )


def ring_index(state: SlitherState) -> np.ndarray:
    """(N, S, CAP) logical index of the point stored in each ring slot."""
    cap = state.cfg.body_capacity
    return (state.head[..., None].astype(np.int64) - np.arange(cap)) % cap


def head_points(state: SlitherState) -> np.ndarray:
    """(N, S, 2) head positions (meaningless for dead slots)."""
    n, s = state.num_envs, state.num_snakes
    return state.body[np.arange(n)[:, None], np.arange(s)[None, :], state.head]


def logical_body(state: SlitherState, env: int, slot: int) -> np.ndarray:
    """(L, 2) body points of one snake, head first."""
    length = int(state.length[env, slot])
    idx = (int(state.head[env, slot]) - np.arange(length)) % state.cfg.body_capacity
    return state.body[env, slot, idx]


def snapshot(state: SlitherState, env: int) -> dict:
    """One arena as plain Python data (the format tests/slither_reference.py and
    web/slither/rules.js work on)."""
    snakes = []
    for k in range(state.num_snakes):
        snakes.append({
            "body": logical_body(state, env, k).tolist(),
            "heading": float(state.heading[env, k]),
            "mass": float(state.mass[env, k]),
            "alive": bool(state.alive[env, k]),
            "enabled": bool(state.enabled[env, k]),
            "boosting": bool(state.boosting[env, k]),
            "boost_ticks": int(state.boost_ticks[env, k]),
            "respawn": int(state.respawn[env, k]),
        })
    food = [None] * state.cfg.food_capacity
    for f in np.flatnonzero(state.food_active[env]):
        x, y = state.food_pos[env, f]
        food[f] = [float(x), float(y), float(state.food_value[env, f]),
                   bool(state.food_natural[env, f])]
    return {"slots": state.num_snakes, "tick": int(state.tick[env]), "snakes": snakes, "food": food}


def from_snapshots(snaps: list[dict], cfg: SlitherConfig | None = None) -> SlitherState:
    """Batch arenas given as snapshots. Logical point i is stored at ring index L-1-i."""
    cfg = cfg or SlitherConfig()
    slots = snaps[0]["slots"]
    state = SlitherState.empty(len(snaps), slots, cfg)
    for e, snap in enumerate(snaps):
        if snap["slots"] != slots or len(snap["snakes"]) != slots:
            raise ValueError("every snapshot needs the same number of slots")
        if len(snap["food"]) != cfg.food_capacity:
            raise ValueError(f"food must have {cfg.food_capacity} slots, got {len(snap['food'])}")
        state.tick[e] = snap["tick"]
        for k, s in enumerate(snap["snakes"]):
            pts = np.asarray(s["body"], np.float64).reshape(-1, 2)
            length = len(pts)
            if length > cfg.body_capacity - 2:
                raise ValueError("body longer than the ring buffer allows")
            if length:
                state.body[e, k, :length] = pts[::-1]
                state.head[e, k] = length - 1
            state.length[e, k] = length
            state.heading[e, k] = s["heading"]
            state.mass[e, k] = s["mass"]
            state.alive[e, k] = s["alive"]
            state.enabled[e, k] = s["enabled"]
            state.boosting[e, k] = s["boosting"]
            state.boost_ticks[e, k] = s["boost_ticks"]
            state.respawn[e, k] = s["respawn"]
        for f, item in enumerate(snap["food"]):
            if item is not None:
                state.food_pos[e, f] = item[:2]
                state.food_value[e, f] = item[2]
                state.food_natural[e, f] = item[3]
                state.food_active[e, f] = True
    return state
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `uv run pytest tests/test_slither_state.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/slither tests/slither_helpers.py tests/test_slither_state.py
git commit -m "Slither: config, batched state and snapshots

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Tick physics: steering, movement, boost, collisions

**Files:**
- Modify: `snake_rl/slither/rules.py` (append)
- Create: `tests/test_slither_rules.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `TickEvents(died bool (N,S), killer int32 (N,S) [-1 = none], kills int32 (N,S), eaten float64 (N,S))`.
  - `step_tick(state, turn (N,S) float, boost (N,S) bool, u (N, u_layout(S)["size"])) -> TickEvents`.
    It mutates `state`. This task implements spec §2 steps 1–3; Task 3 completes steps 4–7
    and replaces `step_tick`'s body.
  - `near_pairs(pe, px, py, heads, live, reach) -> (point_idx, slot)`: candidate
    point/head pairs within `reach`. Task 5 reuses it and Task 7 may replace its body.

- [ ] **Step 1: Write the failing tests**

`tests/test_slither_rules.py`:

```python
import math

import numpy as np
import pytest

from slither_helpers import QUIET, dead, scene, snake, state_of, straight
from snake_rl.slither.config import u_layout
from snake_rl.slither.rules import head_points, logical_body, snapshot, step_tick

R0 = 2.0 + 0.5 * math.sqrt(10.0)  # radius at mass 10


def tick(state, turn=0.0, boost=False, u=0.5):
    n, s = state.num_envs, state.num_snakes
    turn = np.broadcast_to(np.asarray(turn, float), (n, s)).copy()
    boost = np.broadcast_to(np.asarray(boost, bool), (n, s)).copy()
    size = u_layout(s, state.cfg)["size"]
    u = np.broadcast_to(np.asarray(u, float), (n, size)).copy()
    return step_tick(state, turn, boost, u)


def test_straight_move_pushes_one_point_and_keeps_length():
    state = state_of(scene([snake(straight(0.0, 0.0, 0.0, 40))]))
    ev = tick(state)
    body = logical_body(state, 0, 0)
    np.testing.assert_allclose(body[0], [1.0, 0.0])
    np.testing.assert_allclose(body[1], [0.0, 0.0])
    assert len(body) == 40 and not ev.died.any()
    assert int(state.tick[0]) == 1


def test_turn_is_capped():
    state = state_of(scene([snake(straight(0, 0, 0, 40)), snake(straight(0, 50, 0, 40))]))
    tick(state, turn=np.array([[1.0, -0.05]]))
    np.testing.assert_allclose(state.heading[0], [0.15, -0.05])
    np.testing.assert_allclose(head_points(state)[0, 0], [math.cos(0.15), math.sin(0.15)])


def test_boost_moves_two_points_costs_mass_and_counts_ticks():
    state = state_of(scene([snake(straight(0, 0, 0, 40), mass=20.0)]))
    tick(state, boost=True)
    assert state.boosting[0, 0] and state.boost_ticks[0, 0] == 1
    assert state.mass[0, 0] == pytest.approx(19.8)
    np.testing.assert_allclose(head_points(state)[0, 0], [2.0, 0.0])
    assert state.length[0, 0] == 42  # grows by the points pushed, target is 59


def test_boost_needs_minimum_mass():
    state = state_of(scene([snake(straight(0, 0, 0, 40), mass=14.0)]))
    tick(state, boost=True)
    assert not state.boosting[0, 0] and state.mass[0, 0] == 14.0
    np.testing.assert_allclose(head_points(state)[0, 0], [1.0, 0.0])


def test_long_body_is_cut_to_target_length():
    state = state_of(scene([snake(straight(0, 0, 0, 60), mass=10.0)]))
    tick(state)
    assert state.length[0, 0] == 40


def test_growth_is_one_point_per_tick():
    state = state_of(scene([snake(straight(0, 0, 0, 40), mass=20.0)]))
    tick(state)
    assert state.length[0, 0] == 41


def test_wall_kills():
    state = state_of(scene([snake(straight(296.0, 0.0, 0.0, 40))]))
    ev = tick(state)
    assert ev.died[0, 0] and ev.killer[0, 0] == -1 and not state.alive[0, 0]


def test_head_into_body_kills_and_credits_the_owner():
    a = snake(straight(0.0, 0.0, 0.0, 40))
    b = snake(straight(3.0, 20.0, math.pi / 2, 40), heading=math.pi / 2)
    state = state_of(scene([a, b]))
    ev = tick(state)
    assert ev.died[0].tolist() == [True, False]
    assert ev.killer[0].tolist() == [1, -1]
    assert ev.kills[0].tolist() == [0, 1]


def test_head_on_kills_both_without_credit():
    a = snake(straight(0.0, 0.0, 0.0, 40))
    b = snake(straight(6.0, 0.0, math.pi, 40), heading=math.pi)
    state = state_of(scene([a, b]))
    ev = tick(state)
    assert ev.died[0].all() and (ev.killer[0] == -1).all() and (ev.kills[0] == 0).all()


def test_no_self_collision():
    curl = [[0, 0], [0, -1], [0, -2], [1, -2], [2, -2], [2, -1], [2, 0], [3, 0], [4, 0]]
    state = state_of(scene([snake(curl, mass=10.0)]))
    ev = tick(state)
    assert not ev.died.any()


def test_snake_dying_this_tick_still_kills():
    b = snake(straight(296.0, 0.0, 0.0, 40))                   # hits the wall
    a = snake(straight(270.0, 6.0, -math.pi / 2, 40), heading=-math.pi / 2)  # into b's body
    state = state_of(scene([a, b]))
    ev = tick(state)
    assert ev.died[0].all() and ev.killer[0].tolist() == [1, -1]


def test_dead_snakes_are_ignored():
    state = state_of(scene([snake(straight(0, 0, 0, 40)), dead(respawn=100)]))
    ev = tick(state, turn=0.1, boost=True)
    assert not state.alive[0, 1] and state.length[0, 1] == 0 and not ev.died[0, 1]


def test_step_tick_rejects_bad_inputs():
    state = state_of(scene([snake(straight(0, 0, 0, 40))]))
    size = u_layout(1, QUIET)["size"]
    ok_u = np.full((1, size), 0.5)
    with pytest.raises(ValueError):
        step_tick(state, np.array([[np.nan]]), np.array([[False]]), ok_u)
    with pytest.raises(ValueError):
        step_tick(state, np.zeros((1, 2)), np.zeros((1, 2), bool), ok_u)
    with pytest.raises(ValueError):
        step_tick(state, np.zeros((1, 1)), np.zeros((1, 1), bool), ok_u[:, :-1])
    assert snapshot(state, 0)["tick"] == 0  # nothing was applied
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `uv run pytest tests/test_slither_rules.py -q`
Expected: FAIL with `ImportError: cannot import name 'step_tick'`

- [ ] **Step 3: Implement (append to `rules.py`)**

```python
@dataclass
class TickEvents:
    died: np.ndarray    # bool (N, S) died this tick
    killer: np.ndarray  # int32 (N, S) slot whose body it hit; -1 for wall/head-on deaths
    kills: np.ndarray   # int32 (N, S) kills credited this tick
    eaten: np.ndarray   # float64 (N, S) pellet value eaten this tick


def step_tick(state: SlitherState, turn, boost, u) -> TickEvents:
    """Advance every arena by one physics tick (spec §2 steps 1-3; Task 3 adds 4-7)."""
    cfg = state.cfg
    n, s = state.num_envs, state.num_snakes
    turn, boost, u = _validate(state, turn, boost, u)
    alive = state.alive.copy()

    # 1. steer
    state.heading = np.where(alive, state.heading + np.clip(turn, -cfg.max_turn, cfg.max_turn),
                             state.heading)
    # 2. boost, move, trim
    boosting = alive & boost & (state.mass >= cfg.boost_min_mass)
    state.mass = np.where(boosting, state.mass - cfg.boost_cost, state.mass)
    state.boost_ticks = np.where(boosting, state.boost_ticks + 1, 0).astype(np.int32)
    state.boosting = boosting
    _push(state, alive)
    _push(state, boosting)
    pushed = alive.astype(np.int32) + boosting
    state.length = np.where(alive, np.minimum(state.length + pushed, target_length(state.mass, cfg)),
                            state.length).astype(np.int32)
    # 3. deaths
    radius = body_radius(state.mass, cfg)
    heads = head_points(state)
    hx, hy = heads[..., 0], heads[..., 1]
    died = alive & (np.sqrt(hx * hx + hy * hy) + radius > cfg.arena_radius)
    head_on = _head_on(hx, hy, radius, alive)
    first = _body_hits(state, heads, radius, alive)
    died |= head_on | (first < s)
    killer = np.where(died & (first < s) & ~head_on, first, -1).astype(np.int32)
    kills = np.zeros((n, s), np.int32)
    ke, ki = np.nonzero(killer >= 0)
    np.add.at(kills, (ke, killer[ke, ki]), 1)
    state.alive = alive & ~died
    state.length[died] = 0
    state.tick += 1
    return TickEvents(died, killer, kills, np.zeros((n, s)))


def _validate(state, turn, boost, u):
    shape = (state.num_envs, state.num_snakes)
    turn = np.asarray(turn, np.float64)
    boost = np.asarray(boost, bool)
    u = np.asarray(u, np.float64)
    if turn.shape != shape or boost.shape != shape:
        raise ValueError(f"turn and boost must have shape {shape}, got {turn.shape} and {boost.shape}")
    size = u_layout(state.num_snakes, state.cfg)["size"]
    if u.shape != (state.num_envs, size):
        raise ValueError(f"u must have shape {(state.num_envs, size)}, got {u.shape}")
    if not np.isfinite(turn).all():
        raise ValueError("turn must be finite")
    return turn, boost, u


def _push(state, mask):
    """Add a point 1 unit ahead of the head, along the heading, for every snake in `mask`."""
    e, k = np.nonzero(mask)
    old = state.head[e, k]
    new = (old + 1) % state.cfg.body_capacity
    h = state.heading[e, k]
    state.body[e, k, new, 0] = state.body[e, k, old, 0] + np.cos(h)
    state.body[e, k, new, 1] = state.body[e, k, old, 1] + np.sin(h)
    state.head[e, k] = new


def _head_on(hx, hy, radius, alive):
    s = hx.shape[1]
    out = np.zeros_like(alive)
    for i in range(s):
        for j in range(i + 1, s):
            dx = hx[:, i] - hx[:, j]
            dy = hy[:, i] - hy[:, j]
            t = radius[:, i] + radius[:, j]
            hit = alive[:, i] & alive[:, j] & (dx * dx + dy * dy < t * t)
            out[:, i] |= hit
            out[:, j] |= hit
    return out


def near_pairs(pe, px, py, heads, live, reach):
    """Candidate (point index, head slot) pairs: live heads of arena pe[q] within `reach`
    of point q. Callers recheck their exact predicate, so this only has to be conservative.
    Dense version: every point against every head of its arena, in blocks."""
    qs, ks = [], []
    r2 = reach * reach
    for lo in range(0, pe.size, CHUNK):
        ec = pe[lo:lo + CHUNK]
        dx = px[lo:lo + CHUNK, None] - heads[ec, :, 0]
        dy = py[lo:lo + CHUNK, None] - heads[ec, :, 1]
        q, k = np.nonzero(live[ec] & (dx * dx + dy * dy < r2))
        qs.append(q + lo)
        ks.append(k)
    if not qs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(qs), np.concatenate(ks)


def _body_hits(state, heads, radius, alive):
    """For each live head, the lowest slot j != i whose body (points 1..) it touches; S if none."""
    n, s = state.num_envs, state.num_snakes
    first = np.full((n, s), s, np.int32)
    lidx = ring_index(state)
    pe, pj, pp = np.nonzero((lidx >= 1) & (lidx < state.length[..., None]) & alive[..., None])
    px, py = state.body[pe, pj, pp, 0], state.body[pe, pj, pp, 1]
    reach = 2.0 * float(radius.max(initial=0.0)) + 1e-6
    q, i = near_pairs(pe, px, py, heads, alive, reach)
    e, j = pe[q], pj[q]
    dx = px[q] - heads[e, i, 0]
    dy = py[q] - heads[e, i, 1]
    t = radius[e, i] + radius[e, j]
    hit = (dx * dx + dy * dy < t * t) & (i != j)
    np.minimum.at(first, (e[hit], i[hit]), j[hit].astype(np.int32))
    return first
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `uv run pytest tests/test_slither_rules.py tests/test_slither_state.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/slither/rules.py tests/test_slither_rules.py
git commit -m "Slither: steering, boost, movement and collision rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Food, death drops, respawn and arena reset

**Files:**
- Modify: `snake_rl/slither/rules.py` (replace `step_tick`, append helpers)
- Modify: `tests/test_slither_rules.py` (append tests)
- Modify: `snake_rl/slither/__init__.py`

**Interfaces:**
- Consumes: Task 2.
- Produces:
  - the final `step_tick`, covering spec §2 steps 1–7;
  - `reset_arenas(state, enabled (N,S) bool, rng: np.random.Generator) -> None`: clears
    every arena, spawns the enabled slots one by one and fills the natural food.

- [ ] **Step 1: Write the failing tests (append to `tests/test_slither_rules.py`)**

```python
from dataclasses import replace

from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.rules import SlitherState, reset_arenas


def test_eating_adds_value_and_frees_the_slot():
    state = state_of(scene([snake(straight(0, 0, 0, 40))], food=[(3.0, 0.0, 1.0)]))
    ev = tick(state)
    assert ev.eaten[0, 0] == 1.0 and state.mass[0, 0] == 11.0
    assert not state.food_active[0, 0]


def test_pellet_goes_to_nearest_head_and_ties_to_lower_slot():
    a = snake(straight(-5.0, 0.0, 0.0, 40))
    b = snake(straight(5.0, 0.0, math.pi, 40), heading=math.pi)
    state = state_of(scene([a, b], food=[(0.0, 0.0, 1.0), (3.5, 0.0, 2.0)]))
    ev = tick(state)
    assert ev.eaten[0].tolist() == [1.0, 2.0]


def test_mass_is_capped():
    state = state_of(scene([snake(straight(0, 0, 0, 60), mass=499.5)], food=[(2.0, 0.0, 1.0)]))
    tick(state)
    assert state.mass[0, 0] == 500.0


def test_death_drops_pellets_along_the_body():
    state = state_of(scene([snake(straight(296.0, 0.0, 0.0, 40), mass=20.0)]))
    ev = tick(state, u=0.5)  # u = 0.5 means zero jitter
    assert ev.died[0, 0]
    pellets = state.food_pos[0, state.food_active[0]]
    values = state.food_value[0, state.food_active[0]]
    assert len(values) == 10 and np.allclose(values, 1.6)
    assert pellets[0].tolist() == pytest.approx([297.0, 0.0])
    assert state.length[0, 0] == 0 and state.respawn[0, 0] == QUIET.respawn_delay
    assert state.mass[0, 0] == 20.0  # mass is kept until respawn


def test_death_drops_beyond_capacity_are_discarded():
    cfg = replace(QUIET, food_capacity=4)
    snap = scene([snake(straight(296.0, 0.0, 0.0, 40), mass=20.0)],
                 food=[(100.0, 0.0, 1.0), (101.0, 0.0, 1.0), (102.0, 0.0, 1.0)], cfg=cfg)
    state = state_of(snap, cfg=cfg)
    tick(state)
    assert state.food_active[0].sum() == 4


def test_boost_drops_a_pellet_at_the_tail_every_fifth_tick():
    state = state_of(scene([snake(straight(0, 0, 0, 60), mass=20.0, boosting=True, boost_ticks=4)]))
    tick(state, boost=True)
    assert state.boost_ticks[0, 0] == 5
    assert state.food_active[0].sum() == 1 and state.food_value[0, 0] == 0.8
    tail = logical_body(state, 0, 0)[-1]
    np.testing.assert_allclose(state.food_pos[0, 0], tail)


def test_respawn_waits_for_the_timer():
    state = state_of(scene([dead(respawn=2)]))
    tick(state)
    assert not state.alive[0, 0] and state.respawn[0, 0] == 1
    tick(state)
    assert state.alive[0, 0] and state.length[0, 0] == 40 and state.mass[0, 0] == 10.0
    assert np.hypot(*head_points(state)[0, 0]) <= QUIET.arena_radius - QUIET.respawn_margin + 1e-9


def _respawn_u(cfg, slots, slot, pairs, heading_u=0.0):
    u = np.full((1, u_layout(slots, cfg)["size"]), 0.5)
    base = u_layout(slots, cfg)["respawn"] + slot * (2 * cfg.respawn_candidates + 1)
    for c, (ur, ua) in enumerate(pairs):
        u[0, base + 2 * c], u[0, base + 2 * c + 1] = ur, ua
    u[0, base + 2 * cfg.respawn_candidates] = heading_u
    return u


def test_respawn_skips_candidates_near_a_live_head():
    cfg = QUIET
    a = snake(straight(230.0, 0.0, math.pi, 40), heading=math.pi)
    state = state_of(scene([a, dead(respawn=1)]))
    pairs = [(1.0, 0.0), (1.0, 0.5)] + [(1.0, 0.0)] * (cfg.respawn_candidates - 2)
    step_tick(state, np.zeros((1, 2)), np.zeros((1, 2), bool), _respawn_u(cfg, 2, 1, pairs))
    np.testing.assert_allclose(head_points(state)[0, 1], [-240.0, 0.0], atol=1e-9)


def test_respawn_without_a_clear_candidate_takes_the_farthest():
    cfg = QUIET
    a = snake(straight(0.0, 0.0, 0.0, 40))
    state = state_of(scene([a, dead(respawn=1)]))
    pairs = [(0.0, 0.0)] * 3 + [(0.01, 0.5)] + [(0.0, 0.0)] * (cfg.respawn_candidates - 4)
    step_tick(state, np.zeros((1, 2)), np.zeros((1, 2), bool), _respawn_u(cfg, 2, 1, pairs))
    np.testing.assert_allclose(head_points(state)[0, 1], [-24.0, 0.0], atol=1e-9)


def test_respawn_with_no_live_heads_takes_first_candidate():
    cfg = QUIET
    state = state_of(scene([dead(respawn=1), dead(respawn=1)]))
    pairs = [(0.25, 0.25)] * cfg.respawn_candidates
    u = _respawn_u(cfg, 2, 0, pairs)
    step_tick(state, np.zeros((1, 2)), np.zeros((1, 2), bool), u)
    assert state.alive[0].all()
    np.testing.assert_allclose(head_points(state)[0, 0], [0.0, 120.0], atol=1e-9)


def test_disabled_slots_never_spawn():
    state = state_of(scene([dead(respawn=0, enabled=False)]))
    for _ in range(40):
        tick(state)
    assert not state.alive.any()


def test_natural_food_tops_up_two_per_tick():
    cfg = replace(QUIET, natural_food_target=3)
    state = state_of(scene([snake(straight(0, 0, 0, 40))], cfg=cfg), cfg=cfg)
    tick(state, u=0.9)
    assert (state.food_active & state.food_natural).sum() == 2
    tick(state, u=0.9)
    assert (state.food_active & state.food_natural).sum() == 3


def test_max_mass_boosting_snake_stays_within_capacity():
    cfg = QUIET
    state = state_of(scene([snake(straight(0.0, 0.0, 0.0, 1020), mass=500.0)]))
    for _ in range(30):
        tick(state, turn=0.01, boost=True)
    body = logical_body(state, 0, 0)
    assert len(body) <= cfg.body_capacity - 2
    gaps = np.hypot(*np.diff(body, axis=0).T)
    np.testing.assert_allclose(gaps, 1.0, atol=1e-9)


def test_reset_arenas_spawns_enabled_slots_and_food():
    cfg = SlitherConfig()
    state = SlitherState.empty(4, 12, cfg)
    enabled = np.arange(12) < np.array([6, 9, 12, 7])[:, None]
    reset_arenas(state, enabled, np.random.default_rng(0))
    assert (state.alive == enabled).all() and (state.length[enabled] == 40).all()
    assert ((state.food_active & state.food_natural).sum(axis=1) == cfg.natural_food_target).all()
```

The values in these tests:

- In `test_respawn_with_no_live_heads_takes_first_candidate`, `u_r = 0.25` gives
  `rr = 240 · 0.5 = 120` and `u_a = 0.25` gives `θ = π/2`, so the head is at (0, 120).
- In the "farthest" test, the chosen candidate has `rr = 240 · 0.1 = 24` and `θ = π`.

- [ ] **Step 2: Run the tests to check they fail**

Run: `uv run pytest tests/test_slither_rules.py -q`
Expected: FAIL. `reset_arenas` can't be imported, and the eat, drop and respawn tests
fail.

- [ ] **Step 3: Implement**

Replace `step_tick` in `rules.py` with the full version, and append the helpers:

```python
def step_tick(state: SlitherState, turn, boost, u) -> TickEvents:
    """Advance every arena by one physics tick (spec §2 steps 1-7)."""
    cfg = state.cfg
    n, s = state.num_envs, state.num_snakes
    turn, boost, u = _validate(state, turn, boost, u)
    alive = state.alive.copy()

    # 1. steer
    state.heading = np.where(alive, state.heading + np.clip(turn, -cfg.max_turn, cfg.max_turn),
                             state.heading)
    # 2. boost, move, trim
    boosting = alive & boost & (state.mass >= cfg.boost_min_mass)
    state.mass = np.where(boosting, state.mass - cfg.boost_cost, state.mass)
    state.boost_ticks = np.where(boosting, state.boost_ticks + 1, 0).astype(np.int32)
    state.boosting = boosting
    _push(state, alive)
    _push(state, boosting)
    pushed = alive.astype(np.int32) + boosting
    state.length = np.where(alive, np.minimum(state.length + pushed, target_length(state.mass, cfg)),
                            state.length).astype(np.int32)
    de, dk = np.nonzero(boosting & (state.boost_ticks % cfg.boost_drop_every == 0))
    tail = (state.head[de, dk] - (state.length[de, dk] - 1)) % cfg.body_capacity
    boost_drops = (de, state.body[de, dk, tail, 0], state.body[de, dk, tail, 1],
                   np.full(de.size, cfg.boost_drop_value))
    # 3. deaths
    radius = body_radius(state.mass, cfg)
    heads = head_points(state)
    hx, hy = heads[..., 0], heads[..., 1]
    died = alive & (np.sqrt(hx * hx + hy * hy) + radius > cfg.arena_radius)
    head_on = _head_on(hx, hy, radius, alive)
    first = _body_hits(state, heads, radius, alive)
    died |= head_on | (first < s)
    killer = np.where(died & (first < s) & ~head_on, first, -1).astype(np.int32)
    kills = np.zeros((n, s), np.int32)
    ke, ki = np.nonzero(killer >= 0)
    np.add.at(kills, (ke, killer[ke, ki]), 1)
    # 4. eat
    eaten = _eat(state, heads, radius, alive & ~died)
    # 5. death drops; the slot waits to respawn, keeping its mass
    death_drops = _death_drops(state, died, u)
    state.alive = alive & ~died
    state.length[died] = 0
    state.respawn[died] = cfg.respawn_delay
    state.boosting &= ~died
    state.boost_ticks[died] = 0
    # 6. respawn slots that were dead when the tick started
    waiting = state.enabled & ~alive
    state.respawn = np.where(waiting, state.respawn - 1, state.respawn).astype(np.int32)
    _respawn(state, waiting & (state.respawn <= 0), u)
    # 7. new pellets: boost drops, death drops, natural food
    _add_pellets(state, [boost_drops, death_drops, _natural_food(state, u)], [False, False, True])
    state.tick += 1
    return TickEvents(died, killer, kills, eaten)


def _eat(state, heads, radius, eaters):
    """Each pellet within reach goes to the nearest eligible head (ties: lower slot)."""
    cfg = state.cfg
    eaten = np.zeros((state.num_envs, state.num_snakes))
    e, f = np.nonzero(state.food_active)
    reach = radius + cfg.eat_margin
    reach2 = reach * reach
    we, wk, wf = [], [], []
    for lo in range(0, e.size, CHUNK):
        ec, fc = e[lo:lo + CHUNK], f[lo:lo + CHUNK]
        dx = state.food_pos[ec, fc, 0][:, None] - heads[ec, :, 0]
        dy = state.food_pos[ec, fc, 1][:, None] - heads[ec, :, 1]
        d2 = dx * dx + dy * dy
        ok = eaters[ec] & (d2 < reach2[ec])
        best = np.argmin(np.where(ok, d2, np.inf), axis=1)
        got = ok.any(axis=1)
        we.append(ec[got])
        wk.append(best[got])
        wf.append(fc[got])
    if we:
        we, wk, wf = np.concatenate(we), np.concatenate(wk), np.concatenate(wf)
        np.add.at(eaten, (we, wk), state.food_value[we, wf])
        state.food_active[we, wf] = False
    state.mass = np.minimum(state.mass + eaten, cfg.max_mass)
    return eaten


def _death_drops(state, died, u):
    """Pellets for every snake that died: count = min(ceil(mass/2), max), spread along the body."""
    cfg = state.cfg
    lay = u_layout(state.num_snakes, cfg)
    e, k = np.nonzero(died)
    mass = state.mass[e, k]
    count = np.maximum(np.minimum(np.ceil(mass / 2), cfg.death_drop_max_pellets), 1).astype(np.int64)
    value = cfg.death_drop_fraction * mass / count
    owner = np.repeat(np.arange(e.size), count)
    q = np.arange(owner.size) - np.repeat(np.cumsum(count) - count, count)
    ee, kk = e[owner], k[owner]
    idx = (q * state.length[ee, kk]) // count[owner]
    ring = (state.head[ee, kk] - idx) % cfg.body_capacity
    base = lay["drops"] + (kk * cfg.death_drop_max_pellets + q) * 2
    jitter = 2.0 * cfg.death_drop_jitter
    x = state.body[ee, kk, ring, 0] + (u[ee, base] - 0.5) * jitter
    y = state.body[ee, kk, ring, 1] + (u[ee, base + 1] - 0.5) * jitter
    return ee, x, y, value[owner]


def _respawn(state, due, u):
    """Spawn every slot in `due`, slot by slot, at the first candidate point clear of all
    live heads (or the candidate farthest from them)."""
    cfg = state.cfg
    c = cfg.respawn_candidates
    lay = u_layout(state.num_snakes, cfg)
    clearance2 = cfg.respawn_clearance * cfg.respawn_clearance
    for k in range(state.num_snakes):
        envs = np.flatnonzero(due[:, k])
        if envs.size == 0:
            continue
        base = lay["respawn"] + k * (2 * c + 1)
        rr = (cfg.arena_radius - cfg.respawn_margin) * np.sqrt(u[envs, base:base + 2 * c:2])
        th = 2.0 * np.pi * u[envs, base + 1:base + 2 * c:2]
        cx, cy = rr * np.cos(th), rr * np.sin(th)
        heads = head_points(state)[envs]
        dx = cx[:, :, None] - heads[:, None, :, 0]
        dy = cy[:, :, None] - heads[:, None, :, 1]
        d2 = np.where(state.alive[envs][:, None, :], dx * dx + dy * dy, np.inf)
        mind = d2.min(axis=2)
        clear = mind >= clearance2
        pick = np.where(clear.any(axis=1), clear.argmax(axis=1), mind.argmax(axis=1))
        rows = np.arange(envs.size)
        _place(state, envs, k, cx[rows, pick], cy[rows, pick], 2.0 * np.pi * u[envs, base + 2 * c])


def _place(state, envs, k, x, y, heading):
    """A fresh snake: straight body behind (x, y), head at ring index length-1."""
    cfg = state.cfg
    length = int(target_length(cfg.start_mass, cfg))
    i = np.arange(length, dtype=np.float64)
    c, sn = np.cos(heading)[:, None], np.sin(heading)[:, None]
    state.body[envs, k, :length, 0] = (x[:, None] - i * c)[:, ::-1]
    state.body[envs, k, :length, 1] = (y[:, None] - i * sn)[:, ::-1]
    state.head[envs, k] = length - 1
    state.length[envs, k] = length
    state.heading[envs, k] = heading
    state.mass[envs, k] = cfg.start_mass
    state.alive[envs, k] = True
    state.boosting[envs, k] = False
    state.boost_ticks[envs, k] = 0
    state.respawn[envs, k] = 0


def _natural_food(state, u):
    cfg = state.cfg
    lay = u_layout(state.num_snakes, cfg)
    count = (state.food_active & state.food_natural).sum(axis=1)
    need = np.clip(cfg.natural_food_target - count, 0, cfg.natural_food_per_tick)
    env, q = np.nonzero(np.arange(cfg.natural_food_per_tick) < need[:, None])
    rr = (cfg.arena_radius - cfg.food_spawn_margin) * np.sqrt(u[env, lay["food"] + 2 * q])
    th = 2.0 * np.pi * u[env, lay["food"] + 2 * q + 1]
    return env, rr * np.cos(th), rr * np.sin(th), np.full(env.size, cfg.natural_food_value)


def _add_pellets(state, groups, natural):
    """Place new pellets into free food slots, lowest slot first, in group order within each
    arena. Pellets beyond the free slots are discarded."""
    env = np.concatenate([g[0] for g in groups]).astype(np.int64)
    if env.size == 0:
        return
    x = np.concatenate([g[1] for g in groups])
    y = np.concatenate([g[2] for g in groups])
    value = np.concatenate([g[3] for g in groups])
    nat = np.concatenate([np.full(g[0].size, flag) for g, flag in zip(groups, natural)])
    order = np.argsort(env, kind="stable")
    env, x, y, value, nat = env[order], x[order], y[order], value[order], nat[order]
    rank = np.arange(env.size) - np.searchsorted(env, env, side="left")
    free = np.argsort(state.food_active, axis=1, kind="stable")
    keep = rank < (~state.food_active).sum(axis=1)[env]
    e, slot = env[keep], free[env[keep], rank[keep]]
    state.food_pos[e, slot, 0] = x[keep]
    state.food_pos[e, slot, 1] = y[keep]
    state.food_value[e, slot] = value[keep]
    state.food_natural[e, slot] = nat[keep]
    state.food_active[e, slot] = True


def reset_arenas(state: SlitherState, enabled, rng: np.random.Generator) -> None:
    """Start every arena from scratch: enabled slots spawn one by one (as in a respawn) and
    natural food is filled to its target."""
    cfg = state.cfg
    enabled = np.asarray(enabled, bool)
    if enabled.shape != (state.num_envs, state.num_snakes):
        raise ValueError(f"enabled must have shape {(state.num_envs, state.num_snakes)}")
    fresh = SlitherState.empty(state.num_envs, state.num_snakes, cfg)
    for field in fields(SlitherState):
        if field.name != "cfg":
            setattr(state, field.name, getattr(fresh, field.name))
    state.enabled = enabled.copy()
    _respawn(state, enabled, rng.random((state.num_envs, u_layout(state.num_snakes, cfg)["size"])))
    env = np.repeat(np.arange(state.num_envs), cfg.natural_food_target)
    rr = (cfg.arena_radius - cfg.food_spawn_margin) * np.sqrt(rng.random(env.size))
    th = 2.0 * np.pi * rng.random(env.size)
    _add_pellets(state, [(env, rr * np.cos(th), rr * np.sin(th),
                          np.full(env.size, cfg.natural_food_value))], [True])
```

Update `snake_rl/slither/__init__.py` so it exports the rules:

```python
"""Slither mode: a continuous, slither.io-style arena (spec 2026-10-15-slither-mode-design)."""

from snake_rl.slither.config import SlitherConfig, u_layout
from snake_rl.slither.rules import (SlitherState, TickEvents, from_snapshots, reset_arenas,
                                    snapshot, step_tick)

__all__ = ["SlitherConfig", "SlitherState", "TickEvents", "from_snapshots", "reset_arenas",
           "snapshot", "step_tick", "u_layout"]
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `uv run pytest tests/test_slither_rules.py tests/test_slither_state.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/slither tests/test_slither_rules.py
git commit -m "Slither: eating, death drops, respawn and arena reset

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 4: Reference implementation and tick-for-tick match

**Files:**
- Create: `tests/slither_reference.py`, `tests/test_slither_reference.py`
- Modify: `tests/slither_helpers.py` (append `SMALL`, `random_ticks`)

**Interfaces:**
- Consumes: Tasks 1–3 (`snapshot`, `from_snapshots`, `reset_arenas`, `step_tick`, `u_layout`).
- Produces:
  - `slither_reference.step_tick(snap, turns: list[float], boosts: list[bool], u: list[float], cfg) -> {"died", "killer", "kills", "eaten"}`
    (lists per slot). It mutates `snap`. Task 12 ports this function line by line to JS.
  - `slither_helpers.SMALL` (a crowded test config).
  - `slither_helpers.random_ticks(cfg, num_envs, slots, ticks, seed)`, which yields
    `(state, turn, boost, u)` before each tick.

- [ ] **Step 1: Add the run generator to `tests/slither_helpers.py`**

```python
import numpy as np

from snake_rl.slither.config import u_layout
from snake_rl.slither.rules import SlitherState, reset_arenas

# A small, crowded arena: frequent deaths, kills and respawns.
SMALL = replace(SlitherConfig(), arena_radius=120.0, respawn_margin=30.0, respawn_clearance=20.0,
                respawn_delay=5, natural_food_target=60)


def random_ticks(cfg, num_envs, slots, ticks, seed):
    """Yield (state, turn, boost, u) before each of `ticks` random ticks; the caller steps."""
    rng = np.random.default_rng(seed)
    state = SlitherState.empty(num_envs, slots, cfg)
    counts = rng.integers(2, slots + 1, num_envs)
    reset_arenas(state, np.arange(slots) < counts[:, None], rng)
    size = u_layout(slots, cfg)["size"]
    for _ in range(ticks):
        turn = rng.uniform(-0.2, 0.2, (num_envs, slots))
        boost = rng.random((num_envs, slots)) < 0.3
        u = rng.random((num_envs, size))
        yield state, turn, boost, u
```

Put the new imports at the top of the file, next to the existing ones.

- [ ] **Step 2: Write the failing tests**

`tests/test_slither_reference.py`:

```python
"""The batched engine must match the slow reference tick for tick (positions within 1e-9,
discrete events exactly), and batching arenas must not change any result."""

import copy

import numpy as np
import pytest

import slither_reference as ref
from slither_helpers import SMALL, random_ticks
from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.rules import from_snapshots, snapshot, step_tick


def assert_same(got, want, where):
    assert got["tick"] == want["tick"], where
    for k, (a, b) in enumerate(zip(got["snakes"], want["snakes"], strict=True)):
        for key in ("alive", "enabled", "boosting", "boost_ticks", "respawn"):
            assert a[key] == b[key], f"{where} snake {k} {key}: {a[key]} != {b[key]}"
        for key in ("heading", "mass"):
            assert abs(a[key] - b[key]) <= 1e-9, f"{where} snake {k} {key}: {a[key]} != {b[key]}"
        assert len(a["body"]) == len(b["body"]), f"{where} snake {k} body length"
        if a["body"]:
            np.testing.assert_allclose(a["body"], b["body"], rtol=0, atol=1e-9,
                                       err_msg=f"{where} snake {k} body")
    for f, (a, b) in enumerate(zip(got["food"], want["food"], strict=True)):
        assert (a is None) == (b is None), f"{where} food slot {f}"
        if a is not None:
            np.testing.assert_allclose(a[:3], b[:3], rtol=0, atol=1e-9, err_msg=f"{where} food {f}")
            assert a[3] == b[3], f"{where} food {f} natural flag"


@pytest.mark.parametrize("cfg,num_envs,slots,ticks,seed,crowded", [
    (SMALL, 4, 8, 400, 0, True),
    (SlitherConfig(), 2, 12, 300, 1, False),
])
def test_batched_engine_matches_reference(cfg, num_envs, slots, ticks, seed, crowded):
    died = kills = respawns = 0
    eaten = 0.0
    for t, (state, turn, boost, u) in enumerate(random_ticks(cfg, num_envs, slots, ticks, seed)):
        before = [snapshot(state, e) for e in range(num_envs)]
        alive_before = state.alive.copy()
        ev = step_tick(state, turn, boost, u)
        for e in range(num_envs):
            want = copy.deepcopy(before[e])
            rev = ref.step_tick(want, turn[e].tolist(), boost[e].tolist(), u[e].tolist(), cfg)
            where = f"tick {t} arena {e}"
            assert_same(snapshot(state, e), want, where)
            assert ev.died[e].tolist() == rev["died"], where
            assert ev.killer[e].tolist() == rev["killer"], where
            assert ev.kills[e].tolist() == rev["kills"], where
            np.testing.assert_allclose(ev.eaten[e], rev["eaten"], rtol=0, atol=1e-9, err_msg=where)
        died += int(ev.died.sum())
        kills += int(ev.kills.sum())
        eaten += float(ev.eaten.sum())
        respawns += int((~alive_before & state.alive).sum())
    assert eaten > 0
    if crowded:  # make sure the run exercised every rule, not just movement
        assert died > 10 and kills > 0 and respawns > 0


def test_batching_does_not_change_results():
    singles = None
    for state, turn, boost, u in random_ticks(SMALL, 4, 8, 150, 2):
        if singles is None:
            singles = [from_snapshots([snapshot(state, e)], SMALL) for e in range(4)]
        step_tick(state, turn, boost, u)
        for e, single in enumerate(singles):
            step_tick(single, turn[e:e + 1], boost[e:e + 1], u[e:e + 1])
            assert snapshot(single, 0) == snapshot(state, e)
```

- [ ] **Step 3: Run the tests to check they fail**

Run: `uv run pytest tests/test_slither_reference.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'slither_reference'`. (The
batching test may already pass.)

- [ ] **Step 4: Write the reference**

`tests/slither_reference.py`. Every expression mirrors `rules.py`, operation for
operation. Keep it that way, because `web/slither/rules.js` is ported from this file.

```python
"""Deliberately simple slither rules: plain Python, one arena, one snake at a time.

Works on snapshot dicts (snake_rl.slither.rules.snapshot). The batched engine must
reproduce it tick for tick, and web/slither/rules.js is a line-by-line port of it, so keep
every arithmetic expression in the same form and order as the other two.
"""

import math

from snake_rl.slither.config import u_layout


def target_length(mass, cfg):
    return math.floor(cfg.base_length + cfg.length_per_mass * mass)


def body_radius(mass, cfg):
    return cfg.base_radius + cfg.radius_per_sqrt_mass * math.sqrt(mass)


def step_tick(snap, turns, boosts, u, cfg):
    """Advance one arena by one physics tick, in place (spec §2 steps 1-7)."""
    snakes, food = snap["snakes"], snap["food"]
    count = len(snakes)
    lay = u_layout(count, cfg)
    alive = [s["alive"] for s in snakes]

    # 1. steer
    for k, s in enumerate(snakes):
        if alive[k]:
            s["heading"] = s["heading"] + min(max(turns[k], -cfg.max_turn), cfg.max_turn)

    # 2. boost, move, trim
    boost_drops = []
    for k, s in enumerate(snakes):
        b = alive[k] and boosts[k] and s["mass"] >= cfg.boost_min_mass
        if b:
            s["mass"] = s["mass"] - cfg.boost_cost
        s["boost_ticks"] = s["boost_ticks"] + 1 if b else 0
        s["boosting"] = b
        if not alive[k]:
            continue
        for _ in range(2 if b else 1):
            hx, hy = s["body"][0]
            s["body"].insert(0, [hx + math.cos(s["heading"]), hy + math.sin(s["heading"])])
        del s["body"][min(len(s["body"]), target_length(s["mass"], cfg)):]
        if b and s["boost_ticks"] % cfg.boost_drop_every == 0:
            tx, ty = s["body"][-1]
            boost_drops.append([tx, ty, cfg.boost_drop_value, False])

    # 3. deaths
    radius = [body_radius(s["mass"], cfg) for s in snakes]
    died, killer, kills = [False] * count, [-1] * count, [0] * count
    for i in range(count):
        if not alive[i]:
            continue
        hx, hy = snakes[i]["body"][0]
        wall = math.sqrt(hx * hx + hy * hy) + radius[i] > cfg.arena_radius
        head_on, first = False, -1
        for j in range(count):
            if j == i or not alive[j]:
                continue
            t = radius[i] + radius[j]
            ox, oy = snakes[j]["body"][0]
            dx, dy = hx - ox, hy - oy
            if dx * dx + dy * dy < t * t:
                head_on = True
            if first < 0:
                for px, py in snakes[j]["body"][1:]:
                    dx, dy = px - hx, py - hy
                    if dx * dx + dy * dy < t * t:
                        first = j
                        break
        died[i] = wall or head_on or first >= 0
        killer[i] = first if first >= 0 and not head_on else -1
    for j in killer:
        if j >= 0:
            kills[j] += 1

    # 4. eat
    eaten = [0.0] * count
    for f, item in enumerate(food):
        if item is None:
            continue
        best, best_d2 = -1, math.inf
        for k in range(count):
            if not alive[k] or died[k]:
                continue
            hx, hy = snakes[k]["body"][0]
            dx, dy = item[0] - hx, item[1] - hy
            d2 = dx * dx + dy * dy
            reach = radius[k] + cfg.eat_margin
            if d2 < reach * reach and d2 < best_d2:
                best, best_d2 = k, d2
        if best >= 0:
            eaten[best] += item[2]
            food[f] = None
    for k, s in enumerate(snakes):
        s["mass"] = min(s["mass"] + eaten[k], cfg.max_mass)

    # 5. death drops
    death_drops = []
    most = cfg.death_drop_max_pellets
    jitter = 2.0 * cfg.death_drop_jitter
    for k, s in enumerate(snakes):
        if not died[k]:
            continue
        length, m = len(s["body"]), s["mass"]
        n = max(min(math.ceil(m / 2), most), 1)
        value = cfg.death_drop_fraction * m / n
        for q in range(n):
            px, py = s["body"][(q * length) // n]
            base = lay["drops"] + (k * most + q) * 2
            death_drops.append([px + (u[base] - 0.5) * jitter, py + (u[base + 1] - 0.5) * jitter,
                                value, False])
        s.update(alive=False, body=[], respawn=cfg.respawn_delay, boosting=False, boost_ticks=0)

    # 6. respawn slots that were dead when the tick started
    for k, s in enumerate(snakes):
        if s["enabled"] and not alive[k]:
            s["respawn"] -= 1
            if s["respawn"] <= 0:
                respawn(snap, k, u, cfg)

    # 7. new pellets
    natural = sum(1 for item in food if item is not None and item[3])
    need = max(0, min(cfg.natural_food_target - natural, cfg.natural_food_per_tick))
    fresh = boost_drops + death_drops
    for q in range(need):
        rr = (cfg.arena_radius - cfg.food_spawn_margin) * math.sqrt(u[lay["food"] + 2 * q])
        th = 2.0 * math.pi * u[lay["food"] + 2 * q + 1]
        fresh.append([rr * math.cos(th), rr * math.sin(th), cfg.natural_food_value, True])
    free = [f for f, item in enumerate(food) if item is None]
    for f, item in zip(free, fresh):
        food[f] = item
    snap["tick"] += 1
    return {"died": died, "killer": killer, "kills": kills, "eaten": eaten}


def respawn(snap, k, u, cfg):
    """Place slot k at the first candidate clear of every live head, else the farthest one."""
    snakes = snap["snakes"]
    c = cfg.respawn_candidates
    base = u_layout(len(snakes), cfg)["respawn"] + k * (2 * c + 1)
    clearance2 = cfg.respawn_clearance * cfg.respawn_clearance
    candidates = []
    for q in range(c):
        rr = (cfg.arena_radius - cfg.respawn_margin) * math.sqrt(u[base + 2 * q])
        th = 2.0 * math.pi * u[base + 2 * q + 1]
        x, y = rr * math.cos(th), rr * math.sin(th)
        nearest = math.inf
        for other in snakes:
            if other["alive"]:
                ox, oy = other["body"][0]
                dx, dy = x - ox, y - oy
                nearest = min(nearest, dx * dx + dy * dy)
        candidates.append((x, y, nearest))
    pick = next((q for q in range(c) if candidates[q][2] >= clearance2), None)
    if pick is None:
        pick = max(range(c), key=lambda q: candidates[q][2])  # first of equal maxima
    x, y, _ = candidates[pick]
    h = 2.0 * math.pi * u[base + 2 * c]
    cos_h, sin_h = math.cos(h), math.sin(h)
    body = [[x - i * cos_h, y - i * sin_h] for i in range(target_length(cfg.start_mass, cfg))]
    snakes[k].update(body=body, heading=h, mass=cfg.start_mass, alive=True, boosting=False,
                     boost_ticks=0, respawn=0)
```

- [ ] **Step 5: Run the tests to check they pass**

Run: `uv run pytest tests/test_slither_reference.py -q`
Expected: 3 passed.

If a mismatch appears, the assertion message names the tick, arena and field. Find the
expression in `rules.py` that differs in form or order from the reference and fix it,
not the tolerance. The usual culprits:

- `a - b` written as `b - a` before squaring (harmless);
- a sum taken in a different order;
- using a value computed before a mutation instead of after it.

- [ ] **Step 6: Commit**

```bash
git add tests/slither_reference.py tests/test_slither_reference.py tests/slither_helpers.py
git commit -m "Slither: reference rules and tick-for-tick match test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Actions and sector observations

**Files:**
- Create: `snake_rl/slither/actions.py`, `snake_rl/slither/observations.py`, `tests/test_slither_observations.py`
- Modify: `tests/slither_reference.py` (append `observe`)

**Interfaces:**
- Consumes: Tasks 1–4 (`near_pairs`, `ring_index`, `head_points`, `body_radius`, `random_ticks`, `SMALL`).
- Produces:
  - `NUM_ACTIONS = 18` and `STEER_BINS = 9`.
  - `decode_actions(actions, cfg) -> (turn f64, boost bool)`.
  - `action_mask(state) -> (N,S,18) bool`.
  - `steer_bin(phi, cfg) -> int64`.
  - `OBS_CHANNELS = 8`, `NUM_SCALARS = 7` and `obs_size(cfg) -> 263`.
  - `build_observations(state) -> float32 (N,S,263)`, laid out as
    `obs[q*8 + ch]` for sector `q` and channel `ch`, then the scalars at `256..262`.
  - `slither_reference.observe(snap, k, cfg) -> list[float]`, which Task 12 ports to JS.
- **Channels:**
  0. wall;
  1. enemy body;
  2. enemy size: the largest radius seen in the sector, / 15;
  3. own body (points with index ≥ `neck`);
  4. food: total value / 10, capped at 1;
  5. enemy head;
  6. and 7. the enemy head's relative heading, as cos and sin.

  Distances are encoded as `clip(1 − d/V, 0, 1)`, where `d` is measured to the body
  surface for bodies and heads.
- **Scalars:**
  - `log(mass)/log(max_mass)`;
  - `radius/15`;
  - whether boosting;
  - whether boost is available;
  - `|head|/R`;
  - the bearing to the centre in the heading frame, as cos and sin, or (1, 0) at the
    centre.

- [ ] **Step 1: Write the failing tests**

`tests/test_slither_observations.py`:

```python
import math

import numpy as np
import pytest

import slither_reference as ref
from slither_helpers import QUIET, SMALL, random_ticks, scene, snake, state_of, straight
from snake_rl.slither.actions import NUM_ACTIONS, action_mask, decode_actions, steer_bin
from snake_rl.slither.observations import build_observations, obs_size
from snake_rl.slither.rules import snapshot

C = 8
V = QUIET.vision
W = 2.0 * math.pi / QUIET.sectors
R0 = 2.0 + 0.5 * math.sqrt(10.0)


def obs_of(snap):
    return build_observations(state_of(snap))[0]


def at(q, d):
    """A point at distance d along the centre of sector q, for an observer at the origin facing +x."""
    a = -math.pi + (q + 0.5) * W
    return d * math.cos(a), d * math.sin(a)


def me():
    return snake(straight(0.0, 0.0, 0.0, 40))


def test_shape_dtype_and_dead_rows_are_zero():
    o = build_observations(state_of(scene([me(), snake([], alive=False)])))
    assert o.shape == (1, 2, obs_size(QUIET)) == (1, 2, 263) and o.dtype == np.float32
    assert not o[0, 1].any()


def test_wall_ahead():
    o = obs_of(scene([snake(straight(280.0, 0.0, 0.0, 40))]))[0]
    phi = 0.5 * W  # centre of sector 16
    b = 280.0 * math.cos(phi)
    t = -b + math.sqrt(b * b - (280.0 ** 2 - 300.0 ** 2))
    assert o[16 * C + 0] == pytest.approx(1 - t / V, abs=1e-6)
    assert o[15 * C + 0] == pytest.approx(1 - t / V, abs=1e-6)
    assert o[0 * C + 0] == 0.0
    assert not obs_of(scene([me()]))[0][0:256:C].any()  # far from every wall


def test_enemy_body_size_and_head():
    enemy = snake(straight(20.0, 25.0, math.pi / 2, 40), heading=math.pi / 2)
    o = obs_of(scene([me(), enemy]))[0]
    assert o[16 * C + 1] == pytest.approx(1 - (20.0 - R0) / V, abs=1e-6)
    assert o[16 * C + 2] == pytest.approx(R0 / 15, abs=1e-6)
    assert o[20 * C + 5] == pytest.approx(1 - (math.hypot(20, 25) - R0) / V, abs=1e-6)
    assert o[20 * C + 6] == pytest.approx(0.0, abs=1e-6)
    assert o[20 * C + 7] == pytest.approx(1.0, abs=1e-6)


def test_own_body_skips_the_neck():
    o = obs_of(scene([me()]))[0]
    assert o[0 * C + 3] == pytest.approx(1 - (10.0 - R0) / V, abs=1e-6)
    assert not o[1:256:C].any()  # no enemies


def test_food_is_summed_per_sector():
    pellets = [(*at(24, 10.0), 1.0), (*at(24, 30.0), 2.0), (*at(8, 10.0), 1.0), (*at(8, 70.0), 9.0)]
    o = obs_of(scene([me()], food=pellets))[0]
    assert o[24 * C + 4] == pytest.approx(0.3, abs=1e-6)
    assert o[8 * C + 4] == pytest.approx(0.1, abs=1e-6)  # the far pellet is out of sight


def test_scalars():
    o = obs_of(scene([me(), snake(straight(100.0, 0.0, math.pi / 2, 40), heading=math.pi / 2, mass=16.0)]))
    np.testing.assert_allclose(o[0, 256:], [math.log(10) / math.log(500), R0 / 15, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(o[1, 256:], [math.log(16) / math.log(500), 4.0 / 15, 0, 1, 1 / 3, 0, 1], atol=1e-6)


def _scene_rotated(alpha=0.0, extra_turns=0):
    def rot(p):
        return [p[0] * math.cos(alpha) - p[1] * math.sin(alpha), p[0] * math.sin(alpha) + p[1] * math.cos(alpha)]
    turn = 2 * math.pi * extra_turns
    a = snake([rot(p) for p in straight(10.0, -20.0, 0.4, 40)], heading=0.4 + alpha + turn)
    b = snake([rot(p) for p in straight(35.0, 5.0, 2.5, 60)], heading=2.5 + alpha + turn, mass=20.0)
    food = [(*rot(p), v) for p, v in [((30.0, -10.0), 1.0), ((0.0, 0.0), 2.0), ((-15.0, -40.0), 1.0)]]
    return scene([a, b], food=food)


def test_observation_invariant_to_rotation():
    np.testing.assert_allclose(obs_of(_scene_rotated(0.7)), obs_of(_scene_rotated(0.0)), atol=1e-6)


def test_observation_invariant_to_full_turns_of_heading():
    np.testing.assert_allclose(obs_of(_scene_rotated(0.0, extra_turns=200)), obs_of(_scene_rotated(0.0)),
                               atol=1e-6)


def test_matches_reference():
    from snake_rl.slither.rules import step_tick
    for t, (state, turn, boost, u) in enumerate(random_ticks(SMALL, 3, 8, 120, 5)):
        if t % 10 == 0:
            obs = build_observations(state)
            for e in range(3):
                snap = snapshot(state, e)
                for k in range(8):
                    np.testing.assert_allclose(obs[e, k], ref.observe(snap, k, SMALL), atol=1e-6,
                                               err_msg=f"tick {t} arena {e} snake {k}")
        step_tick(state, turn, boost, u)


def test_actions_decode_and_mask():
    turn, boost = decode_actions(np.array([0, 1, 8, 9, 17]), QUIET)
    np.testing.assert_allclose(turn, [-0.15, -0.15, 0.0, 0.0, 0.15])
    assert boost.tolist() == [False, True, False, True, True]
    state = state_of(scene([snake(straight(0, 0, 0, 40), mass=14.0), snake(straight(0, 50, 0, 40), mass=15.0)]))
    mask = action_mask(state)
    assert mask.shape == (1, 2, NUM_ACTIONS)
    assert mask[0, 0, ::2].all() and not mask[0, 0, 1::2].any() and mask[0, 1].all()


def test_steer_bin():
    assert steer_bin(np.array([0.0, 0.45, -10.0, 0.1125]), QUIET).tolist() == [4, 8, 0, 5]
```

These scenes avoid sector boundaries:

- The pellets sit on sector centres.
- The enemy-body point (20, 0) is at angle 0. `π / (π/16)` is exactly 16, so it lands
  deterministically in sector 16.
- The enemy head is at about 51°, the middle of sector 20.

- [ ] **Step 2: Run the tests to check they fail**

Run: `uv run pytest tests/test_slither_observations.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'snake_rl.slither.actions'`

- [ ] **Step 3: Implement**

`snake_rl/slither/actions.py`:

```python
"""The 18 discrete actions: 9 steering bins x {cruise, boost} (spec §3). action = steer * 2 + boost."""

from __future__ import annotations

import numpy as np

STEER_BINS = 9
NUM_ACTIONS = 2 * STEER_BINS


def decode_actions(actions, cfg):
    """-> (turn per tick, boost flag). Bin b turns max_turn * (b - 4) / 4 every tick of the decision."""
    a = np.asarray(actions, np.int64)
    steer = a // 2
    return cfg.max_turn * (steer - 4) / 4.0, a % 2 == 1


def action_mask(state) -> np.ndarray:
    """(N, S, 18) bool: boost actions need boost_min_mass; steering is always allowed."""
    mask = np.ones(state.mass.shape + (NUM_ACTIONS,), bool)
    mask[..., 1::2] = (state.mass >= state.cfg.boost_min_mass)[..., None]
    return mask


def steer_bin(phi, cfg) -> np.ndarray:
    """The steering bin that turns toward relative angle `phi` (radians) over one decision."""
    full = cfg.max_turn * cfg.ticks_per_decision
    return (np.rint(np.clip(np.asarray(phi, np.float64) / full, -1.0, 1.0) * 4) + 4).astype(np.int64)
```

`snake_rl/slither/observations.py`:

```python
"""Head-centred sector observation (spec §3): 32 sectors x 8 channels + 7 scalars.

Everything is rotated into the snake's heading frame. A point's sector is
floor((atan2(y', x') + pi) / (2 pi / K)) mod K; sector K/2 starts straight ahead and
positive angles are to the left (counter-clockwise). tests/slither_reference.observe is
the loop version; web/slither/observe.js ports it.
"""

from __future__ import annotations

import numpy as np

from snake_rl.slither.rules import body_radius, head_points, near_pairs, ring_index

OBS_CHANNELS = 8
NUM_SCALARS = 7
SIZE_SCALE = 15.0  # radius normaliser (radius is about 13 at max mass)


def obs_size(cfg) -> int:
    return cfg.sectors * OBS_CHANNELS + NUM_SCALARS


def _close(d, vision):
    return np.clip(1.0 - d / vision, 0.0, 1.0)


def _sector(rx, ry, c, sn, width, k):
    x = rx * c + ry * sn
    y = ry * c - rx * sn
    return np.floor((np.arctan2(y, x) + np.pi) / width).astype(np.int64) % k


def build_observations(state) -> np.ndarray:
    """float32 (N, S, obs_size); all zeros for dead or disabled slots."""
    cfg = state.cfg
    n, s = state.num_envs, state.num_snakes
    k, vision = cfg.sectors, cfg.vision
    width = 2.0 * np.pi / k
    alive = state.alive
    heads = head_points(state)
    hx, hy = heads[..., 0], heads[..., 1]
    c, sn = np.cos(state.heading), np.sin(state.heading)
    radius = body_radius(state.mass, cfg)
    sec = np.zeros((n, s, k, OBS_CHANNELS))
    flat = sec.reshape(-1)
    slots = np.arange(s)

    # 0: wall, along each sector's centre ray
    th = state.heading[..., None] + (-np.pi + (np.arange(k) + 0.5) * width)
    dx, dy = np.cos(th), np.sin(th)
    b = hx[..., None] * dx + hy[..., None] * dy
    cc = (hx * hx + hy * hy - cfg.arena_radius * cfg.arena_radius)[..., None]
    sec[..., 0] = _close(-b + np.sqrt(np.maximum(b * b - cc, 0.0)), vision)

    # 1-3: bodies (every live point, including heads)
    lidx = ring_index(state)
    pe, pj, pp = np.nonzero((lidx < state.length[..., None]) & alive[..., None])
    pl = lidx[pe, pj, pp]
    px, py = state.body[pe, pj, pp, 0], state.body[pe, pj, pp, 1]
    q, i = near_pairs(pe, px, py, heads, alive, vision + float(radius.max(initial=0.0)) + 1e-6)
    e, j, l = pe[q], pj[q], pl[q]
    rx, ry = px[q] - hx[e, i], py[q] - hy[e, i]
    v = _close(np.sqrt(rx * rx + ry * ry) - radius[e, j], vision)
    base = ((e * s + i) * k + _sector(rx, ry, c[e, i], sn[e, i], width, k)) * OBS_CHANNELS
    enemy = (j != i) & (v > 0)
    mine = (j == i) & (l >= cfg.neck) & (v > 0)
    np.maximum.at(flat, base[enemy] + 1, v[enemy])
    np.maximum.at(flat, base[enemy] + 2, radius[e, j][enemy] / SIZE_SCALE)
    np.maximum.at(flat, base[mine] + 3, v[mine])

    # 4: food value per sector
    fe, ff = np.nonzero(state.food_active)
    fx, fy = state.food_pos[fe, ff, 0], state.food_pos[fe, ff, 1]
    q, i = near_pairs(fe, fx, fy, heads, alive, vision)
    e = fe[q]
    rx, ry = fx[q] - hx[e, i], fy[q] - hy[e, i]
    seen = np.sqrt(rx * rx + ry * ry) < vision
    food = np.zeros((n, s, k))
    idx = ((e * s + i) * k + _sector(rx, ry, c[e, i], sn[e, i], width, k))[seen]
    np.add.at(food.reshape(-1), idx, state.food_value[e, ff[q]][seen])
    sec[..., 4] = np.minimum(food / 10.0, 1.0)

    # 5-7: nearest enemy head per sector (ties: lower slot) and its relative heading
    for j in range(s):
        rx, ry = hx[:, j:j + 1] - hx, hy[:, j:j + 1] - hy
        v = _close(np.sqrt(rx * rx + ry * ry) - radius[:, j:j + 1], vision)
        ok = alive & alive[:, j:j + 1] & (slots != j) & (v > 0)
        ee, ii = np.nonzero(ok)
        qq = _sector(rx[ee, ii], ry[ee, ii], c[ee, ii], sn[ee, ii], width, k)
        better = v[ee, ii] > sec[ee, ii, qq, 5]
        ee, ii, qq = ee[better], ii[better], qq[better]
        sec[ee, ii, qq, 5] = v[ee, ii]
        dh = state.heading[ee, j] - state.heading[ee, ii]
        sec[ee, ii, qq, 6] = np.cos(dh)
        sec[ee, ii, qq, 7] = np.sin(dh)

    # scalars
    sc = np.zeros((n, s, NUM_SCALARS))
    sc[..., 0] = np.log(np.where(alive, state.mass, 1.0)) / np.log(cfg.max_mass)
    sc[..., 1] = radius / SIZE_SCALE
    sc[..., 2] = state.boosting
    sc[..., 3] = state.mass >= cfg.boost_min_mass
    dist = np.sqrt(hx * hx + hy * hy)
    sc[..., 4] = dist / cfg.arena_radius
    away = dist > 1e-9
    safe = np.where(away, dist, 1.0)
    bx, by = -hx / safe, -hy / safe
    sc[..., 5] = np.where(away, bx * c + by * sn, 1.0)
    sc[..., 6] = np.where(away, by * c - bx * sn, 0.0)

    obs = np.concatenate([sec.reshape(n, s, -1), sc], axis=-1)
    obs[~alive] = 0.0
    return obs.astype(np.float32)
```

Note on the food pass: inside a single arena, `near_pairs` returns pairs ordered by
pellet, so each sector's sum is taken in pellet-slot order, the same as the reference.

Append to `tests/slither_reference.py`:

```python
def observe(snap, k, cfg):
    """The spec §3 observation of snake k, as a list of floats (zeros if it is dead)."""
    K, V = cfg.sectors, cfg.vision
    width = 2.0 * math.pi / K
    out = [0.0] * (K * 8 + 7)
    snakes = snap["snakes"]
    me = snakes[k]
    if not me["alive"]:
        return out
    hx, hy = me["body"][0]
    h = me["heading"]
    c, sn = math.cos(h), math.sin(h)
    radius = [body_radius(s["mass"], cfg) for s in snakes]

    def close(d):
        return min(max(1.0 - d / V, 0.0), 1.0)

    def sector(rx, ry):
        x = rx * c + ry * sn
        y = ry * c - rx * sn
        return math.floor((math.atan2(y, x) + math.pi) / width) % K

    for q in range(K):
        th = h + (-math.pi + (q + 0.5) * width)
        dx, dy = math.cos(th), math.sin(th)
        b = hx * dx + hy * dy
        cc = hx * hx + hy * hy - cfg.arena_radius * cfg.arena_radius
        out[q * 8] = close(-b + math.sqrt(max(b * b - cc, 0.0)))

    for j, other in enumerate(snakes):
        if not other["alive"]:
            continue
        for idx, (px, py) in enumerate(other["body"]):
            if j == k and idx < cfg.neck:
                continue
            rx, ry = px - hx, py - hy
            v = close(math.sqrt(rx * rx + ry * ry) - radius[j])
            if v <= 0:
                continue
            q = sector(rx, ry)
            if j == k:
                out[q * 8 + 3] = max(out[q * 8 + 3], v)
            else:
                out[q * 8 + 1] = max(out[q * 8 + 1], v)
                out[q * 8 + 2] = max(out[q * 8 + 2], radius[j] / 15.0)

    food = [0.0] * K
    for item in snap["food"]:
        if item is None:
            continue
        rx, ry = item[0] - hx, item[1] - hy
        if math.sqrt(rx * rx + ry * ry) < V:
            food[sector(rx, ry)] += item[2]
    for q in range(K):
        out[q * 8 + 4] = min(food[q] / 10.0, 1.0)

    for j, other in enumerate(snakes):
        if j == k or not other["alive"]:
            continue
        ox, oy = other["body"][0]
        rx, ry = ox - hx, oy - hy
        v = close(math.sqrt(rx * rx + ry * ry) - radius[j])
        if v <= 0:
            continue
        q = sector(rx, ry)
        if v > out[q * 8 + 5]:
            out[q * 8 + 5] = v
            out[q * 8 + 6] = math.cos(other["heading"] - h)
            out[q * 8 + 7] = math.sin(other["heading"] - h)

    base = K * 8
    out[base] = math.log(me["mass"]) / math.log(cfg.max_mass)
    out[base + 1] = radius[k] / 15.0
    out[base + 2] = 1.0 if me["boosting"] else 0.0
    out[base + 3] = 1.0 if me["mass"] >= cfg.boost_min_mass else 0.0
    dist = math.sqrt(hx * hx + hy * hy)
    out[base + 4] = dist / cfg.arena_radius
    if dist > 1e-9:
        bx, by = -hx / dist, -hy / dist
        out[base + 5] = bx * c + by * sn
        out[base + 6] = by * c - bx * sn
    else:
        out[base + 5] = 1.0
    return out
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `uv run pytest tests/test_slither_observations.py tests/test_slither_reference.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/slither/actions.py snake_rl/slither/observations.py tests/test_slither_observations.py tests/slither_reference.py
git commit -m "Slither: 18 discrete actions and the 263-float sector observation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: SlitherEnv

**Files:**
- Create: `snake_rl/slither/env.py`, `tests/test_slither_env.py`
- Modify: `snake_rl/slither/__init__.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `SlitherEnv(num_envs, max_snakes=12, min_snakes=6, config=None, seed=None, mass_scale=0.01, death_penalty=1.0, survival_bonus=0.0)`.
  - Properties `.cfg`, `.state`, `.num_envs`, `.num_snakes`, `.obs_size`.
  - `reset() -> (obs (N,S,263) float32, info)`.
  - `step(actions (N,S) int in [0,18)) -> (obs, rewards float32 (N,S), dones bool (N,S), info)`.
  - `info` keys: `active`, `alive`, `enabled`, `action_mask`, `mass`, `died`, `killer`,
    `kills`.

**Semantics:**

- One `step` is one decision of `cfg.ticks_per_decision` ticks.
- `active` means the slot was alive when the decision started. Every other slot is
  padding: reward 0, done False, and its action is ignored.
- `rewards = mass_scale · (mass after − mass before)` for active slots. Add
  `− death_penalty` if the snake died during the decision, and `+ survival_bonus` if it
  is active and survived.
- `dones = died` (a subset of active).
- Arenas never reset after `reset()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_slither_env.py`:

```python
from dataclasses import replace

import numpy as np
import pytest

from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.env import SlitherEnv

# Small arena so random play produces deaths and kills within a few dozen decisions.
CROWDED = replace(SlitherConfig(), arena_radius=120.0, respawn_margin=30.0, respawn_clearance=20.0,
                  natural_food_target=60)


def test_reset_shapes_and_enabled_counts():
    env = SlitherEnv(16, max_snakes=12, min_snakes=6, seed=0)
    obs, info = env.reset()
    assert obs.shape == (16, 12, 263) and obs.dtype == np.float32
    counts = info["enabled"].sum(axis=1)
    assert counts.min() >= 6 and counts.max() <= 12
    assert (info["alive"] == info["enabled"]).all()
    assert not obs[~info["enabled"]].any()
    assert info["action_mask"].shape == (16, 12, 18)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        SlitherEnv(2, seed=0).step(np.zeros((2, 12), int))


def test_rewards_follow_mass_change_and_death():
    env = SlitherEnv(32, max_snakes=8, min_snakes=8, config=CROWDED, seed=1, survival_bonus=0.001)
    obs, info = env.reset()
    rng = np.random.default_rng(1)
    seen_death = False
    for _ in range(60):
        before = info["mass"].copy()
        actions = rng.integers(0, 18, (32, 8))
        obs, rewards, dones, info = env.step(actions)
        active, died = info["active"], info["died"]
        want = np.where(active, 0.01 * (info["mass"] - before) - 1.0 * died + 0.001 * (active & ~died), 0.0)
        np.testing.assert_allclose(rewards, want, atol=1e-6)
        assert (dones == died).all() and not (dones & ~active).any()
        assert not obs[~info["alive"]].any()
        seen_death |= bool(died.any())
    assert seen_death


def test_kill_credit_matches_killer_slots():
    env = SlitherEnv(32, max_snakes=8, min_snakes=8, config=CROWDED, seed=2)
    env.reset()
    rng = np.random.default_rng(2)
    total = 0
    for _ in range(80):
        _, _, _, info = env.step(rng.integers(0, 18, (32, 8)))
        killer = info["killer"]
        expect = np.zeros_like(info["kills"])
        e, k = np.nonzero(killer >= 0)
        np.add.at(expect, (e, killer[e, k]), 1)
        assert (info["kills"] == expect).all()
        total += int(expect.sum())
    assert total > 0


def test_same_seed_same_game():
    def run(seed):
        env = SlitherEnv(4, seed=seed)
        obs, _ = env.reset()
        for t in range(20):
            obs, _, _, _ = env.step(np.full((4, 12), t % 18))
        return obs
    np.testing.assert_array_equal(run(5), run(5))


def test_env_rejects_bad_actions():
    env = SlitherEnv(2, max_snakes=4, min_snakes=4, seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros((2, 3), int))
    with pytest.raises(ValueError):
        env.step(np.full((2, 4), 18))
    with pytest.raises(ValueError):
        env.step(np.full((2, 4), -1))


def test_env_rejects_respawn_faster_than_a_decision():
    with pytest.raises(ValueError):
        SlitherEnv(2, config=replace(SlitherConfig(), respawn_delay=3))
    with pytest.raises(ValueError):
        SlitherEnv(2, max_snakes=4, min_snakes=5)
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `uv run pytest tests/test_slither_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'snake_rl.slither.env'`

- [ ] **Step 3: Implement**

`snake_rl/slither/env.py`:

```python
"""SlitherEnv: many endless slither arenas stepped together, one decision (3 ticks) per step."""

from __future__ import annotations

import numpy as np

from snake_rl.slither.actions import NUM_ACTIONS, action_mask, decode_actions
from snake_rl.slither.config import SlitherConfig, u_layout
from snake_rl.slither.observations import build_observations, obs_size
from snake_rl.slither.rules import SlitherState, reset_arenas, step_tick


class SlitherEnv:
    def __init__(self, num_envs: int, max_snakes: int = 12, min_snakes: int = 6,
                 config: SlitherConfig | None = None, seed: int | None = None,
                 mass_scale: float = 0.01, death_penalty: float = 1.0, survival_bonus: float = 0.0):
        self.cfg = config or SlitherConfig()
        if not 1 <= min_snakes <= max_snakes:
            raise ValueError(f"need 1 <= min_snakes <= max_snakes, got {min_snakes}, {max_snakes}")
        if self.cfg.respawn_delay <= self.cfg.ticks_per_decision:
            raise ValueError("respawn_delay must exceed ticks_per_decision, or a snake could die "
                             "and respawn within one decision")
        self.state = SlitherState.empty(num_envs, max_snakes, self.cfg)
        self.min_snakes, self.max_snakes = min_snakes, max_snakes
        self.mass_scale, self.death_penalty, self.survival_bonus = mass_scale, death_penalty, survival_bonus
        self.rng = np.random.default_rng(seed)
        self.u_size = u_layout(max_snakes, self.cfg)["size"]
        self._needs_reset = True

    @property
    def num_envs(self) -> int:
        return self.state.num_envs

    @property
    def num_snakes(self) -> int:
        return self.state.num_snakes

    @property
    def obs_size(self) -> int:
        return obs_size(self.cfg)

    def reset(self):
        """Fresh arenas with min_snakes..max_snakes enabled slots each (fixed from now on)."""
        counts = self.rng.integers(self.min_snakes, self.max_snakes + 1, self.num_envs)
        reset_arenas(self.state, np.arange(self.num_snakes) < counts[:, None], self.rng)
        self._needs_reset = False
        zeros = np.zeros((self.num_envs, self.num_snakes))
        info = self._info(self.state.alive.copy(), zeros.astype(bool), np.full(zeros.shape, -1, np.int32),
                          zeros.astype(np.int32))
        return build_observations(self.state), info

    def step(self, actions):
        """One decision. Arenas are endless: dead snakes respawn inside the rules, and
        `info["active"]` marks the slots whose transition is real (alive at the start)."""
        if self._needs_reset:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, self.num_snakes):
            raise ValueError(f"actions must have shape {(self.num_envs, self.num_snakes)}, got {actions.shape}")
        if actions.min() < 0 or actions.max() >= NUM_ACTIONS:
            raise ValueError(f"actions must be in [0, {NUM_ACTIONS})")
        turn, boost = decode_actions(actions, self.cfg)
        active = self.state.alive.copy()
        mass_before = self.state.mass.copy()
        died = np.zeros(active.shape, bool)
        killer = np.full(active.shape, -1, np.int32)
        kills = np.zeros(active.shape, np.int32)
        for _ in range(self.cfg.ticks_per_decision):
            ev = step_tick(self.state, turn, boost, self.rng.random((self.num_envs, self.u_size)))
            killer = np.where(ev.died, ev.killer, killer)
            died |= ev.died
            kills += ev.kills
        gained = np.where(active, self.state.mass - mass_before, 0.0)
        rewards = (self.mass_scale * gained - self.death_penalty * died
                   + self.survival_bonus * (active & ~died)).astype(np.float32)
        return build_observations(self.state), rewards, died.copy(), self._info(active, died, killer, kills)

    def _info(self, active, died, killer, kills) -> dict:
        return {
            "active": active,
            "alive": self.state.alive.copy(),
            "enabled": self.state.enabled.copy(),
            "action_mask": action_mask(self.state),
            "mass": self.state.mass.copy(),
            "died": died,
            "killer": killer,
            "kills": kills,
        }
```

Because `respawn_delay > ticks_per_decision`, a snake that dies during a decision still
holds its death mass when the decision ends. That keeps `gained` correct for it.

Add `SlitherEnv` to `snake_rl/slither/__init__.py`: import it with
`from snake_rl.slither.env import SlitherEnv` and add `"SlitherEnv"` to `__all__`.

- [ ] **Step 4: Run the tests to check they pass**

Run: `uv run pytest tests/test_slither_env.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/slither/env.py snake_rl/slither/__init__.py tests/test_slither_env.py
git commit -m "Slither: SlitherEnv with per-life episodes and active masks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 7: Scripted bots

**Files:**
- Create: `snake_rl/slither/bots.py`, `tests/test_slither_bots.py`

**Interfaces:**
- Consumes:
  - `steer_bin` and `action_mask` (Task 5);
  - `OBS_CHANNELS` and `SIZE_SCALE` (Task 5);
  - `SlitherEnv` (Task 6).
- Produces:
  - `BOT_NAMES = ("wander", "forager", "hunter")`;
  - `SlitherBot(name, seed=None, cfg=None)`, a callable `(obs[..., 263], mask[..., 18]) -> int64 actions[...]`
    that works on any leading shape.

- [ ] **Step 1: Write the failing tests**

`tests/test_slither_bots.py`:

```python
import math

import numpy as np
import pytest

from slither_helpers import scene, snake, state_of, straight
from snake_rl.slither.actions import action_mask
from snake_rl.slither.bots import BOT_NAMES, SlitherBot
from snake_rl.slither.env import SlitherEnv
from snake_rl.slither.observations import build_observations


def act(name, snap, slot=0):
    state = state_of(snap)
    return int(SlitherBot(name, seed=0)(build_observations(state), action_mask(state))[0, slot])


def at(q, d):
    a = -math.pi + (q + 0.5) * 2 * math.pi / 32
    return d * math.cos(a), d * math.sin(a)


@pytest.mark.parametrize("name", BOT_NAMES)
def test_actions_are_valid_and_respect_the_mask(name):
    env = SlitherEnv(8, seed=0)
    obs, info = env.reset()
    for _ in range(5):
        actions = SlitherBot(name, seed=1)(obs, info["action_mask"])
        assert actions.shape == (8, 12) and actions.dtype == np.int64
        assert actions.min() >= 0 and actions.max() < 18
        assert not ((actions % 2 == 1) & ~info["action_mask"][..., 1]).any()
        obs, _, _, info = env.step(actions)


@pytest.mark.parametrize("name", BOT_NAMES)
def test_turns_hard_away_from_a_wall_ahead(name):
    assert act(name, scene([snake(straight(280.0, 0.0, 0.0, 40))])) // 2 in (0, 8)


def test_forager_turns_toward_food():
    food = [(*at(20, 15.0), 5.0), (*at(20, 18.0), 5.0)]
    action = act("forager", scene([snake(straight(0.0, 0.0, 0.0, 40))], food=food))
    assert action == 8 * 2  # hard left (sector 20 is about 51 degrees left), no boost


def test_hunter_boosts_at_a_smaller_head_ahead_and_forager_does_not():
    hx, hy = at(17, 20.0)
    me = snake(straight(0.0, 0.0, 0.0, 100), mass=40.0)
    prey = snake(straight(hx, hy, math.pi / 2, 40), heading=math.pi / 2, mass=10.0)
    assert act("hunter", scene([me, prey])) % 2 == 1
    assert act("forager", scene([me, prey])) % 2 == 0


def test_unknown_bot():
    with pytest.raises(ValueError):
        SlitherBot("nope")
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `uv run pytest tests/test_slither_bots.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'snake_rl.slither.bots'`

- [ ] **Step 3: Implement**

`snake_rl/slither/bots.py`:

```python
"""Scripted slither bots (spec §4). They read the same observation as the network, so they
run batched over any leading shape: bot(obs[..., 263], mask[..., 18]) -> actions[...].

Each bot scores the 32 sectors (food is good, danger is bad, turning costs a little, plus
noise) and steers toward the best one. The hunter also boosts to cut across the path of a
smaller snake whose head is ahead of it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snake_rl.slither.actions import steer_bin
from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.observations import OBS_CHANNELS, SIZE_SCALE

BOT_NAMES = ("wander", "forager", "hunter")
HUNT_LEAD = 20.0  # aim this far ahead of the prey's head, so we cross its path


@dataclass(frozen=True)
class BotStyle:
    food: float    # weight on a sector's food channel
    danger: float  # weight on a sector's danger (wall, enemy body, enemy head)
    turn: float    # cost per radian of turning
    noise: float   # random jitter on sector scores
    hunt: bool     # boost across smaller snakes' paths


STYLES = {
    "wander": BotStyle(food=0.0, danger=4.0, turn=0.3, noise=1.0, hunt=False),
    "forager": BotStyle(food=3.0, danger=4.0, turn=0.2, noise=0.05, hunt=False),
    "hunter": BotStyle(food=3.0, danger=4.0, turn=0.2, noise=0.05, hunt=True),
}


class SlitherBot:
    def __init__(self, name: str, seed: int | None = None, cfg: SlitherConfig | None = None):
        if name not in STYLES:
            raise ValueError(f"unknown bot {name!r}; choose from {BOT_NAMES}")
        self.name = name
        self.style = STYLES[name]
        self.cfg = cfg or SlitherConfig()
        self.rng = np.random.default_rng(seed)
        k = self.cfg.sectors
        self.phi = -np.pi + (np.arange(k) + 0.5) * 2.0 * np.pi / k  # sector centre angles

    def __call__(self, obs, mask) -> np.ndarray:
        obs = np.asarray(obs, np.float64)
        mask = np.asarray(mask, bool)
        k = self.cfg.sectors
        lead = obs.shape[:-1]
        sec = obs[..., :k * OBS_CHANNELS].reshape(*lead, k, OBS_CHANNELS)
        danger = np.maximum(np.maximum(sec[..., 0], sec[..., 1]), sec[..., 5])
        # Bodies are wide, so a sector is nearly as dangerous as its neighbours.
        danger = np.maximum(danger, 0.8 * np.maximum(np.roll(danger, 1, -1), np.roll(danger, -1, -1)))
        st = self.style
        score = (st.food * sec[..., 4] - st.danger * danger - st.turn * np.abs(self.phi)
                 + st.noise * self.rng.random(danger.shape))
        target = self.phi[np.argmax(score, axis=-1)]
        boost = np.zeros(lead, bool)
        if st.hunt:
            target, boost = self._hunt(obs, sec, target, mask)
        return (steer_bin(target, self.cfg) * 2 + boost).astype(np.int64)

    def _hunt(self, obs, sec, target, mask):
        k = self.cfg.sectors
        my_size = obs[..., k * OBS_CHANNELS + 1]
        prey = ((np.abs(self.phi) < np.pi / 3) & (sec[..., 5] > 0.5)
                & (sec[..., 2] < my_size[..., None]) & (sec[..., 0] < 0.5))
        has = prey.any(axis=-1)
        pick = np.argmax(np.where(prey, sec[..., 5], -1.0), axis=-1)[..., None]
        near = np.take_along_axis(sec[..., 5], pick, -1)[..., 0]
        size = np.take_along_axis(sec[..., 2], pick, -1)[..., 0]
        cos_h = np.take_along_axis(sec[..., 6], pick, -1)[..., 0]
        sin_h = np.take_along_axis(sec[..., 7], pick, -1)[..., 0]
        ang = self.phi[pick[..., 0]]
        dist = (1.0 - near) * self.cfg.vision + size * SIZE_SCALE
        aim = np.arctan2(dist * np.sin(ang) + HUNT_LEAD * sin_h, dist * np.cos(ang) + HUNT_LEAD * cos_h)
        return np.where(has, aim, target), has & mask[..., 1]
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `uv run pytest tests/test_slither_bots.py -q`
Expected: all pass.

If `test_hunter_boosts…` fails, print the observation's sector 17 channels 2 and 5 and
scalar 257 for the scene, and check the prey condition against them. The scene is
built so that enemy-head closeness is ≈0.73, the enemy is smaller, and the wall is far
away.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/slither/bots.py tests/test_slither_bots.py
git commit -m "Slither: wander, forager and hunter bots

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Throughput benchmark (and a grid pair search if it's needed)

**Files:**
- Create: `scripts/benchmark_slither.py`
- Conditional: Modify `snake_rl/slither/rules.py` (`near_pairs`); Create `tests/test_slither_pairs.py`

**Interfaces:**
- Consumes: `SlitherEnv` and `SlitherBot`.
- Produces:
  - `scripts/benchmark_slither.py` (`main(argv) -> float` snake-decisions/s).
  - Conditionally: `rules._near_pairs_dense`, `rules._near_pairs_grid`, with
    `near_pairs = _near_pairs_grid`.

- [ ] **Step 1: Write the benchmark**

`scripts/benchmark_slither.py`:

```python
"""Measure SlitherEnv throughput in snake decisions per second (spec §5 target: 20,000).

  uv run python scripts/benchmark_slither.py
  uv run python scripts/benchmark_slither.py --num-envs 256 --decisions 100

Every slot is driven by the forager bot, so bodies grow to realistic lengths during the
warm-up; only env.step is timed.
"""

import argparse
import time

from snake_rl.slither.bots import SlitherBot
from snake_rl.slither.env import SlitherEnv


def main(argv=None) -> float:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--max-snakes", type=int, default=12)
    p.add_argument("--min-snakes", type=int, default=6)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--decisions", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    env = SlitherEnv(args.num_envs, args.max_snakes, args.min_snakes, seed=args.seed)
    bot = SlitherBot("forager", seed=args.seed, cfg=env.cfg)
    obs, info = env.reset()
    for _ in range(args.warmup):
        obs, _, _, info = env.step(bot(obs, info["action_mask"]))
    snakes, spent = 0, 0.0
    for _ in range(args.decisions):
        actions = bot(obs, info["action_mask"])
        start = time.perf_counter()
        obs, _, _, info = env.step(actions)
        spent += time.perf_counter() - start
        snakes += int(info["active"].sum())
    alive = info["alive"]
    rate = snakes / spent
    print(f"{args.num_envs} arenas, {alive.sum(axis=1).mean():.1f} live snakes each, "
          f"mean mass {info['mass'][alive].mean():.1f}, mean length {env.state.length[alive].mean():.0f} points")
    print(f"{rate:,.0f} snake decisions/s (env only, {1000 * spent / args.decisions:.1f} ms per step)")
    return rate


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/benchmark_slither.py`
Expected: two lines ending in `… snake decisions/s (env only, … ms per step)`. Write the
number down.

**If the rate is ≥ 20,000, skip to Step 6.** Otherwise do Steps 3–5.

- [ ] **Step 3: Profile**

Run: `uv run python -m cProfile -s cumtime scripts/benchmark_slither.py --warmup 100 --decisions 20 2>&1 | head -45`

Expected: most of the time is in `near_pairs` (called from `_body_hits` and
`build_observations`) and in `ring_index`. If something else dominates, optimise that
instead, and keep the reference, parity and observation tests green.

- [ ] **Step 4: Write the failing pair-search test**

`tests/test_slither_pairs.py`:

```python
import numpy as np

from snake_rl.slither import rules


def test_grid_pair_search_finds_exactly_the_dense_pairs():
    rng = np.random.default_rng(0)
    n, s, p = 5, 7, 4000
    pe = rng.integers(0, n, p)
    px, py = rng.uniform(-300, 300, p), rng.uniform(-300, 300, p)
    heads = rng.uniform(-300, 300, (n, s, 2))
    live = rng.random((n, s)) < 0.8
    for reach in (5.0, 27.0, 75.0):
        want = list(zip(*rules._near_pairs_dense(pe, px, py, heads, live, reach)))
        got = list(zip(*rules._near_pairs_grid(pe, px, py, heads, live, reach)))
        assert got == want  # same pairs in the same (point, slot) order
    assert rules.near_pairs is rules._near_pairs_grid
```

Run: `uv run pytest tests/test_slither_pairs.py -q`
Expected: FAIL with `AttributeError: module 'snake_rl.slither.rules' has no attribute '_near_pairs_dense'`.

- [ ] **Step 5: Implement the grid search**

In `rules.py`, rename the existing `def near_pairs(...)` to `def _near_pairs_dense(...)`,
keeping its body. Then add:

```python
def _near_pairs_grid(pe, px, py, heads, live, reach):
    """Same pairs as _near_pairs_dense, found through a uniform grid with cells of side
    `reach`: a point can only be within `reach` of heads in its own or the 8 neighbouring
    cells. Returned in (point, slot) order, like the dense version."""
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if pe.size == 0 or not live.any():
        return empty
    he, hk = np.nonzero(live)
    hx, hy = heads[he, hk, 0], heads[he, hk, 1]
    lo_x, lo_y = min(px.min(), hx.min()), min(py.min(), hy.min())
    gx = np.floor((px - lo_x) / reach).astype(np.int64) + 1
    gy = np.floor((py - lo_y) / reach).astype(np.int64) + 1
    hgx = np.floor((hx - lo_x) / reach).astype(np.int64) + 1
    hgy = np.floor((hy - lo_y) / reach).astype(np.int64) + 1
    width = int(max(gx.max(), gy.max(), hgx.max(), hgy.max())) + 2
    key = (pe.astype(np.int64) * width + gx) * width + gy
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    qs, ks = [], []
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            hkey = (he * width + hgx + ox) * width + hgy + oy
            start = np.searchsorted(sorted_key, hkey, "left")
            count = np.searchsorted(sorted_key, hkey, "right") - start
            rep = np.repeat(np.arange(he.size), count)
            offset = np.arange(rep.size) - np.repeat(np.cumsum(count) - count, count)
            qs.append(order[start[rep] + offset])
            ks.append(hk[rep])
    q, k = np.concatenate(qs), np.concatenate(ks)
    dx = px[q] - heads[pe[q], k, 0]
    dy = py[q] - heads[pe[q], k, 1]
    keep = dx * dx + dy * dy < reach * reach
    q, k = q[keep], k[keep]
    o = np.lexsort((k, q))
    return q[o], k[o]


near_pairs = _near_pairs_grid
```

Because `near_pairs` keeps its name and its (point, slot) order, `_body_hits` and
`build_observations` need no changes.

Run: `uv run pytest tests/test_slither_pairs.py tests/test_slither_reference.py tests/test_slither_observations.py tests/test_slither_rules.py -q`
Expected: all pass.

Run: `uv run python scripts/benchmark_slither.py`

If the rate is still below 20,000, **stop and ask the user**. Report the profile and
the measured rate, and ask whether to accept it or to change approach (for example,
fewer arenas, or Numba). Do not lower the target silently.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark_slither.py
git add snake_rl/slither/rules.py tests/test_slither_pairs.py 2>/dev/null || true
git commit -m "Slither: throughput benchmark

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Put the measured rate in the commit body. Task 14 copies it into the README.

---

### Task 9: MLP actor-critic, policy wrapper and evaluation

**Files:**
- Create: `snake_rl/slither/model.py`, `snake_rl/slither/eval.py`
- Create: `tests/test_slither_model.py`, `tests/test_slither_eval.py`

**Interfaces:**
- Consumes:
  - `SlitherEnv`, `SlitherBot` and `NUM_ACTIONS`;
  - `snake_rl.eval.harness.bootstrap_mean_ci(values, rng) -> (lo, hi)`, which is
    imported and not modified.
- Produces:
  - `SlitherNet(obs_dim=263, hidden=256, actions=18)`, with `forward(obs) -> (logits (B,18), value (B,))`
    and `.config()`.
  - `masked_logits(logits, mask)`.
  - `load_slither_policy(path, device="cpu") -> SlitherNet` (in eval mode).
  - `SlitherPolicy(model, deterministic=True, seed=0)`, a callable
    `(obs[..., 263], mask[..., 18]) -> int64 actions[...]`.
  - `SlitherEvalResult`, with fields `policy, bot, arenas, decisions, mean_mass, mean_mass_ci, mass_share, mass_share_ci, kills, deaths, kills_per_death, kills_per_death_ci, mean_life`
    and `.to_dict()`.
  - `evaluate_vs_bot(policy, bot: str, arenas=64, decisions=1000, warmup=100, seed=0, cfg=None, snakes=8, name="policy") -> SlitherEvalResult`.

`snake_rl/slither/eval.py` must not import torch, so bot-vs-bot evaluation works
without the train group.

- [ ] **Step 1: Write the failing tests**

`tests/test_slither_model.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from snake_rl.slither.model import SlitherNet, SlitherPolicy, load_slither_policy, masked_logits  # noqa: E402


def test_shapes_and_config():
    net = SlitherNet()
    logits, value = net(torch.zeros(5, 263))
    assert logits.shape == (5, 18) and value.shape == (5,)
    assert net.config() == {"obs_dim": 263, "hidden": 256, "actions": 18}


def test_masked_logits_keeps_all_when_nothing_is_allowed():
    out = masked_logits(torch.zeros(2, 3), torch.tensor([[True, False, True], [False, False, False]]))
    assert out[0, 1] < -1e8 and (out[1] == 0).all()


def test_policy_respects_mask_and_leading_shape():
    torch.manual_seed(0)
    policy = SlitherPolicy(SlitherNet())
    obs = np.random.default_rng(0).random((4, 3, 263)).astype(np.float32)
    mask = np.ones((4, 3, 18), bool)
    mask[..., 1::2] = False
    actions = policy(obs, mask)
    assert actions.shape == (4, 3) and (actions % 2 == 0).all()


def test_save_and_load(tmp_path):
    net = SlitherNet(hidden=32)
    path = tmp_path / "net.pt"
    torch.save({"config": net.config(), "model": net.state_dict()}, path)
    loaded = load_slither_policy(path)
    x = torch.rand(3, 263)
    torch.testing.assert_close(loaded(x)[0], net(x)[0])
```

`tests/test_slither_eval.py`:

```python
from snake_rl.slither.bots import SlitherBot
from snake_rl.slither.eval import evaluate_vs_bot


def run(policy_bot, opponent, arenas=8, decisions=100, seed=3):
    return evaluate_vs_bot(SlitherBot(policy_bot, seed=seed + 10), opponent, arenas=arenas,
                           decisions=decisions, warmup=10, seed=seed, name=policy_bot)


def test_result_is_well_formed_and_deterministic():
    a, b = run("wander", "wander"), run("wander", "wander")
    assert a == b
    assert 0 <= a.mass_share <= 1 and a.mass_share_ci[0] <= a.mass_share <= a.mass_share_ci[1]
    assert a.mean_mass > 0 and a.mean_mass_ci[0] <= a.mean_mass <= a.mean_mass_ci[1]
    assert a.kills_per_death_ci[0] <= a.kills_per_death <= a.kills_per_death_ci[1]
    assert set(a.to_dict()) >= {"policy", "bot", "mass_share", "kills_per_death", "mean_life"}


def test_bot_strength_ordering():
    """Spec §4: hunter beats forager beats wander, measured as mass share above the fair 1/8."""
    assert run("forager", "wander", arenas=16, decisions=400).mass_share > 0.125
    assert run("hunter", "forager", arenas=16, decisions=400).mass_share > 0.125
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `uv run --group train pytest tests/test_slither_model.py tests/test_slither_eval.py -q`
Expected: FAIL with `ModuleNotFoundError` for `snake_rl.slither.model` and `snake_rl.slither.eval`.

- [ ] **Step 3: Implement**

`snake_rl/slither/model.py`:

```python
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
```

`snake_rl/slither/eval.py`:

```python
"""Evaluate a policy in slot 0 against copies of a scripted bot (spec §4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from snake_rl.eval.harness import bootstrap_mean_ci
from snake_rl.slither.bots import SlitherBot
from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.env import SlitherEnv


@dataclass
class SlitherEvalResult:
    policy: str
    bot: str
    arenas: int
    decisions: int
    mean_mass: float                     # slot 0's mass averaged over time (0 while dead)
    mean_mass_ci: tuple[float, float]
    mass_share: float                    # slot 0's share of all live snake mass; fair = 1/snakes
    mass_share_ci: tuple[float, float]
    kills: int
    deaths: int
    kills_per_death: float
    kills_per_death_ci: tuple[float, float]
    mean_life: float                     # decisions per completed life

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio_ci(num, den, rng, n_resamples=2000):
    """95% bootstrap interval (over arenas) of sum(num) / sum(den)."""
    idx = rng.integers(0, num.size, (n_resamples, num.size))
    ratios = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_vs_bot(policy, bot: str, arenas: int = 64, decisions: int = 1000, warmup: int = 100,
                    seed: int = 0, cfg: SlitherConfig | None = None, snakes: int = 8,
                    name: str = "policy") -> SlitherEvalResult:
    """Slot 0 plays `policy` (callable(obs, mask) -> actions over leading dims); the other
    slots play the bot named `bot`. Metrics are measured over `decisions` after `warmup`."""
    env = SlitherEnv(arenas, max_snakes=snakes, min_snakes=snakes, config=cfg, seed=seed)
    bots = SlitherBot(bot, seed=seed + 1, cfg=env.cfg)
    obs, info = env.reset()
    mass_sum, total_sum = np.zeros(arenas), np.zeros(arenas)
    kills, deaths = np.zeros(arenas, np.int64), np.zeros(arenas, np.int64)
    life = np.zeros(arenas, np.int64)
    life_sum = life_count = 0
    for t in range(warmup + decisions):
        mask = info["action_mask"]
        actions = bots(obs, mask)
        actions[:, 0] = policy(obs[:, 0], mask[:, 0])
        obs, _, dones, info = env.step(actions)
        if t < warmup:
            continue
        mass = np.where(info["alive"], info["mass"], 0.0)
        mass_sum += mass[:, 0]
        total_sum += mass.sum(axis=1)
        kills += info["kills"][:, 0]
        deaths += dones[:, 0]
        life += info["active"][:, 0]
        ended = dones[:, 0]
        life_sum += int(life[ended].sum())
        life_count += int(ended.sum())
        life[ended] = 0
    rng = np.random.default_rng(seed)
    per_arena_mass = mass_sum / decisions
    share = mass_sum / np.maximum(total_sum, 1e-9)
    return SlitherEvalResult(
        policy=name, bot=bot, arenas=arenas, decisions=decisions,
        mean_mass=float(per_arena_mass.mean()), mean_mass_ci=bootstrap_mean_ci(per_arena_mass, rng),
        mass_share=float(share.mean()), mass_share_ci=bootstrap_mean_ci(share, rng),
        kills=int(kills.sum()), deaths=int(deaths.sum()),
        kills_per_death=float(kills.sum() / max(int(deaths.sum()), 1)),
        kills_per_death_ci=_ratio_ci(kills, deaths, rng),
        mean_life=float(life_sum / max(life_count, 1)),
    )
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `uv run --group train pytest tests/test_slither_model.py tests/test_slither_eval.py -q`
Expected: all pass.

If `test_bot_strength_ordering` fails, tune the bots rather than the test: either
`STYLES`, or `HUNT_LEAD` and the prey thresholds in `bots.py`. To see where things stand,
print `run(...).mass_share` for `forager` vs `wander` and `hunter` vs `forager`. If no
reasonable tuning gets hunter above 1/8 against foragers, stop and report it. Never
loosen the 0.125 bar. Re-run `tests/test_slither_bots.py` after any tuning.

- [ ] **Step 5: Commit**

```bash
git add snake_rl/slither/model.py snake_rl/slither/eval.py tests/test_slither_model.py tests/test_slither_eval.py snake_rl/slither/bots.py
git commit -m "Slither: MLP actor-critic, policy wrapper and bot evaluation with CIs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: PPO self-play trainer

**Files:**
- Create: `scripts/train_slither.py`, `tests/test_train_slither.py`

**Interfaces:**
- Consumes:
  - `SlitherEnv` (`step` returns `info["active"]`, `info["action_mask"]`, `info["kills"]`, `info["mass"]`);
  - `SlitherNet`, `SlitherPolicy` and `masked_logits`;
  - `evaluate_vs_bot`.
- Produces:
  - `scripts/train_slither.py` with `main(argv=None)`;
  - run directory layout `runs/slither/{config.json, metrics.csv, latest.pt, best.pt, checkpoints/iter_XXXXX.pt}`;
  - checkpoints of the form `{"config", "model", "iteration", "env_steps", "metrics"}`;
  - `metrics.csv` columns that include `mass_share_vs_hunter` and `mass_share_vs_forager`,
    which Task 11 reads.

- [ ] **Step 1: Write the failing smoke test**

`tests/test_train_slither.py`:

```python
import csv
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("train_slither", ROOT / "scripts" / "train_slither.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_iterations_write_checkpoints_and_metrics(tmp_path):
    run = tmp_path / "run"
    load_script().main(["--run-dir", str(run), "--num-envs", "4", "--max-snakes", "4", "--min-snakes", "3",
                        "--rollout", "8", "--iterations", "2", "--eval-every", "2", "--eval-arenas", "2",
                        "--eval-decisions", "10", "--minibatch", "32", "--hidden", "32", "--device", "cpu"])
    for name in ("config.json", "metrics.csv", "latest.pt", "best.pt", "checkpoints/iter_00002.pt"):
        assert (run / name).exists(), name
    rows = list(csv.DictReader((run / "metrics.csv").open()))
    assert len(rows) == 2 and rows[-1]["mass_share_vs_hunter"] != ""
```

- [ ] **Step 2: Run the test to check it fails**

Run: `uv run --group train pytest tests/test_train_slither.py -q`
Expected: FAIL with `FileNotFoundError` for `scripts/train_slither.py`.

- [ ] **Step 3: Implement**

`scripts/train_slither.py`:

```python
"""PPO self-play for the slither mode (spec docs/superpowers/specs/2026-10-15-slither-mode-design.md §4).

Every arena has 6-12 snakes. Each life is played either by the learner or, with
probability --pool-prob, by a frozen snapshot from the opponent pool; only learner lives
produce training samples. A slot's owner is redrawn when its snake dies.

  uv run --group train python scripts/train_slither.py --device mps --hours 2.5
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from snake_rl.slither.actions import NUM_ACTIONS
from snake_rl.slither.env import SlitherEnv
from snake_rl.slither.eval import evaluate_vs_bot
from snake_rl.slither.model import SlitherNet, SlitherPolicy, masked_logits

EVAL_BOTS = ("forager", "hunter")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/slither")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--max-snakes", type=int, default=12)
    p.add_argument("--min-snakes", type=int, default=6)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--iterations", type=int, default=100_000)
    p.add_argument("--hours", type=float, default=None, help="stop after this much wall time")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--survival-bonus", type=float, default=0.0)
    p.add_argument("--pool-prob", type=float, default=0.3)
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-arenas", type=int, default=32)
    p.add_argument("--eval-decisions", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def evaluate(model, args, seed):
    """Learner in slot 0 against 7 copies of each evaluation bot."""
    policy = SlitherPolicy(model)
    out = {}
    for bot in EVAL_BOTS:
        r = evaluate_vs_bot(policy, bot, arenas=args.eval_arenas, decisions=args.eval_decisions,
                            warmup=min(100, args.eval_decisions), seed=seed)
        out[f"mass_share_vs_{bot}"] = r.mass_share
        out[f"kpd_vs_{bot}"] = r.kills_per_death
        out[f"mass_vs_{bot}"] = r.mean_mass
    return out


def save_checkpoint(path, model, iteration, env_steps, metrics=None):
    torch.save({"config": model.config(), "model": model.state_dict(), "iteration": iteration,
                "env_steps": env_steps, "metrics": metrics or {}}, path)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    n, s, t_len = args.num_envs, args.max_snakes, args.rollout
    env = SlitherEnv(n, max_snakes=s, min_snakes=args.min_snakes, seed=args.seed,
                     survival_bonus=args.survival_bonus)
    d = env.obs_size
    model = SlitherNet(d, args.hidden).to(device)
    opponent = SlitherNet(d, args.hidden).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    pool = [copy.deepcopy(model.state_dict())]

    obs, info = env.reset()
    learner = rng.random((n, s)) >= args.pool_prob

    obs_buf = np.zeros((t_len, n, s, d), np.float16)
    mask_buf = np.zeros((t_len, n, s, NUM_ACTIONS), bool)
    act_buf = np.zeros((t_len, n, s), np.int64)
    logp_buf = np.zeros((t_len, n, s), np.float32)
    val_buf = np.zeros((t_len, n, s), np.float32)
    rew_buf = np.zeros((t_len, n, s), np.float32)
    done_buf = np.zeros((t_len, n, s), bool)
    valid_buf = np.zeros((t_len, n, s), bool)

    metrics_path = run_dir / "metrics.csv"
    fields = ["iteration", "env_steps", "hours", "samples", "deaths", "kills", "mean_reward",
              "mean_mass", "policy_loss", "value_loss", "entropy", "approx_kl"]
    fields += [f"{m}_vs_{bot}" for bot in EVAL_BOTS for m in ("mass_share", "kpd", "mass")]
    with metrics_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    start = time.time()
    env_steps = 0
    best = -1.0
    for iteration in range(1, args.iterations + 1):
        opponent.load_state_dict(pool[rng.integers(len(pool))])
        deaths = kills = 0

        model.eval()
        for t in range(t_len):
            mask = info["action_mask"]
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.reshape(n * s, d)).to(device)
                mask_t = torch.from_numpy(mask.reshape(n * s, NUM_ACTIONS)).to(device)
                logits, value = model(obs_t)
                dist = Categorical(logits=masked_logits(logits, mask_t))
                action = dist.sample()
                logp = dist.log_prob(action)
                actions = action.cpu().numpy().reshape(n, s)
                opp = np.flatnonzero(~learner.reshape(-1))
                if opp.size:
                    opp_t = torch.from_numpy(opp).to(device)
                    o_logits, _ = opponent(obs_t[opp_t])
                    o_action = Categorical(logits=masked_logits(o_logits, mask_t[opp_t])).sample()
                    actions.reshape(-1)[opp] = o_action.cpu().numpy()

            obs_buf[t] = obs
            mask_buf[t] = mask
            act_buf[t] = actions
            logp_buf[t] = logp.cpu().numpy().reshape(n, s)
            val_buf[t] = value.cpu().numpy().reshape(n, s)

            obs, rewards, dones, info = env.step(actions)
            rew_buf[t] = rewards
            done_buf[t] = dones
            valid_buf[t] = info["active"] & learner
            deaths += int((dones & learner).sum())
            kills += int((info["kills"] * learner).sum())
            env_steps += n
            died = np.flatnonzero(dones.reshape(-1))
            if died.size:  # each new life gets a fresh owner
                learner.reshape(-1)[died] = rng.random(died.size) >= args.pool_prob

        with torch.no_grad():
            _, next_value = model(torch.from_numpy(obs.reshape(n * s, d)).to(device))
            next_value = next_value.cpu().numpy().reshape(n, s)
        adv = np.zeros_like(rew_buf)
        last = np.zeros((n, s), np.float32)
        for t in reversed(range(t_len)):
            nv = next_value if t == t_len - 1 else val_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * nv * nonterminal - val_buf[t]
            last = delta + args.gamma * args.lam * nonterminal * last
            adv[t] = last
        returns = adv + val_buf

        valid = valid_buf.reshape(-1)
        b_obs = torch.from_numpy(obs_buf.reshape(-1, d)[valid].astype(np.float32)).to(device)
        b_mask = torch.from_numpy(mask_buf.reshape(-1, NUM_ACTIONS)[valid]).to(device)
        b_act = torch.from_numpy(act_buf.reshape(-1)[valid]).to(device)
        b_logp = torch.from_numpy(logp_buf.reshape(-1)[valid]).to(device)
        b_adv = torch.from_numpy(adv.reshape(-1)[valid]).to(device)
        b_ret = torch.from_numpy(returns.reshape(-1)[valid]).to(device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        model.train()
        samples = b_obs.shape[0]
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        for _ in range(args.epochs):
            perm = torch.randperm(samples, device=device)
            for lo in range(0, samples, args.minibatch):
                idx = perm[lo:lo + args.minibatch]
                logits, value = model(b_obs[idx])
                dist = Categorical(logits=masked_logits(logits, b_mask[idx]))
                logp = dist.log_prob(b_act[idx])
                ratio = torch.exp(logp - b_logp[idx])
                pg = -torch.min(ratio * b_adv[idx],
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[idx]).mean()
                v_loss = 0.5 * ((value - b_ret[idx]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                with torch.no_grad():
                    stats["approx_kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
                stats["policy_loss"].append(pg.item())
                stats["value_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.item())

        hours = (time.time() - start) / 3600
        valid_rew = rew_buf[valid_buf]
        row = {
            "iteration": iteration, "env_steps": env_steps, "hours": round(hours, 4),
            "samples": samples, "deaths": deaths, "kills": kills,
            "mean_reward": round(float(valid_rew.mean()), 5) if valid_rew.size else "",
            "mean_mass": round(float(info["mass"][info["alive"] & learner].mean()), 2)
            if (info["alive"] & learner).any() else "",
            **{k: round(float(np.mean(v)), 5) for k, v in stats.items()},
        }

        if iteration % args.snapshot_every == 0:
            pool.append(copy.deepcopy(model.state_dict()))
            pool = pool[-args.pool_size:]

        stop = args.hours is not None and hours >= args.hours
        if iteration % args.eval_every == 0 or stop or iteration == args.iterations:
            model.eval()
            row.update({k: round(v, 4) for k, v in evaluate(model, args, iteration).items()})
            save_checkpoint(run_dir / "checkpoints" / f"iter_{iteration:05d}.pt", model, iteration, env_steps, row)
            save_checkpoint(run_dir / "latest.pt", model, iteration, env_steps, row)
            if row["mass_share_vs_hunter"] > best:
                best = row["mass_share_vs_hunter"]
                save_checkpoint(run_dir / "best.pt", model, iteration, env_steps, row)
            print(f"[eval] iter {iteration} steps {env_steps:,} mass share vs hunter "
                  f"{row['mass_share_vs_hunter']:.3f} vs forager {row['mass_share_vs_forager']:.3f} "
                  f"kpd vs hunter {row['kpd_vs_hunter']:.2f}", flush=True)

        with metrics_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"iter {iteration} steps {env_steps:,} {hours:.2f}h samples {samples} deaths {deaths} "
              f"kills {kills} reward {row['mean_reward']} mass {row['mean_mass']} "
              f"ent {row['entropy']:.3f} kl {row['approx_kl']:.4f}", flush=True)
        if stop:
            break


if __name__ == "__main__":
    main()
```

`env_steps` counts arena-decisions (`n` per step), just as the grid trainer counts
arena-turns.

- [ ] **Step 4: Run the smoke test to check it passes**

Run: `uv run --group train pytest tests/test_train_slither.py -q`
Expected: 1 passed.

- [ ] **Step 5: Short real run on MPS**

Run: `uv run --group train python scripts/train_slither.py --run-dir /tmp/slither-smoke --num-envs 64 --iterations 30 --eval-every 15 --eval-arenas 8 --eval-decisions 200 --device mps`

Expected:
- 30 `iter …` lines and two `[eval]` lines;
- no NaN in loss or entropy;
- `approx_kl` mostly below 0.05;
- `mean_mass` / `reward` not collapsing.

It's fine if the network doesn't improve over 30 iterations. Note the seconds per
iteration; Task 14 uses it to size the main run.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_slither.py tests/test_train_slither.py
git commit -m "Slither: PPO self-play trainer with a per-slot opponent pool

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Final evaluation CLI and web asset export

**Files:**
- Create: `scripts/eval_slither.py`, `scripts/build_slither_assets.py`, `tests/test_slither_assets.py`

**Interfaces:**
- Consumes: `evaluate_vs_bot`, `SlitherBot`, `load_slither_policy`, `SlitherPolicy`,
  `SlitherConfig`, and `metrics.csv` from Task 10.
- Produces:
  - `results/slither_eval.json`: `{"policy": str, "snakes": 8, "results": {bot: SlitherEvalResult.to_dict()}}`.
  - `web/slither/model.onnx`: input `obs (batch, 263)`, outputs `logits (batch, 18)` and `value (batch)`.
  - `web/slither/stats.json`: `{"final_eval": {bot: {...}} | null, "curve": [{"env_steps", "hours", "mass_share_vs_hunter", "mass_share_vs_forager"}], "checkpoint": {...}, "config": {...}}`.
  - `build_slither_assets.export_onnx(model, path)`.

- [ ] **Step 1: Write the failing test**

`tests/test_slither_assets.py`:

```python
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from snake_rl.slither.model import SlitherNet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onnx_export_matches_pytorch(tmp_path):
    net = SlitherNet(hidden=32).eval()
    path = tmp_path / "m.onnx"
    load("build_slither_assets").export_onnx(net, path)
    x = np.random.default_rng(0).random((7, 263)).astype(np.float32)
    logits, value = ort.InferenceSession(str(path)).run(None, {"obs": x})
    with torch.no_grad():
        want_logits, want_value = net(torch.from_numpy(x))
    np.testing.assert_allclose(logits, want_logits.numpy(), atol=1e-4)
    np.testing.assert_allclose(value, want_value.numpy(), atol=1e-4)


def test_eval_cli_bot_vs_bot(tmp_path):
    out = tmp_path / "e.json"
    load("eval_slither").main(["bot:forager", "--bots", "wander", "--arenas", "4", "--decisions", "30",
                               "--warmup", "5", "--out", str(out)])
    import json
    data = json.loads(out.read_text())
    assert data["policy"] == "bot:forager" and 0 <= data["results"]["wander"]["mass_share"] <= 1
```

- [ ] **Step 2: Run the test to check it fails**

Run: `uv run --group train pytest tests/test_slither_assets.py -q`
Expected: FAIL with `FileNotFoundError` for the scripts.

- [ ] **Step 3: Implement**

`scripts/eval_slither.py`:

```python
"""Final evaluation of a slither policy against each scripted bot (spec §4).

  uv run --group train python scripts/eval_slither.py runs/slither/best.pt --arenas 300 --decisions 3000
  uv run python scripts/eval_slither.py bot:hunter --bots wander forager --arenas 100

The policy plays slot 0 of 8; the other 7 slots are copies of the bot. Writes JSON with
mass share (fair share is 1/8), mean mass, kills per death and mean life, with 95% CIs.
"""

import argparse
import json
from pathlib import Path

from snake_rl.slither.bots import BOT_NAMES, SlitherBot
from snake_rl.slither.eval import evaluate_vs_bot

SNAKES = 8


def make_policy(spec, seed):
    if spec.startswith("bot:"):
        return SlitherBot(spec[4:], seed=seed)
    from snake_rl.slither.model import SlitherPolicy, load_slither_policy  # needs torch
    return SlitherPolicy(load_slither_policy(spec))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy", help="checkpoint path, or bot:<name>")
    p.add_argument("--bots", nargs="+", default=list(BOT_NAMES), choices=BOT_NAMES)
    p.add_argument("--arenas", type=int, default=300)
    p.add_argument("--decisions", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default="results/slither_eval.json")
    args = p.parse_args(argv)

    results = {}
    for i, bot in enumerate(args.bots):
        r = evaluate_vs_bot(make_policy(args.policy, args.seed + 100 + i), bot, arenas=args.arenas,
                            decisions=args.decisions, warmup=args.warmup, seed=args.seed + i,
                            snakes=SNAKES, name=args.policy)
        results[bot] = r.to_dict()
        print(f"vs 7x {bot:8s} mass share {r.mass_share:.3f} [{r.mass_share_ci[0]:.3f}, {r.mass_share_ci[1]:.3f}] "
              f"(fair {1 / SNAKES:.3f})  mass {r.mean_mass:.1f}  kills/death {r.kills_per_death:.2f} "
              f"[{r.kills_per_death_ci[0]:.2f}, {r.kills_per_death_ci[1]:.2f}]  life {r.mean_life:.0f} decisions",
              flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": args.policy, "snakes": SNAKES, "results": results}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

`scripts/build_slither_assets.py`:

```python
"""Export a trained slither run for the browser: web/slither/model.onnx and web/slither/stats.json.

  uv run --group train python scripts/build_slither_assets.py --run-dir runs/slither \
      --final-eval results/slither_eval.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.model import load_slither_policy

ROOT = Path(__file__).resolve().parents[1]


def export_onnx(model, path):
    dummy = torch.zeros(1, model.obs_dim)
    torch.onnx.export(model, (dummy,), str(path), input_names=["obs"], output_names=["logits", "value"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, dynamo=False)
    sample = torch.rand(16, model.obs_dim)
    with torch.no_grad():
        want_logits, want_value = model(sample)
    got_logits, got_value = ort.InferenceSession(str(path)).run(None, {"obs": sample.numpy()})
    np.testing.assert_allclose(got_logits, want_logits.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(got_value, want_value.numpy(), rtol=1e-3, atol=1e-3)


def curve(metrics_path):
    points = []
    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            if row["mass_share_vs_hunter"]:
                points.append({"env_steps": int(row["env_steps"]), "hours": float(row["hours"]),
                               "mass_share_vs_hunter": float(row["mass_share_vs_hunter"]),
                               "mass_share_vs_forager": float(row["mass_share_vs_forager"])})
    return points


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/slither")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--final-eval", help="results JSON from scripts/eval_slither.py")
    p.add_argument("--out-dir", default=str(ROOT / "web" / "slither"))
    args = p.parse_args(argv)

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_slither_policy(run_dir / args.checkpoint)
    export_onnx(model, out_dir / "model.onnx")
    checkpoint = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=True)
    final = json.loads(Path(args.final_eval).read_text())["results"] if args.final_eval else None
    stats = {
        "final_eval": final,
        "curve": curve(run_dir / "metrics.csv"),
        "checkpoint": {"iteration": checkpoint["iteration"], "env_steps": checkpoint["env_steps"],
                       "metrics": checkpoint["metrics"]},
        "config": SlitherConfig().to_dict(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out_dir / 'model.onnx'} and {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `uv run --group train pytest tests/test_slither_assets.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_slither.py scripts/build_slither_assets.py tests/test_slither_assets.py
git commit -m "Slither: final evaluation CLI and ONNX/stats export for the browser

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 12: JavaScript rules and observation, with a parity test

**Files:**
- Create: `web/slither/rules.js`, `web/slither/observe.js`, `web/slither/test/parity.mjs`, `tests/test_slither_js_parity.py`

**Interfaces:**
- Consumes:
  - `tests/slither_reference.py`: both JS files are line-by-line ports of it;
  - `snapshot`, `step_tick`, `build_observations` and `random_ticks`/`SMALL` for the fixtures.
- Produces (ES modules):
  - `rules.js`:
    - `CONFIG` (frozen; equal to `SlitherConfig().to_dict()`, key for key);
    - `NUM_ACTIONS`;
    - `targetLength(mass, cfg)`, `bodyRadius(mass, cfg)`, `uLayout(slots, cfg)`;
    - `randomU(slots, cfg, random = Math.random) -> Float64Array`;
    - `decodeAction(a, cfg) -> {turn, boost}`;
    - `stepTick(game, turns, boosts, u, cfg) -> {died, killer, kills, eaten}`, which
      mutates `game`;
    - `respawnSlot(game, k, u, cfg)`;
    - `newArena(slots, enabledCount, cfg = CONFIG, random = Math.random) -> game`.
  - `observe.js`:
    - `OBS_CHANNELS`, `NUM_SCALARS`;
    - `obsSize(cfg)`;
    - `observe(game, k, cfg) -> Float64Array(263)`;
    - `actionMask(game, k, cfg) -> boolean[18]`.
  - A game object has the snapshot format from Task 1, with snake_case keys
    (`boost_ticks`).

- [ ] **Step 1: Write the failing parity test**

`tests/test_slither_js_parity.py`:

```python
"""web/slither/rules.js and observe.js must reproduce the Python engine (spec §5): positions
within 1e-9, discrete events exactly, observations within 1e-6. Fixtures use fixed seeds,
so the test is deterministic; a failure is investigated, not retried."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from slither_helpers import SMALL, random_ticks
from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.observations import build_observations
from snake_rl.slither.rules import snapshot, step_tick

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def make_cases(cfg, num_envs, slots, ticks, seed, every):
    cases = []
    for t, (state, turn, boost, u) in enumerate(random_ticks(cfg, num_envs, slots, ticks, seed)):
        record = t % every == 0
        if record:
            before = [snapshot(state, e) for e in range(num_envs)]
            obs = build_observations(state)
        ev = step_tick(state, turn, boost, u)
        if record:
            for e in range(num_envs):
                cases.append({
                    "before": before[e], "after": snapshot(state, e),
                    "turns": turn[e].tolist(), "boosts": boost[e].tolist(), "u": u[e].tolist(),
                    "obs": obs[e].astype(float).tolist(),
                    "events": {"died": ev.died[e].tolist(), "killer": ev.killer[e].tolist(),
                               "kills": ev.kills[e].tolist(), "eaten": ev.eaten[e].tolist()},
                })
    return cases


def test_js_matches_python(tmp_path):
    default = SlitherConfig()
    fixtures = {
        "defaults": default.to_dict(),
        "groups": [
            {"config": default.to_dict(), "cases": make_cases(default, 3, 12, 240, 0, 8)},
            {"config": SMALL.to_dict(), "cases": make_cases(SMALL, 4, 8, 300, 1, 6)},
        ],
    }
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(fixtures))
    result = subprocess.run(["node", str(ROOT / "web" / "slither" / "test" / "parity.mjs"), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert " 0 failures" in result.stdout
```

`web/slither/test/parity.mjs`:

```js
// Replays Python-generated slither ticks through rules.js and observe.js and checks every field.
// Usage: node web/slither/test/parity.mjs fixtures.json
import { readFileSync } from "node:fs";
import { CONFIG, stepTick } from "../rules.js";
import { observe } from "../observe.js";

const { defaults, groups } = JSON.parse(readFileSync(process.argv[2], "utf8"));
let failures = 0;
let total = 0;
const near = (a, b, tol) => Math.abs(a - b) <= tol;

function fail(where, what, got, want) {
  failures++;
  if (failures <= 10) console.error(`${where}: ${what}\n  got  ${JSON.stringify(got)}\n  want ${JSON.stringify(want)}`);
}

// CONFIG must equal the Python defaults, key for key.
for (const key of new Set([...Object.keys(CONFIG), ...Object.keys(defaults)])) {
  if (CONFIG[key] !== defaults[key]) fail("CONFIG", key, CONFIG[key], defaults[key]);
}

function compareGame(where, got, want) {
  if (got.tick !== want.tick) fail(where, "tick", got.tick, want.tick);
  got.snakes.forEach((s, k) => {
    const w = want.snakes[k];
    for (const key of ["alive", "enabled", "boosting", "boost_ticks", "respawn"]) {
      if (s[key] !== w[key]) fail(where, `snake ${k} ${key}`, s[key], w[key]);
    }
    for (const key of ["heading", "mass"]) {
      if (!near(s[key], w[key], 1e-9)) fail(where, `snake ${k} ${key}`, s[key], w[key]);
    }
    if (s.body.length !== w.body.length) {
      fail(where, `snake ${k} body length`, s.body.length, w.body.length);
      return;
    }
    for (let p = 0; p < s.body.length; p++) {
      if (!near(s.body[p][0], w.body[p][0], 1e-9) || !near(s.body[p][1], w.body[p][1], 1e-9)) {
        fail(where, `snake ${k} point ${p}`, s.body[p], w.body[p]);
        break;
      }
    }
  });
  got.food.forEach((item, f) => {
    const w = want.food[f];
    if ((item === null) !== (w === null)) return fail(where, `food ${f}`, item, w);
    if (item !== null && (!near(item[0], w[0], 1e-9) || !near(item[1], w[1], 1e-9)
        || !near(item[2], w[2], 1e-9) || item[3] !== w[3])) fail(where, `food ${f}`, item, w);
  });
}

groups.forEach((group, g) => {
  const cfg = group.config;
  group.cases.forEach((c, i) => {
    const where = `group ${g} case ${i}`;
    total++;
    const game = structuredClone(c.before);
    game.snakes.forEach((_, k) => {
      const obs = observe(game, k, cfg);
      const want = c.obs[k];
      for (let j = 0; j < want.length; j++) {
        if (!near(obs[j], want[j], 1e-6)) { fail(where, `obs snake ${k} index ${j}`, obs[j], want[j]); break; }
      }
    });
    const ev = stepTick(game, c.turns, c.boosts, c.u, cfg);
    for (const key of ["died", "killer", "kills"]) {
      if (JSON.stringify(ev[key]) !== JSON.stringify(c.events[key])) fail(where, key, ev[key], c.events[key]);
    }
    ev.eaten.forEach((v, k) => { if (!near(v, c.events.eaten[k], 1e-9)) fail(where, `eaten ${k}`, v, c.events.eaten[k]); });
    compareGame(where, game, c.after);
  });
});

console.log(`${total} cases, ${failures} failures`);
process.exit(failures ? 1 : 0);
```

- [ ] **Step 2: Run the test to check it fails**

Run: `uv run pytest tests/test_slither_js_parity.py -q`
Expected: FAIL, because Node can't find `../rules.js` (`ERR_MODULE_NOT_FOUND`).

- [ ] **Step 3: Write `web/slither/rules.js`**

```js
// Slither rules for the browser: a line-by-line port of tests/slither_reference.py
// (spec §2). tests/test_slither_js_parity.py replays Python-generated ticks through this
// file, so keep every expression in the same form and order as the Python.
//
// A game is a plain object in the snapshot format of snake_rl.slither.rules.snapshot:
//   { slots, tick, snakes: [{ body: [[x, y], ...] head first, heading, mass, alive,
//     enabled, boosting, boost_ticks, respawn }], food: [null | [x, y, value, natural]] }

export const CONFIG = Object.freeze({
  arena_radius: 300, tick_hz: 30, ticks_per_decision: 3, max_turn: 0.15,
  start_mass: 10, max_mass: 500, base_length: 20, length_per_mass: 2,
  base_radius: 2, radius_per_sqrt_mass: 0.5, body_capacity: 1024,
  boost_min_mass: 15, boost_cost: 0.2, boost_drop_every: 5, boost_drop_value: 0.8,
  eat_margin: 3, food_capacity: 1024, natural_food_target: 200, natural_food_per_tick: 2,
  natural_food_value: 1, food_spawn_margin: 10, death_drop_fraction: 0.8,
  death_drop_max_pellets: 64, death_drop_jitter: 2, respawn_delay: 30,
  respawn_candidates: 16, respawn_margin: 60, respawn_clearance: 40,
  vision: 60, sectors: 32, neck: 10,
});

export const NUM_ACTIONS = 18;

export const targetLength = (mass, cfg) => Math.floor(cfg.base_length + cfg.length_per_mass * mass);
export const bodyRadius = (mass, cfg) => cfg.base_radius + cfg.radius_per_sqrt_mass * Math.sqrt(mass);

export function uLayout(slots, cfg) {
  const drops = slots * (2 * cfg.respawn_candidates + 1);
  const food = drops + slots * cfg.death_drop_max_pellets * 2;
  return { respawn: 0, drops, food, size: food + 2 * cfg.natural_food_per_tick };
}

export function randomU(slots, cfg, random = Math.random) {
  const u = new Float64Array(uLayout(slots, cfg).size);
  for (let i = 0; i < u.length; i++) u[i] = random();
  return u;
}

// action = steer * 2 + boost; steering bin b turns max_turn * (b - 4) / 4 per tick.
export function decodeAction(action, cfg) {
  const steer = Math.floor(action / 2);
  return { turn: cfg.max_turn * (steer - 4) / 4, boost: action % 2 === 1 };
}

export function stepTick(g, turns, boosts, u, cfg) {
  const snakes = g.snakes, food = g.food, count = snakes.length;
  const lay = uLayout(count, cfg);
  const alive = snakes.map((s) => s.alive);

  // 1. steer
  snakes.forEach((s, k) => {
    if (alive[k]) s.heading = s.heading + Math.min(Math.max(turns[k], -cfg.max_turn), cfg.max_turn);
  });

  // 2. boost, move, trim
  const boostDrops = [];
  snakes.forEach((s, k) => {
    const b = alive[k] && Boolean(boosts[k]) && s.mass >= cfg.boost_min_mass;
    if (b) s.mass = s.mass - cfg.boost_cost;
    s.boost_ticks = b ? s.boost_ticks + 1 : 0;
    s.boosting = b;
    if (!alive[k]) return;
    for (let step = 0; step < (b ? 2 : 1); step++) {
      const [hx, hy] = s.body[0];
      s.body.unshift([hx + Math.cos(s.heading), hy + Math.sin(s.heading)]);
    }
    s.body.length = Math.min(s.body.length, targetLength(s.mass, cfg));
    if (b && s.boost_ticks % cfg.boost_drop_every === 0) {
      const [tx, ty] = s.body[s.body.length - 1];
      boostDrops.push([tx, ty, cfg.boost_drop_value, false]);
    }
  });

  // 3. deaths
  const radius = snakes.map((s) => bodyRadius(s.mass, cfg));
  const died = Array(count).fill(false), killer = Array(count).fill(-1), kills = Array(count).fill(0);
  for (let i = 0; i < count; i++) {
    if (!alive[i]) continue;
    const [hx, hy] = snakes[i].body[0];
    const wall = Math.sqrt(hx * hx + hy * hy) + radius[i] > cfg.arena_radius;
    let headOn = false, first = -1;
    for (let j = 0; j < count; j++) {
      if (j === i || !alive[j]) continue;
      const t = radius[i] + radius[j];
      const [ox, oy] = snakes[j].body[0];
      let dx = hx - ox, dy = hy - oy;
      if (dx * dx + dy * dy < t * t) headOn = true;
      if (first < 0) {
        const body = snakes[j].body;
        for (let p = 1; p < body.length; p++) {
          dx = body[p][0] - hx;
          dy = body[p][1] - hy;
          if (dx * dx + dy * dy < t * t) { first = j; break; }
        }
      }
    }
    died[i] = wall || headOn || first >= 0;
    killer[i] = first >= 0 && !headOn ? first : -1;
  }
  for (const j of killer) if (j >= 0) kills[j] += 1;

  // 4. eat
  const eaten = Array(count).fill(0);
  for (let f = 0; f < food.length; f++) {
    const item = food[f];
    if (item === null) continue;
    let best = -1, bestD2 = Infinity;
    for (let k = 0; k < count; k++) {
      if (!alive[k] || died[k]) continue;
      const [hx, hy] = snakes[k].body[0];
      const dx = item[0] - hx, dy = item[1] - hy;
      const d2 = dx * dx + dy * dy;
      const reach = radius[k] + cfg.eat_margin;
      if (d2 < reach * reach && d2 < bestD2) { best = k; bestD2 = d2; }
    }
    if (best >= 0) {
      eaten[best] += item[2];
      food[f] = null;
    }
  }
  snakes.forEach((s, k) => { s.mass = Math.min(s.mass + eaten[k], cfg.max_mass); });

  // 5. death drops
  const deathDrops = [];
  const most = cfg.death_drop_max_pellets, jitter = 2.0 * cfg.death_drop_jitter;
  snakes.forEach((s, k) => {
    if (!died[k]) return;
    const length = s.body.length, m = s.mass;
    const n = Math.max(Math.min(Math.ceil(m / 2), most), 1);
    const value = cfg.death_drop_fraction * m / n;
    for (let q = 0; q < n; q++) {
      const [px, py] = s.body[Math.floor((q * length) / n)];
      const base = lay.drops + (k * most + q) * 2;
      deathDrops.push([px + (u[base] - 0.5) * jitter, py + (u[base + 1] - 0.5) * jitter, value, false]);
    }
    Object.assign(s, { alive: false, body: [], respawn: cfg.respawn_delay, boosting: false, boost_ticks: 0 });
  });

  // 6. respawn slots that were dead when the tick started
  snakes.forEach((s, k) => {
    if (!s.enabled || alive[k]) return;
    s.respawn -= 1;
    if (s.respawn <= 0) respawnSlot(g, k, u, cfg);
  });

  // 7. new pellets: boost drops, death drops, natural food, into the lowest free slots
  let natural = 0;
  for (const item of food) if (item !== null && item[3]) natural++;
  const need = Math.max(0, Math.min(cfg.natural_food_target - natural, cfg.natural_food_per_tick));
  const fresh = [...boostDrops, ...deathDrops];
  for (let q = 0; q < need; q++) {
    const rr = (cfg.arena_radius - cfg.food_spawn_margin) * Math.sqrt(u[lay.food + 2 * q]);
    const th = 2.0 * Math.PI * u[lay.food + 2 * q + 1];
    fresh.push([rr * Math.cos(th), rr * Math.sin(th), cfg.natural_food_value, true]);
  }
  let next = 0;
  for (let f = 0; f < food.length && next < fresh.length; f++) if (food[f] === null) food[f] = fresh[next++];
  g.tick += 1;
  return { died, killer, kills, eaten };
}

// Place slot k at the first candidate clear of every live head, else the farthest one.
export function respawnSlot(g, k, u, cfg) {
  const snakes = g.snakes, c = cfg.respawn_candidates;
  const base = uLayout(snakes.length, cfg).respawn + k * (2 * c + 1);
  const clearance2 = cfg.respawn_clearance * cfg.respawn_clearance;
  let pick = -1, far = -1, farD2 = -Infinity;
  const candidates = [];
  for (let q = 0; q < c; q++) {
    const rr = (cfg.arena_radius - cfg.respawn_margin) * Math.sqrt(u[base + 2 * q]);
    const th = 2.0 * Math.PI * u[base + 2 * q + 1];
    const x = rr * Math.cos(th), y = rr * Math.sin(th);
    let nearest = Infinity;
    for (const other of snakes) {
      if (!other.alive) continue;
      const dx = x - other.body[0][0], dy = y - other.body[0][1];
      nearest = Math.min(nearest, dx * dx + dy * dy);
    }
    candidates.push([x, y]);
    if (pick < 0 && nearest >= clearance2) pick = q;
    if (nearest > farD2) { far = q; farD2 = nearest; }
  }
  if (pick < 0) pick = far;
  const [x, y] = candidates[pick];
  const h = 2.0 * Math.PI * u[base + 2 * c];
  const cosH = Math.cos(h), sinH = Math.sin(h);
  const body = [];
  for (let i = 0; i < targetLength(cfg.start_mass, cfg); i++) body.push([x - i * cosH, y - i * sinH]);
  Object.assign(snakes[k], { body, heading: h, mass: cfg.start_mass, alive: true, boosting: false, boost_ticks: 0, respawn: 0 });
}

// A fresh arena: the first `enabledCount` slots spawn one by one, natural food is filled.
export function newArena(slots, enabledCount, cfg = CONFIG, random = Math.random) {
  const g = { slots, tick: 0, snakes: [], food: Array(cfg.food_capacity).fill(null) };
  for (let k = 0; k < slots; k++) {
    g.snakes.push({ body: [], heading: 0, mass: cfg.start_mass, alive: false, enabled: k < enabledCount,
                    boosting: false, boost_ticks: 0, respawn: 0 });
  }
  const u = randomU(slots, cfg, random);
  for (let k = 0; k < enabledCount; k++) respawnSlot(g, k, u, cfg);
  for (let f = 0; f < cfg.natural_food_target; f++) {
    const rr = (cfg.arena_radius - cfg.food_spawn_margin) * Math.sqrt(random());
    const th = 2.0 * Math.PI * random();
    g.food[f] = [rr * Math.cos(th), rr * Math.sin(th), cfg.natural_food_value, true];
  }
  return g;
}
```

- [ ] **Step 4: Write `web/slither/observe.js`**

```js
// The network's observation (spec §3), a line-by-line port of tests/slither_reference.observe.
// 32 head-centred sectors x 8 channels, then 7 scalars; checked by tests/test_slither_js_parity.py.
import { bodyRadius, NUM_ACTIONS } from "./rules.js";

export const OBS_CHANNELS = 8;
export const NUM_SCALARS = 7;
export const obsSize = (cfg) => cfg.sectors * OBS_CHANNELS + NUM_SCALARS;

export function observe(g, k, cfg) {
  const K = cfg.sectors, V = cfg.vision, width = 2.0 * Math.PI / K;
  const out = new Float64Array(obsSize(cfg));
  const snakes = g.snakes, me = snakes[k];
  if (!me.alive) return out;
  const [hx, hy] = me.body[0];
  const h = me.heading, c = Math.cos(h), sn = Math.sin(h);
  const radius = snakes.map((s) => bodyRadius(s.mass, cfg));
  const close = (d) => Math.min(Math.max(1.0 - d / V, 0.0), 1.0);
  const sector = (rx, ry) => {
    const x = rx * c + ry * sn, y = ry * c - rx * sn;
    return ((Math.floor((Math.atan2(y, x) + Math.PI) / width) % K) + K) % K;
  };

  for (let q = 0; q < K; q++) {
    const th = h + (-Math.PI + (q + 0.5) * width);
    const dx = Math.cos(th), dy = Math.sin(th);
    const b = hx * dx + hy * dy;
    const cc = hx * hx + hy * hy - cfg.arena_radius * cfg.arena_radius;
    out[q * 8] = close(-b + Math.sqrt(Math.max(b * b - cc, 0.0)));
  }

  snakes.forEach((other, j) => {
    if (!other.alive) return;
    other.body.forEach(([px, py], idx) => {
      if (j === k && idx < cfg.neck) return;
      const rx = px - hx, ry = py - hy;
      const v = close(Math.sqrt(rx * rx + ry * ry) - radius[j]);
      if (v <= 0) return;
      const q = sector(rx, ry);
      if (j === k) {
        out[q * 8 + 3] = Math.max(out[q * 8 + 3], v);
      } else {
        out[q * 8 + 1] = Math.max(out[q * 8 + 1], v);
        out[q * 8 + 2] = Math.max(out[q * 8 + 2], radius[j] / 15.0);
      }
    });
  });

  const food = new Float64Array(K);
  for (const item of g.food) {
    if (item === null) continue;
    const rx = item[0] - hx, ry = item[1] - hy;
    if (Math.sqrt(rx * rx + ry * ry) < V) food[sector(rx, ry)] += item[2];
  }
  for (let q = 0; q < K; q++) out[q * 8 + 4] = Math.min(food[q] / 10.0, 1.0);

  snakes.forEach((other, j) => {
    if (j === k || !other.alive) return;
    const [ox, oy] = other.body[0];
    const rx = ox - hx, ry = oy - hy;
    const v = close(Math.sqrt(rx * rx + ry * ry) - radius[j]);
    if (v <= 0) return;
    const q = sector(rx, ry);
    if (v > out[q * 8 + 5]) {
      out[q * 8 + 5] = v;
      out[q * 8 + 6] = Math.cos(other.heading - h);
      out[q * 8 + 7] = Math.sin(other.heading - h);
    }
  });

  const base = K * 8;
  out[base] = Math.log(me.mass) / Math.log(cfg.max_mass);
  out[base + 1] = radius[k] / 15.0;
  out[base + 2] = me.boosting ? 1.0 : 0.0;
  out[base + 3] = me.mass >= cfg.boost_min_mass ? 1.0 : 0.0;
  const dist = Math.sqrt(hx * hx + hy * hy);
  out[base + 4] = dist / cfg.arena_radius;
  if (dist > 1e-9) {
    const bx = -hx / dist, by = -hy / dist;
    out[base + 5] = bx * c + by * sn;
    out[base + 6] = by * c - bx * sn;
  } else {
    out[base + 5] = 1.0;
  }
  return out;
}

// Boost actions (odd indices) need boost_min_mass; steering is always allowed.
export function actionMask(g, k, cfg) {
  const canBoost = g.snakes[k].mass >= cfg.boost_min_mass;
  return Array.from({ length: NUM_ACTIONS }, (_, a) => a % 2 === 0 || canBoost);
}
```

- [ ] **Step 5: Run the parity test to check it passes**

Run: `uv run pytest tests/test_slither_js_parity.py -q`
Expected: 1 passed. Node prints `290 cases, 0 failures`.

If a case fails, the first 10 failures name the group, case and field. Compare the JS
expression with the matching line in `tests/slither_reference.py`. Fix the JS; never
fix the tolerance.

- [ ] **Step 6: Commit**

```bash
git add web/slither/rules.js web/slither/observe.js web/slither/test/parity.mjs tests/test_slither_js_parity.py
git commit -m "Slither: JavaScript rules and observation, checked against Python

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Browser component, standalone page and link

**Files:**
- Create: `web/slither/render.js`, `web/slither/slither-demo.js`, `web/slither/slither-demo.css`, `web/slither/index.html`, `web/slither/page.js`
- Modify: `web/index.html`, adding one header link and nothing else

**Interfaces:**
- Consumes:
  - from `rules.js`: `CONFIG`, `decodeAction`, `newArena`, `randomU`, `respawnSlot`,
    `stepTick`, `bodyRadius`;
  - from `observe.js`: `observe`, `obsSize`, `actionMask`;
  - `web/slither/model.onnx` (from Task 11's export);
  - `web/slither/stats.json`.
- Produces:
  - `mountSlitherDemo(root, {base = "", autoLoad = true, ortCdn, mode = "watch"}) -> {selectMode}`;
  - `drawArena(ctx, game, cfg, view, camera) -> camera`, `headAt`, `COLORS` and `NAMES`
    (from `render.js`).

- [ ] **Step 1: Make a temporary model for development**

Run: `uv run --group train python scripts/build_slither_assets.py --run-dir /tmp/slither-smoke --checkpoint latest.pt`

This uses the smoke run from Task 10 Step 5. Expected:
`wrote …/web/slither/model.onnx and …/web/slither/stats.json`.

These two files are placeholders. **Do not commit them in this task.** Task 14
overwrites them with the trained network.

- [ ] **Step 2: Write `web/slither/render.js`**

```js
// Canvas rendering of a slither arena. A camera follows one snake; snakes are round-capped
// strokes through their trail points (tapering at the tail), with the head interpolated
// between physics ticks so motion stays smooth at 60 fps on a 30 Hz simulation.
import { bodyRadius } from "./rules.js";

export const COLORS = ["#f59e0b", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#facc15",
  "#fb7185", "#22d3ee", "#fb923c", "#4ade80", "#c084fc", "#60a5fa"];
export const NAMES = ["Orange", "Blue", "Violet", "Green", "Pink", "Yellow", "Coral", "Cyan",
  "Amber", "Lime", "Purple", "Sky"];

/** Head of snake k at fraction `alpha` of the way from the previous tick to the current one. */
export function headAt(game, prevHeads, k, alpha) {
  const h = game.snakes[k].body[0], p = prevHeads[k];
  if (!p) return h;
  return [p[0] + (h[0] - p[0]) * alpha, p[1] + (h[1] - p[1]) * alpha];
}

/**
 * Draw the arena and return the camera ({x, y, scale}), eased toward the followed snake.
 * view: { alpha, prevHeads, follow, human, theme, dpr }
 * theme: { board, outside, grid, wall, fg }
 */
export function drawArena(ctx, game, cfg, view, camera) {
  const { width: W, height: H } = ctx.canvas;
  const { theme, dpr } = view;
  const target = game.snakes[view.follow];
  let cam = camera;
  if (target && target.alive) {
    const [fx, fy] = headAt(game, view.prevHeads, view.follow, view.alpha);
    const want = Math.min(W, H) / (2 * (70 + 6 * bodyRadius(target.mass, cfg)));
    cam = camera
      ? { x: camera.x + (fx - camera.x) * 0.2, y: camera.y + (fy - camera.y) * 0.2, scale: camera.scale + (want - camera.scale) * 0.05 }
      : { x: fx, y: fy, scale: want };
  } else if (!cam) {
    cam = { x: 0, y: 0, scale: Math.min(W, H) / (2.1 * cfg.arena_radius) };
  }
  const sx = (x) => W / 2 + (x - cam.x) * cam.scale;
  const sy = (y) => H / 2 + (y - cam.y) * cam.scale;
  const R = cfg.arena_radius;

  ctx.fillStyle = theme.outside;
  ctx.fillRect(0, 0, W, H);
  ctx.beginPath();
  ctx.arc(sx(0), sy(0), R * cam.scale, 0, 2 * Math.PI);
  ctx.fillStyle = theme.board;
  ctx.fill();

  // background grid, clipped to the arena
  ctx.save();
  ctx.clip();
  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = dpr;
  ctx.beginPath();
  const step = 30, halfW = W / 2 / cam.scale, halfH = H / 2 / cam.scale;
  for (let x = Math.floor((cam.x - halfW) / step) * step; x <= cam.x + halfW; x += step) { ctx.moveTo(sx(x), 0); ctx.lineTo(sx(x), H); }
  for (let y = Math.floor((cam.y - halfH) / step) * step; y <= cam.y + halfH; y += step) { ctx.moveTo(0, sy(y)); ctx.lineTo(W, sy(y)); }
  ctx.stroke();
  ctx.restore();

  ctx.beginPath();
  ctx.arc(sx(0), sy(0), R * cam.scale, 0, 2 * Math.PI);
  ctx.strokeStyle = theme.wall;
  ctx.lineWidth = 3 * dpr;
  ctx.stroke();

  // food: a soft halo and a bright core, coloured by slot so the field looks lively
  game.food.forEach((item, f) => {
    if (!item) return;
    const x = sx(item[0]), y = sy(item[1]);
    if (x < -20 || y < -20 || x > W + 20 || y > H + 20) return;
    const r = (0.9 + Math.sqrt(item[2])) * cam.scale;
    const color = COLORS[f % COLORS.length];
    ctx.globalAlpha = 0.22;
    dot(ctx, x, y, r * 2.2, color);
    ctx.globalAlpha = 1;
    dot(ctx, x, y, r, color);
  });

  // snakes, smallest first so big snakes are drawn on top
  game.snakes.map((s, k) => k).filter((k) => game.snakes[k].alive)
    .sort((a, b) => game.snakes[a].mass - game.snakes[b].mass)
    .forEach((k) => drawSnake(ctx, game, cfg, k, view, sx, sy, cam.scale));

  drawMinimap(ctx, game, cfg, view, W, H);
  return cam;
}

function dot(ctx, x, y, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawSnake(ctx, game, cfg, k, view, sx, sy, scale) {
  const s = game.snakes[k];
  const color = COLORS[k % COLORS.length];
  const r = bodyRadius(s.mass, cfg) * scale;
  const head = headAt(game, view.prevHeads, k, view.alpha);
  const pts = [head, ...s.body.slice(1)];
  const tail = Math.min(12, Math.floor(pts.length / 3));
  const main = pts.length - tail;
  const path = (from, to) => {
    ctx.beginPath();
    ctx.moveTo(sx(pts[from][0]), sy(pts[from][1]));
    for (let i = from + 1; i < to; i++) ctx.lineTo(sx(pts[i][0]), sy(pts[i][1]));
  };
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (s.boosting) {
    ctx.globalAlpha = 0.35;
    path(0, pts.length);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * r + 8 * view.dpr;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  // an outline pass, then the body colour; the last `tail` segments taper
  for (const [width, stroke] of [[2 * r + 2 * view.dpr, darker(color)], [2 * r, color]]) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    path(0, main + 1);
    ctx.stroke();
    for (let i = main; i < pts.length - 1; i++) {
      const t = 1 - (i - main + 1) / (tail + 1);
      ctx.lineWidth = Math.max(view.dpr, width * (0.35 + 0.65 * t));
      ctx.beginPath();
      ctx.moveTo(sx(pts[i][0]), sy(pts[i][1]));
      ctx.lineTo(sx(pts[i + 1][0]), sy(pts[i + 1][1]));
      ctx.stroke();
    }
  }
  // eyes, looking along the heading
  const hx = sx(head[0]), hy = sy(head[1]);
  const c = Math.cos(s.heading), sn = Math.sin(s.heading);
  for (const side of [-1, 1]) {
    const ex = hx + (c * 0.35 - sn * side * 0.5) * r, ey = hy + (sn * 0.35 + c * side * 0.5) * r;
    dot(ctx, ex, ey, 0.32 * r, "#ffffff");
    dot(ctx, ex + c * 0.12 * r, ey + sn * 0.12 * r, 0.16 * r, "#1f1f1d");
  }
  if (k === view.human) {
    ctx.fillStyle = view.theme.fg;
    ctx.font = `600 ${12 * view.dpr}px -apple-system, BlinkMacSystemFont, Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("You", hx, hy - r - 8 * view.dpr);
  }
}

function drawMinimap(ctx, game, cfg, view, W, H) {
  const { dpr, theme } = view;
  const m = 34 * dpr, cx = W - m - 10 * dpr, cy = H - m - 10 * dpr, k = m / cfg.arena_radius;
  ctx.globalAlpha = 0.85;
  dot(ctx, cx, cy, m, theme.board);
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, m, 0, 2 * Math.PI);
  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = dpr;
  ctx.stroke();
  game.snakes.forEach((s, i) => {
    if (!s.alive) return;
    const [x, y] = s.body[0];
    dot(ctx, cx + x * k, cy + y * k, (i === view.follow ? 2.6 : 1.6) * dpr, COLORS[i % COLORS.length]);
  });
}

// #rrggbb -> a darker rgb() for outlines
function darker(hex) {
  const v = [1, 3, 5].map((i) => Math.round(parseInt(hex.slice(i, i + 2), 16) * 0.7));
  return `rgb(${v.join(",")})`;
}
```

Screen y equals world y; there is no flip. A positive turn therefore looks clockwise
(to the right) on screen. The steering fan in Step 3 follows the same convention.

- [ ] **Step 3: Write `web/slither/slither-demo.js`**

```js
// Slither self-play demo as a mountable component.
//
//   import { mountSlitherDemo } from "./slither-demo.js";
//   mountSlitherDemo(document.querySelector("#slither"), { base: "/snake/slither/", autoLoad: false });
//
// Everything runs in the browser: the rules (rules.js, a checked port of the Python
// engine), the observation (observe.js), the trained network (ONNX Runtime Web) and the
// rendering (render.js). Styles live in slither-demo.css, scoped to .slither-demo; colours
// come from the host page's --color-* tokens when it has them.

import { CONFIG, decodeAction, newArena, randomU, respawnSlot, stepTick } from "./rules.js";
import { actionMask, observe, obsSize } from "./observe.js";
import { COLORS, NAMES, drawArena } from "./render.js";

const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL = { file: "model.onnx", mb: 0.6 };
const SLOTS = 12;
const SNAKES = 10;        // enabled slots; in Play mode slot 0 is you
const HUMAN = 0;
const MAX_CATCH_UP = 8;   // physics ticks per frame before a backlog is dropped

const TEMPLATE = `
  <div class="sl-tabs" role="tablist">
    <button type="button" role="tab" data-mode="watch">Watch</button>
    <button type="button" role="tab" data-mode="play">Play</button>
  </div>
  <div class="sl-game">
    <div class="sl-board">
      <canvas class="sl-arena" data-sl="arena" aria-label="Slither arena"></canvas>
      <div class="sl-overlay" data-sl="overlay" hidden></div>
      <button type="button" class="sl-boost" data-sl="boost" hidden>Boost</button>
    </div>
    <div class="sl-side">
      <div><p class="sl-lbl">Leaderboard</p><ol class="sl-leaders" data-sl="leaders"></ol></div>
      <div><p class="sl-lbl"><span data-sl="ai-name">AI</span> · steering</p>
        <canvas class="sl-fan" data-sl="fan" aria-label="Steering probabilities"></canvas>
        <p class="sl-meta">Boost <span data-sl="boost-p">–</span> · Value <span data-sl="value">–</span></p></div>
      <div class="sl-pills">
        <button type="button" data-sl="restart">Restart</button>
        <button type="button" data-sl="pause">Pause</button>
        <button type="button" data-sl="sample">Sampling</button>
      </div>
      <p class="sl-meta" data-sl="hint"></p>
    </div>
  </div>`;

let ortPromise = null;
function loadOrt(src) {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  ortPromise ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = () => resolve(globalThis.ort);
    script.onerror = () => reject(new Error("couldn't load ONNX Runtime"));
    document.head.appendChild(script);
  });
  return ortPromise;
}

/**
 * Build the demo inside `root`.
 * @param {HTMLElement} root
 * @param {{ base?: string, autoLoad?: boolean, ortCdn?: string, mode?: "watch" | "play" }} [options]
 *   base: URL prefix of model.onnx (default: same folder as the page).
 *   autoLoad: download the network immediately (default) or wait for a click.
 */
export function mountSlitherDemo(root, options = {}) {
  const cfg = CONFIG;
  const base = options.base ?? "";
  const ortCdn = options.ortCdn ?? ORT_CDN;
  const tickMs = 1000 / cfg.tick_hz;
  const coarse = matchMedia("(pointer: coarse)").matches;
  root.classList.add("slither-demo");
  root.innerHTML = TEMPLATE;

  const $ = (key) => root.querySelector(`[data-sl="${key}"]`);
  const canvas = $("arena");
  const ctx = canvas.getContext("2d");
  const fan = $("fan");
  const fanCtx = fan.getContext("2d");
  const overlay = $("overlay");
  const tabs = [...root.querySelectorAll(".sl-tabs button")];

  let mode = options.mode ?? "watch";
  let session = null;
  let started = false, paused = false, visible = true, busy = false, deterministic = false;
  let theme = {}, dpr = 1, camera = null;
  let game, prevHeads, plan, brain, follow;
  let acc = 0, last = performance.now(), frameNo = 0;
  let pointer = null, boostHeld = false;
  let you = null; // { kills, ticks } for your current life in Play mode

  const heads = () => game.snakes.map((s) => (s.alive ? s.body[0].slice() : null));
  const followed = () => (mode === "play" && game.snakes[HUMAN].alive ? HUMAN : follow);
  const name = (k) => (mode === "play" && k === HUMAN ? "You" : NAMES[k % NAMES.length]);

  function biggest(skipHuman = false) {
    let best = -1;
    game.snakes.forEach((s, k) => {
      if (!s.alive || (skipHuman && mode === "play" && k === HUMAN)) return;
      if (best < 0 || s.mass > game.snakes[best].mass) best = k;
    });
    return Math.max(best, 0);
  }

  function newGame() {
    game = newArena(SLOTS, SNAKES, cfg);
    prevHeads = heads();
    plan = Array.from({ length: SLOTS }, () => ({ turn: 0, boost: false }));
    brain = Array(SLOTS).fill(null);
    follow = mode === "play" ? HUMAN : biggest();
    you = mode === "play" ? { kills: 0, ticks: 0 } : null;
    camera = null;
    acc = 0;
    if (session) hideOverlay();
  }

  // ---------- simulation ----------

  async function decide() {
    const ai = [];
    game.snakes.forEach((s, k) => { if (s.alive && !(mode === "play" && k === HUMAN)) ai.push(k); });
    if (!ai.length || !session) return; // nobody to steer: no inference call
    const D = obsSize(cfg);
    const data = new Float32Array(ai.length * D);
    ai.forEach((k, r) => data.set(observe(game, k, cfg), r * D));
    const out = await session.run({ obs: new globalThis.ort.Tensor("float32", data, [ai.length, D]) });
    const logits = out.logits.data, value = out.value.data;
    ai.forEach((k, r) => {
      const probs = maskedSoftmax(Array.from(logits.subarray(r * 18, r * 18 + 18)), actionMask(game, k, cfg));
      const action = deterministic ? argmax(probs) : sample(probs);
      plan[k] = decodeAction(action, cfg);
      brain[k] = { probs, value: value[r], action };
    });
  }

  function humanTurn() {
    const s = game.snakes[HUMAN];
    if (!pointer || !camera) return 0;
    const [hx, hy] = s.body[0];
    const wx = camera.x + (pointer.x - canvas.width / 2) / camera.scale;
    const wy = camera.y + (pointer.y - canvas.height / 2) / camera.scale;
    if (Math.hypot(wx - hx, wy - hy) < 2) return 0;
    let d = Math.atan2(wy - hy, wx - hx) - s.heading;
    d = Math.atan2(Math.sin(d), Math.cos(d)); // wrap to (-pi, pi]
    return Math.max(-cfg.max_turn, Math.min(cfg.max_turn, d));
  }

  async function tick() {
    if (game.tick % cfg.ticks_per_decision === 0) await decide();
    const turns = plan.map((p) => p.turn), boosts = plan.map((p) => p.boost);
    if (mode === "play" && game.snakes[HUMAN].alive) {
      turns[HUMAN] = humanTurn();
      boosts[HUMAN] = boostHeld;
    }
    prevHeads = heads();
    const ev = stepTick(game, turns, boosts, randomU(SLOTS, cfg), cfg);
    if (mode === "play" && you) {
      if (game.snakes[HUMAN].alive || ev.died[HUMAN]) {
        you.ticks += 1;
        you.kills += ev.kills[HUMAN];
      }
      if (ev.died[HUMAN]) onHumanDeath();
    }
    if (!game.snakes[follow]?.alive) follow = biggest();
  }

  function onHumanDeath() {
    const s = game.snakes[HUMAN];
    s.enabled = false; // you respawn only when you ask to
    const secs = Math.round(you.ticks / cfg.tick_hz);
    showOverlay(`You reached mass ${Math.round(s.mass)}`, `${you.kills} kill${you.kills === 1 ? "" : "s"} · ${secs}s alive`,
      { label: "Play again", run: respawnYou });
  }

  function respawnYou() {
    const s = game.snakes[HUMAN];
    s.enabled = true;
    respawnSlot(game, HUMAN, randomU(SLOTS, cfg), cfg);
    prevHeads[HUMAN] = null;
    you = { kills: 0, ticks: 0 };
    hideOverlay();
  }

  async function advance() {
    let n = 0;
    while (acc >= tickMs && n < MAX_CATCH_UP) {
      await tick();
      acc -= tickMs;
      n++;
    }
    if (acc >= tickMs) acc = 0; // fell behind (slow device or busy tab): drop the backlog
  }

  function frame(now) {
    const dt = Math.min(now - last, 250);
    last = now;
    const running = started && session && !paused && visible;
    if (running) {
      acc += dt;
      if (!busy) {
        busy = true;
        advance().catch((err) => showOverlay("Something went wrong", String(err.message ?? err)))
          .finally(() => { busy = false; });
      }
    }
    camera = drawArena(ctx, game, cfg, {
      alpha: running ? Math.min(acc / tickMs, 1) : 1, prevHeads, follow: followed(),
      human: mode === "play" ? HUMAN : -1, theme, dpr,
    }, camera);
    if (++frameNo % 6 === 0) updatePanel();
    requestAnimationFrame(frame);
  }

  // ---------- panel ----------

  function updatePanel() {
    const top = game.snakes.map((s, k) => [k, s]).filter(([, s]) => s.alive)
      .sort((a, b) => b[1].mass - a[1].mass).slice(0, 5);
    $("leaders").innerHTML = top.map(([k, s]) => `<li data-k="${k}" class="${k === followed() ? "sl-on" : ""}">`
      + `<span class="sl-dot" style="background:${COLORS[k % COLORS.length]}"></span><span>${name(k)}</span>`
      + `<span class="sl-mass">${Math.round(s.mass)}</span></li>`).join("");
    const k = mode === "watch" ? follow : biggest(true);
    const b = brain[k];
    $("ai-name").textContent = NAMES[k % NAMES.length];
    $("boost-p").textContent = b ? `${Math.round(100 * b.probs.filter((_, a) => a % 2).reduce((x, y) => x + y, 0))}%` : "–";
    $("value").textContent = b ? b.value.toFixed(2) : "–";
    drawFan(b);
  }

  // The 9 steering bins as a fan pointing up (forward); wedge length is the bin's probability.
  function drawFan(b) {
    const W = fan.width, H = fan.height;
    fanCtx.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H - 6 * dpr, rMin = 10 * dpr, rMax = Math.min(H - 12 * dpr, W / 2 - 4 * dpr);
    const steer = Array.from({ length: 9 }, (_, i) => (b ? b.probs[2 * i] + b.probs[2 * i + 1] : 0));
    const top = Math.max(...steer, 1e-9);
    for (let i = 0; i < 9; i++) {
      const a = -Math.PI / 2 + (i - 4) * (Math.PI / 9);
      const r = rMin + (rMax - rMin) * (steer[i] / top);
      fanCtx.beginPath();
      fanCtx.moveTo(cx, cy);
      fanCtx.arc(cx, cy, r, a - Math.PI / 20, a + Math.PI / 20);
      fanCtx.closePath();
      fanCtx.fillStyle = b && Math.floor(b.action / 2) === i ? theme.accent : theme.faint;
      fanCtx.fill();
    }
  }

  // ---------- overlay ----------

  function showOverlay(title, sub = "", action = null) {
    overlay.hidden = false;
    overlay.innerHTML = `<div>${title}${sub ? `<small>${sub}</small>` : ""}</div>`;
    if (action) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sl-start";
      btn.textContent = action.label;
      btn.addEventListener("click", action.run);
      overlay.firstElementChild.appendChild(btn);
    }
  }
  function hideOverlay() { overlay.hidden = true; overlay.innerHTML = ""; }

  // ---------- network and modes ----------

  async function loadModel() {
    if (session) return true;
    showOverlay("Loading…");
    try {
      const ort = await loadOrt(`${ortCdn}ort.min.js`);
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = ortCdn;
      session = await ort.InferenceSession.create(base + MODEL.file);
      hideOverlay();
      return true;
    } catch (err) {
      showOverlay("Couldn't load the network", String(err.message ?? err));
      return false;
    }
  }

  function selectMode(next) {
    mode = next;
    tabs.forEach((b) => b.classList.toggle("sl-active", b.dataset.mode === mode));
    root.classList.toggle("sl-playing", mode === "play");
    $("boost").hidden = !(mode === "play" && coarse);
    $("hint").textContent = mode === "play"
      ? (coarse ? "Touch to steer, hold Boost to speed up." : "Steer with the mouse. Hold the button or Space to boost.")
      : "Click a snake to follow it.";
    newGame();
  }

  async function start() {
    started = true;
    if (await loadModel()) selectMode(mode);
  }

  // ---------- input and wiring ----------

  const toCanvas = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
  };
  function pickSnake(p) {
    let best = -1, bestD = 40 * dpr;
    game.snakes.forEach((s, k) => {
      if (!s.alive || !camera) return;
      const x = canvas.width / 2 + (s.body[0][0] - camera.x) * camera.scale;
      const y = canvas.height / 2 + (s.body[0][1] - camera.y) * camera.scale;
      const d = Math.hypot(x - p.x, y - p.y);
      if (d < bestD) { best = k; bestD = d; }
    });
    if (best >= 0) follow = best;
  }

  tabs.forEach((btn) => btn.addEventListener("click", () => { btn.blur(); if (started) selectMode(btn.dataset.mode); }));
  canvas.addEventListener("pointermove", (e) => { pointer = toCanvas(e); });
  canvas.addEventListener("pointerdown", (e) => {
    pointer = toCanvas(e);
    if (mode === "play" && e.pointerType === "mouse") boostHeld = true;
    if (mode === "watch") pickSnake(pointer);
  });
  window.addEventListener("pointerup", () => { boostHeld = false; });
  $("boost").addEventListener("pointerdown", (e) => { e.preventDefault(); boostHeld = true; });
  $("boost").addEventListener("pointerleave", () => { boostHeld = false; });
  $("leaders").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-k]");
    if (li && mode === "watch") follow = Number(li.dataset.k);
  });
  // Space boosts only while you're playing and the arena is on screen.
  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || mode !== "play" || !visible || !root.isConnected) return;
    if (e.target instanceof HTMLElement && e.target.closest("input, textarea, select, button, [contenteditable]")) return;
    e.preventDefault();
    boostHeld = true;
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") boostHeld = false; });
  $("restart").addEventListener("click", () => { if (started) newGame(); });
  $("pause").addEventListener("click", () => {
    paused = !paused;
    $("pause").textContent = paused ? "Resume" : "Pause";
  });
  $("sample").addEventListener("click", () => {
    deterministic = !deterministic;
    $("sample").textContent = deterministic ? "Greedy" : "Sampling";
  });

  function readTheme() {
    const css = getComputedStyle(root);
    const v = (n) => css.getPropertyValue(n).trim();
    const board = v("--sl-board"), fg = v("--sl-fg");
    theme = { board, fg, outside: mix(board, fg, 0.06), grid: mix(board, v("--sl-line"), 0.7),
              wall: v("--sl-wall"), accent: v("--sl-accent"), faint: mix(board, fg, 0.25) };
  }
  function resizeCanvas() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    fan.width = Math.max(1, Math.round(fan.clientWidth * dpr));
    fan.height = Math.max(1, Math.round(fan.clientHeight * dpr));
  }

  new ResizeObserver(resizeCanvas).observe(canvas);
  // Follow the host page's light/dark switch (a class or data attribute on <html>).
  new MutationObserver(readTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", readTheme);
  // Pause while scrolled out of view.
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(root);

  readTheme();
  resizeCanvas();
  tabs.forEach((b) => b.classList.toggle("sl-active", b.dataset.mode === mode));
  newGame();
  requestAnimationFrame(frame);
  if (options.autoLoad === false) {
    showOverlay("Ten snakes, all trained by self-play", "", { label: `▶ Load the AI (${MODEL.mb} MB)`, run: () => start() });
  } else {
    start();
  }
  return { selectMode };
}

// ---------- helpers ----------

function sample(probs) {
  let u = Math.random();
  for (let a = 0; a < probs.length; a++) {
    u -= probs[a];
    if (u <= 0) return a;
  }
  return probs.length - 1;
}

const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);

function maskedSoftmax(logits, mask) {
  const allowed = mask.some(Boolean) ? mask : mask.map(() => true);
  const max = Math.max(...logits.filter((_, a) => allowed[a]));
  const exp = logits.map((l, a) => (allowed[a] ? Math.exp(l - max) : 0));
  const sum = exp.reduce((x, y) => x + y, 0);
  return exp.map((e) => e / sum);
}

// Blend colour a toward b by t (0..1). Accepts #rrggbb or rgb(...) strings.
function mix(a, b, t) {
  const parse = (c) => (c.startsWith("#")
    ? [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16))
    : (c.match(/\d+(\.\d+)?/g) || [0, 0, 0]).slice(0, 3).map(Number));
  const [x, y] = [parse(a), parse(b)];
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
}
```

- [ ] **Step 4: Write `web/slither/slither-demo.css`**

```css
/* Slither demo component. Everything is scoped to .slither-demo.
   Colours use the host page's tokens when present (alexkozik.com names them --color-*),
   with standalone fallbacks; dark mode follows a .dark class on <html>. */
.slither-demo {
  --sl-bg: var(--color-bg-elev, #ffffff);
  --sl-board: var(--color-bg, #faf9f5);
  --sl-fg: var(--color-fg, #2b2a26);
  --sl-muted: var(--color-fg-muted, #5e5d59);
  --sl-faint: var(--color-fg-faint, #8f8d86);
  --sl-line: var(--color-line, #e3e0d6);
  --sl-accent: var(--color-accent, #2f7a6b);
  --sl-wall: #e11d48;
  --sl-mono: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
  --sl-display: var(--font-display, Newsreader, Georgia, serif);

  display: block;
  background: var(--sl-bg);
  border: 1px solid var(--sl-line);
  border-radius: 12px;
  padding: 16px;
  color: var(--sl-fg);
  font-size: 13px;
  line-height: 1.45;
}
:root.dark .slither-demo { --sl-wall: #fb7185; }
@media (min-width: 640px) { .slither-demo { padding: 20px; } }

.slither-demo button { font: inherit; color: inherit; margin: 0; }
.slither-demo button:focus-visible { outline: 2px solid var(--sl-accent); outline-offset: 2px; }
.slither-demo ol { list-style: none; margin: 0; padding: 0; }
.slither-demo p { margin: 0; }
.slither-demo [hidden] { display: none !important; }

.slither-demo .sl-tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
.slither-demo .sl-tabs button,
.slither-demo .sl-pills button {
  border: 1px solid var(--sl-line); background: none; border-radius: 999px; padding: 4px 12px;
  font-size: 12px; font-weight: 500; line-height: 1.5; color: var(--sl-muted); cursor: pointer;
  transition: border-color .15s, color .15s;
}
.slither-demo .sl-tabs button:hover,
.slither-demo .sl-pills button:hover { border-color: var(--sl-accent); color: var(--sl-fg); }
.slither-demo .sl-tabs button.sl-active { background: var(--sl-fg); border-color: var(--sl-fg); color: var(--sl-bg); }

.slither-demo .sl-game { display: grid; gap: 20px; }
@media (min-width: 640px) { .slither-demo .sl-game { grid-template-columns: minmax(0, 1fr) 11rem; } }

.slither-demo .sl-board { position: relative; aspect-ratio: 4 / 3; }
.slither-demo .sl-arena { width: 100%; height: 100%; display: block; border-radius: 8px;
  box-shadow: 0 0 0 1px var(--sl-line); cursor: crosshair; }
.slither-demo.sl-playing .sl-arena { touch-action: none; }
.slither-demo .sl-overlay {
  position: absolute; inset: 0; display: grid; place-items: center; text-align: center; padding: 20px;
  border-radius: 8px; background: color-mix(in srgb, var(--sl-board) 72%, transparent);
  font-family: var(--sl-display); font-size: 20px; font-weight: 500; pointer-events: none;
}
.slither-demo .sl-overlay small { display: block; margin-top: 4px; font-family: inherit; font-size: 13px; color: var(--sl-faint); font-weight: 400; }
.slither-demo .sl-start {
  pointer-events: auto; display: block; margin: 14px auto 0; border: 0; border-radius: 999px;
  background: var(--sl-fg); color: var(--sl-bg); padding: 9px 18px; font-size: 13px; font-weight: 600;
  cursor: pointer; transition: opacity .15s;
}
.slither-demo .sl-start:hover { opacity: .85; }
.slither-demo .sl-boost {
  position: absolute; left: 12px; bottom: 12px; border: 0; border-radius: 999px; padding: 12px 20px;
  background: var(--sl-fg); color: var(--sl-bg); font-weight: 600; touch-action: none; user-select: none;
}

.slither-demo .sl-side { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.slither-demo .sl-lbl { margin-bottom: 6px; font-size: 11px; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; color: var(--sl-faint); }
.slither-demo .sl-leaders { display: grid; gap: 4px; }
.slither-demo .sl-leaders li { display: grid; grid-template-columns: 8px 1fr auto; gap: 8px; align-items: center; cursor: pointer; }
.slither-demo .sl-leaders li.sl-on { font-weight: 600; }
.slither-demo .sl-dot { width: 8px; height: 8px; border-radius: 999px; }
.slither-demo .sl-mass { font-family: var(--sl-mono); font-size: 12px; color: var(--sl-muted); }
.slither-demo .sl-fan { width: 100%; height: 96px; display: block; }
.slither-demo .sl-pills { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.slither-demo .sl-meta { font-size: 13px; color: var(--sl-faint); }
```

- [ ] **Step 5: Write the standalone page**

`web/slither/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Slither Self-Play</title>
  <meta name="description" content="Neural networks that learned a slither.io-style game by playing against themselves, running live in your browser.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../style.css?v=13">
  <link rel="stylesheet" href="slither-demo.css?v=1">
  <script>
    // Light by default; dark only if the visitor chose it with the toggle.
    try {
      if (localStorage.getItem("theme") === "dark") document.documentElement.classList.add("dark");
    } catch {}
  </script>
</head>
<body>
  <header class="top">
    <div class="wrap bar">
      <span class="brand"><span class="spark" aria-hidden="true">✻</span>snake self-play</span>
      <a class="quiet" href="../">Grid</a>
      <a class="quiet" href="https://github.com/LambdaAK/SnakeDQN">GitHub</a>
      <button id="theme" class="icon" type="button" aria-label="Toggle dark mode">
        <svg class="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
        <svg class="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
      </button>
    </div>
  </header>

  <main class="wrap">
    <h1 class="t-page">Slither, learned by playing itself<span class="accent">.</span></h1>
    <p class="t-sub lede">Continuous steering, boosting, cutting each other off. Every snake but yours is a neural network trained only against copies of itself.</p>

    <section id="demo" aria-label="Live arena"></section>

    <section class="block">
      <h2 class="t-group">Results</h2>
      <div id="results" class="stats"></div>
    </section>

    <section class="block">
      <h2 class="t-group">How it works</h2>
      <ul class="how t-desc">
        <li><span class="t-title">Simulator.</span> Snakes are chains of points; 30 Hz physics in vectorised NumPy, checked against a reference and this page's JavaScript.</li>
        <li><span class="t-title">Senses.</span> 32 rays' worth of sectors around the head: walls, bodies, heads, food.</li>
        <li><span class="t-title">Training.</span> PPO self-play, rewarded for mass gained and penalised for dying.</li>
      </ul>
    </section>
  </main>

  <footer class="wrap t-meta foot">
    <a class="ink" href="https://github.com/LambdaAK/SnakeDQN">Source</a> · Alex Kozik
  </footer>

  <script type="module" src="page.js?v=1"></script>
</body>
</html>
```

`web/slither/page.js`:

```js
// The standalone Slither page: theme toggle, the arena component, and training results.
import { mountSlitherDemo } from "./slither-demo.js?v=1";

const $ = (id) => document.getElementById(id);

if (new URLSearchParams(location.search).has("embed")) document.body.classList.add("embed");

$("theme").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch {}
});

mountSlitherDemo($("demo"));

(async () => {
  let stats = null;
  try { stats = await (await fetch("stats.json?v=1")).json(); } catch {}
  const evals = stats?.final_eval;
  const tiles = [];
  if (evals?.hunter) {
    tiles.push([`${Math.round(100 * evals.hunter.mass_share)}%`, "of all mass vs 7 hunter bots (fair: 12.5%)"]);
    tiles.push([evals.hunter.kills_per_death.toFixed(1), "kills per death vs hunter bots"]);
  }
  if (evals?.forager) tiles.push([`${Math.round(100 * evals.forager.mass_share)}%`, "of all mass vs 7 forager bots"]);
  $("results").innerHTML = tiles.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
})();
```

Before writing the page, check `web/index.html` line 11 for the current
`style.css?v=N` version and use the same `N` in the `../style.css?v=` link above.

- [ ] **Step 6: Add the link from the grid page**

In `web/index.html`, inside `<div class="wrap bar">`, insert one line between the
`brand` span and the GitHub link:

```html
      <a class="quiet" href="slither/">Slither →</a>
```

Then run: `git diff --stat web/index.html`
Expected: `1 file changed, 1 insertion(+)`, with no deletions.

- [ ] **Step 7: Verify in the browser**

Serve `web/`. Use the preview tool with a `.claude/launch.json` entry that runs
`python3 -m http.server --directory web 8000` on port 8000, or run that command in a
terminal. Then open `http://localhost:8000/slither/` and check each of these:

1. There are no console errors, and the network loads (the overlay disappears).
2. **Watch:** ten snakes move smoothly, and the camera follows a snake. Clicking another
   snake, or a leaderboard row, switches the camera to it. The leaderboard and fan update,
   and Boost % and Value show numbers.
3. **Play:** the "You" label appears, and the snake turns toward the mouse at a capped
   rate. Holding the mouse button or Space makes it visibly faster and drops pellets.
   Dying shows the overlay with mass, kills and time; "Play again" respawns you while the
   AIs keep playing.
4. **Dark mode:** the theme toggle recolours the arena on the next frame.
5. **Mobile:** at a 375 px wide viewport, the panel stacks under the arena, and in Play
   mode the Boost button appears and touch steers.
6. **Pause off-screen:** scroll the arena off-screen and back. The simulation pauses
   and resumes without a burst of catch-up ticks.
7. **Grid page:** `http://localhost:8000/` shows the "Slither →" link, and the grid demo
   still works.
8. **Embed:** `http://localhost:8000/slither/?embed=1` hides the page chrome.

Take one screenshot each of Watch and Play to show the user.

- [ ] **Step 8: Commit (without the placeholder model)**

```bash
git add web/slither/render.js web/slither/slither-demo.js web/slither/slither-demo.css web/slither/index.html web/slither/page.js web/index.html
git commit -m "Slither: browser component, standalone page and link from the grid demo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Check `git status --short web/slither`. It should list `model.onnx` and `stats.json` as
untracked, and they must stay uncommitted until Task 14.

---

### Task 14: Train, evaluate, publish assets, README

**Files:**
- Create:
  - `results/slither_eval.json`, `results/slither_training.csv`, `results/slither_config.json`;
  - `web/slither/model.onnx`, `web/slither/stats.json`.
- Modify: `README.md`, adding a new section only.

**Interfaces:**
- Consumes all of the earlier tasks.

- [ ] **Step 1: Run the full test suite**

Run: `uv run --group train pytest -q`
Expected: everything passes, including every pre-existing test, unmodified.

- [ ] **Step 2: Start the main training run in the background**

```bash
mkdir -p runs/slither
uv run --group train python scripts/train_slither.py --run-dir runs/slither --device mps --hours 2.5 > runs/slither/train.log 2>&1
```

Run it in the background, since it takes about 2.5 h. Don't poll it in a tight loop.
Check `tail -5 runs/slither/train.log` occasionally, at least 20 minutes apart.

What healthy looks like:

- `[eval]` lines every 25 iterations;
- `mass share vs hunter` trending above 0.125;
- entropy declining slowly from about 2.9 (ln 18);
- `approx_kl` mostly below 0.05.

- [ ] **Step 3: Final evaluation (300 arenas × 3,000 decisions vs each bot)**

```bash
uv run --group train python scripts/eval_slither.py runs/slither/best.pt --arenas 300 --decisions 3000 --out results/slither_eval.json
```

Run this in the background too; it takes tens of minutes. **Success criterion (spec):**
the lower end of `mass_share_ci` against `hunter` is above 0.125. The stretch target is
`mass_share` ≥ 0.25.

**If the criterion is not met,** make at most two more training runs, each changing one
thing, and evaluate after each:

1. `--survival-bonus 0.002`;
2. `--hours 5`.

If it's still not met, stop and report the numbers to the user honestly, with the
training curve. Do not change the evaluation.

- [ ] **Step 4: Export the browser assets and copy the results**

```bash
uv run --group train python scripts/build_slither_assets.py --run-dir runs/slither --final-eval results/slither_eval.json
cp runs/slither/metrics.csv results/slither_training.csv
cp runs/slither/config.json results/slither_config.json
```

Reload `http://localhost:8000/slither/`. The Results tiles should now show the final
numbers, and the snakes should visibly forage, avoid walls and (hopefully) cut each
other off.

- [ ] **Step 5: Add the README section**

Insert a new `## Slither mode` section right after the existing `### Arena: 6 snakes on
a 19×19 board` results (before `## How it works`). Don't edit or remove any existing
line. Fill in the numbers from `results/slither_eval.json`, the benchmark rate from
Task 8 and the run length from `results/slither_training.csv`:

````markdown
## Slither mode

A second, continuous game in the style of slither.io: a circular arena, smooth steering,
boosting that costs mass, and snakes that collapse into food when they crash.
**[▶ Play it](https://lambdaak.github.io/SnakeDQN/slither/)**

One network snake plays against 7 copies of a scripted bot in 300 arenas, for 5 minutes
of game time each. An equal snake would hold 12.5% of all the snake mass.

| Opponents (×7) | Share of all snake mass | Kills per death |
|---|---|---|
| Hunter bots (forage, and boost to cut off smaller snakes) | **{hunter share}%** ({lo}–{hi}) | {kpd} ({lo}–{hi}) |
| Forager bots (steer to the most food, avoid danger) | **{forager share}%** ({lo}–{hi}) | {kpd} ({lo}–{hi}) |
| Wander bots (random, avoid danger) | **{wander share}%** ({lo}–{hi}) | {kpd} ({lo}–{hi}) |

How it differs from the grid game:

- **Physics:** each snake is a chain of points 1 unit apart. The head moves 1 unit per
  tick (2 when boosting) and turns at most 0.15 rad per tick. Physics runs at 30 Hz and
  the network decides at 10 Hz.
- **Speed and verification:** `snake_rl/slither` runs {rate} snake decisions per second in
  vectorised NumPy. It is checked tick for tick against a plain-Python reference, and
  against the browser's JavaScript port.
- **Senses and actions:**
  - The network sees 32 sectors around its head, each recording the nearest wall, body
    and head, and the food in it.
  - It also sees its own size and where the centre is.
  - It picks one of 9 steering angles, with or without boost.
  - It is a 2-layer MLP.
- **Training:**
  - PPO self-play in arenas of 6–12 snakes.
  - Each life is played by the learner or by a frozen past version.
  - The reward is mass gained, minus 1 for dying.
  - It trained for {hours} h on an M3 Max.

```bash
uv run python scripts/benchmark_slither.py
uv run --group train python scripts/train_slither.py --device mps --hours 2.5
uv run --group train python scripts/eval_slither.py runs/slither/best.pt
uv run --group train python scripts/build_slither_assets.py --run-dir runs/slither \
    --final-eval results/slither_eval.json                    # -> web/slither/model.onnx, stats.json
```
````

Round shares to one decimal place, and write CIs in the same style as the existing
tables. Replace every `{…}` with a real number. None may remain.

Run: `grep -n "{" README.md | grep -i slither`
Expected: no output.

- [ ] **Step 6: Run the final checks and commit**

Run: `uv run --group train pytest -q`
Expected: all pass.

Run: `git diff --stat HEAD -- README.md web/index.html`
Expected: insertions only.

```bash
git add results/slither_eval.json results/slither_training.csv results/slither_config.json web/slither/model.onnx web/slither/stats.json README.md
git commit -m "Slither: trained network, final evaluation and README section

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

**Don't push.** Pushing to `main` redeploys the public demo through GitHub Pages, so ask
the user first.
