"""Scene builders shared by the slither tests (snapshot dicts, see rules.snapshot)."""

import math
from dataclasses import replace

import numpy as np

from snake_rl.slither.config import SlitherConfig, u_layout
from snake_rl.slither.rules import SlitherState, from_snapshots, reset_arenas

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
