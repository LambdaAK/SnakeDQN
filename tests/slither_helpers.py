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
