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
