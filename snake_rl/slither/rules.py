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


@dataclass
class TickEvents:
    died: np.ndarray    # bool (N, S) died this tick
    killer: np.ndarray  # int32 (N, S) slot whose body it hit; -1 for wall/head-on deaths
    kills: np.ndarray   # int32 (N, S) kills credited this tick
    eaten: np.ndarray   # float64 (N, S) pellet value eaten this tick


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


def _near_pairs_dense(pe, px, py, heads, live, reach):
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
