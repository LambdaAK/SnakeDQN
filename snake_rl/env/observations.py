"""Egocentric grid observations: each snake sees the board as 'self' versus 'enemies'."""

from __future__ import annotations

import numpy as np

from snake_rl.env.rules import GameState

NUM_CHANNELS = 8
(OWN_HEAD, OWN_BODY, OWN_AGE, ENEMY_HEADS, ENEMY_BODIES,
 DANGEROUS_HEADS, FOOD, HEALTH) = range(NUM_CHANNELS)


def _snake_layers(state: GameState, out: np.ndarray):
    """Fill the head, body and age channels of `out` (zeroed, (N, S, C, B, B)) and return
    them as views. Dead snakes are blank."""
    head, body, age = out[:, :, OWN_HEAD], out[:, :, OWN_BODY], out[:, :, OWN_AGE]
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
    out = np.zeros((n, s, NUM_CHANNELS, b, b), np.float32)
    head, body, age = _snake_layers(state, out)
    out[:, :, ENEMY_HEADS] = head.sum(axis=1, keepdims=True) - head
    out[:, :, ENEMY_BODIES] = body.sum(axis=1, keepdims=True) - body
    threat = (
        (state.length[:, None, :] >= state.length[:, :, None])
        & state.alive[:, None, :]
        & ~np.eye(s, dtype=bool)
    )  # (N, viewer, other)
    out[:, :, DANGEROUS_HEADS] = np.einsum("nvk,nkhw->nvhw", threat.astype(np.float32), head)
    out[:, :, FOOD] = state.food[:, None]
    out[:, :, HEALTH] = (state.health * state.alive / state.max_health)[:, :, None, None]
    return out
