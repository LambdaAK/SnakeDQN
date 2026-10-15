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
