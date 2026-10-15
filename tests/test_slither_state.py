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
