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
