import math
from dataclasses import replace

import numpy as np
import pytest

from slither_helpers import QUIET, dead, scene, snake, state_of, straight
from snake_rl.slither.config import SlitherConfig, u_layout
from snake_rl.slither.rules import (SlitherState, head_points, logical_body, reset_arenas,
                                    snapshot, step_tick)

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
