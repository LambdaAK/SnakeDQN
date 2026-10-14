import numpy as np
import pytest

from snake_rl.env.rules import GameState, step_games

NO_SPAWN = (0.99, 0.0)  # no random spawn; a forced spawn picks the first free cell


def step(state, *actions, spawn=NO_SPAWN):
    return step_games(state, np.array([actions]), np.array([spawn]))


def body(state, k):
    return [tuple(int(v) for v in p) for p in state.body[0, k, :state.length[0, k]]]


def test_move_shifts_body_and_costs_health():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)])
    events = step(state, 0)
    assert body(state, 0) == [(2, 3), (3, 3), (4, 3)]
    assert state.health[0, 0] == 99
    assert state.turn[0] == 1
    assert not events.died.any() and not events.game_over.any()


def test_wall_collision_kills():
    state = GameState.from_layout(7, [[(0, 3), (1, 3), (2, 3)]], food=[(6, 6)])
    events = step(state, 0)
    assert events.died[0, 0] and not state.alive[0, 0]
    assert events.game_over[0] and state.over[0]
    assert events.winner[0] == -1


def test_self_collision_kills():
    state = GameState.from_layout(7, [[(2, 2), (2, 3), (3, 3), (3, 2), (4, 2)]], food=[(6, 6)])
    events = step(state, 1)  # down into (3, 2), still part of the body
    assert events.died[0, 0]


def test_neck_reversal_kills():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)])
    events = step(state, 1)
    assert events.died[0, 0]


def test_moving_into_vacating_tail_is_safe():
    state = GameState.from_layout(7, [[(2, 2), (2, 3), (3, 3), (3, 2)]], food=[(6, 6)])
    events = step(state, 1)
    assert not events.died.any()
    assert body(state, 0) == [(3, 2), (2, 2), (2, 3), (3, 3)]


def test_tail_that_just_ate_is_fatal():
    # A duplicated tail segment is what eating leaves behind.
    state = GameState.from_layout(
        7, [[(2, 2), (2, 3), (3, 3), (3, 2), (3, 2)]], food=[(6, 6)]
    )
    events = step(state, 1)
    assert events.died[0, 0]


def test_eating_grows_duplicates_tail_and_restores_health():
    state = GameState.from_layout(
        7, [[(3, 3), (4, 3), (5, 3)]], food=[(2, 3), (0, 0)], health=[50]
    )
    events = step(state, 0)
    assert events.ate[0, 0]
    assert state.length[0, 0] == 4
    assert state.health[0, 0] == 100
    assert body(state, 0) == [(2, 3), (3, 3), (4, 3), (4, 3)]
    assert not state.food[0, 2, 3]


def test_starvation_kills():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)], health=[1])
    events = step(state, 0)
    assert events.died[0, 0]


def test_eating_on_last_health_point_survives():
    state = GameState.from_layout(
        7, [[(3, 3), (4, 3), (5, 3)]], food=[(2, 3), (0, 0)], health=[1]
    )
    events = step(state, 0)
    assert not events.died.any()
    assert state.health[0, 0] == 100


def test_head_to_head_shorter_snake_dies():
    state = GameState.from_layout(
        7,
        [[(3, 2), (3, 1), (3, 0)], [(3, 4), (3, 5), (3, 6), (2, 6)]],
        food=[(0, 0)],
    )
    events = step(state, 3, 2)
    assert events.died[0].tolist() == [True, False]
    assert events.game_over[0] and events.winner[0] == 1


def test_head_to_head_equal_length_both_die():
    state = GameState.from_layout(
        7, [[(3, 2), (3, 1), (3, 0)], [(3, 4), (3, 5), (3, 6)]], food=[(0, 0)]
    )
    events = step(state, 3, 2)
    assert events.died[0].tolist() == [True, True]
    assert events.game_over[0] and events.winner[0] == -1


def test_moving_into_other_snakes_body_kills():
    state = GameState.from_layout(
        7, [[(2, 3), (1, 3), (0, 3)], [(3, 4), (3, 3), (3, 2)]], food=[(6, 6)]
    )
    events = step(state, 1, 3)
    assert events.died[0].tolist() == [True, False]
    assert events.winner[0] == 1


def test_moving_into_other_snakes_vacating_tail_is_safe():
    state = GameState.from_layout(
        7, [[(2, 3), (1, 3), (0, 3)], [(3, 5), (3, 4), (3, 3)]], food=[(6, 6)]
    )
    events = step(state, 1, 3)
    assert not events.died.any()
    assert not events.game_over[0]


def test_dead_snake_action_ignored():
    state = GameState.from_layout(
        7,
        [[(3, 3), (4, 3), (5, 3)], [(1, 1), (1, 2), (1, 3)], [(5, 5), (5, 6), (6, 6)]],
        food=[(0, 6)],
    )
    state.alive[0, 2] = False
    before = body(state, 2)
    events = step(state, 0, 1, 0)
    assert body(state, 2) == before
    assert not events.alive_before[0, 2] and not events.died[0, 2]


def test_finished_game_does_not_change():
    state = GameState.from_layout(7, [[(0, 3), (1, 3), (2, 3)]], food=[(6, 6)])
    step(state, 0)
    snapshot = state.take([0])
    events = step(state, 0)
    assert state.turn[0] == snapshot.turn[0]
    assert np.array_equal(state.body, snapshot.body)
    assert np.array_equal(state.food, snapshot.food)
    assert not events.game_over[0] and not events.died.any() and not events.alive_before.any()


def test_forced_spawn_when_no_food_left():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(2, 3)])
    step(state, 0, spawn=(0.99, 0.0))
    assert state.food[0].sum() == 1
    assert state.food[0, 0, 0]  # first free cell in row-major order


def test_chance_spawn():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(6, 6)])
    step(state, 0, spawn=(0.2, 0.0))
    assert state.food[0].sum() == 1
    step(state, 0, spawn=(0.1, 0.0))
    assert state.food[0].sum() == 2 and state.food[0, 0, 0]


def test_spawn_fraction_selects_cell():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(6, 6)])
    step(state, 0, spawn=(0.1, 0.999999))
    assert state.food[0, 6, 5]  # last free cell, since (6, 6) already holds food


def test_full_board_spawns_nothing():
    path = []
    for r in range(7):
        cols = range(7) if r % 2 == 0 else range(6, -1, -1)
        path += [(r, c) for c in cols]
    segments = list(reversed(path)) + [path[0]]  # head (6, 6), duplicated tail
    state = GameState.from_layout(7, [segments])
    events = step(state, 1)  # off the board
    assert events.died[0, 0]
    assert state.food[0].sum() == 0


def test_invalid_actions_raise():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)])
    with pytest.raises(ValueError):
        step_games(state, np.array([[4]]), np.array([NO_SPAWN]))
    with pytest.raises(ValueError):
        step_games(state, np.array([[-1]]), np.array([NO_SPAWN]))
    with pytest.raises(ValueError):
        step_games(state, np.array([0]), np.array([NO_SPAWN]))
    with pytest.raises(ValueError):
        step_games(state, np.array([[0.0]]), np.array([NO_SPAWN]))
