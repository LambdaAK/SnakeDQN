import numpy as np
import pytest

from snake_rl.env.rules import (
    DIAGONALS,
    START_LENGTH,
    GameState,
    reset_games,
    start_points,
)


def fresh(num_envs=64, num_snakes=4, board_size=11, seed=0):
    state = GameState.empty(num_envs, num_snakes, board_size)
    reset_games(state, np.ones(num_envs, bool), np.random.default_rng(seed))
    return state


def test_empty_validates_arguments():
    with pytest.raises(ValueError):
        GameState.empty(1, 1, 6)
    with pytest.raises(ValueError):
        GameState.empty(1, 9, 11)
    with pytest.raises(ValueError):
        GameState.empty(1, 0, 11)
    with pytest.raises(ValueError):
        GameState.empty(0, 1, 11)


def test_empty_state_is_over_until_reset():
    state = GameState.empty(3, 2, 11)
    assert state.over.all()
    assert not state.alive.any()
    assert state.body.shape == (3, 2, 11 * 11 + 1, 2)


def test_reset_places_stacked_snakes_on_distinct_start_points():
    state = fresh()
    points = {tuple(p) for p in start_points(11)}
    assert (state.length == START_LENGTH).all()
    assert (state.health == 100).all()
    assert state.alive.all()
    assert not state.over.any()
    assert (state.turn == 0).all()
    for env in range(state.num_envs):
        heads = [tuple(state.body[env, k, 0]) for k in range(4)]
        assert len(set(heads)) == 4
        assert set(heads) <= points
        for k in range(4):
            segments = state.body[env, k, :START_LENGTH]
            assert (segments == segments[0]).all()


def test_reset_places_centre_food_and_one_diagonal_food_per_snake():
    state = fresh()
    centre = 11 // 2
    for env in range(state.num_envs):
        assert state.food[env, centre, centre]
        for k in range(4):
            head = state.body[env, k, 0]
            diagonal_food = [state.food[env, head[0] + dr, head[1] + dc] for dr, dc in DIAGONALS]
            assert any(diagonal_food)
        assert 2 <= state.food[env].sum() <= 5


def test_reset_only_touches_masked_games():
    state = fresh(num_envs=4, num_snakes=2)
    before = state.take([1, 3])
    mask = np.array([True, False, True, False])
    reset_games(state, mask, np.random.default_rng(1))
    after = state.take([1, 3])
    assert np.array_equal(before.body, after.body)
    assert np.array_equal(before.food, after.food)


def test_from_layout_and_take():
    state = GameState.from_layout(
        7, snakes=[[(3, 3), (4, 3), (5, 3)]], food=[(0, 0)], health=[42]
    )
    assert state.num_envs == 1 and state.num_snakes == 1 and state.board_size == 7
    assert state.length[0, 0] == 3 and state.health[0, 0] == 42
    assert state.food[0, 0, 0] and state.food.sum() == 1
    assert not state.over[0]
    copy = state.take([0])
    copy.health[0, 0] = 1
    assert state.health[0, 0] == 42


def test_eight_snakes_fit_on_smallest_board():
    state = fresh(num_envs=8, num_snakes=8, board_size=7)
    for env in range(8):
        heads = {tuple(state.body[env, k, 0]) for k in range(8)}
        assert len(heads) == 8


def _groups(board_size):
    a, m, z = 1, board_size // 2, board_size - 2
    corners = {(a, a), (a, z), (z, a), (z, z)}
    cardinals = {(a, m), (m, a), (m, z), (z, m)}
    return corners, cardinals


@pytest.mark.parametrize("num_snakes", [2, 3, 4])
def test_starts_are_all_corners_or_all_cardinals(num_snakes):
    state = fresh(num_envs=200, num_snakes=num_snakes)
    corners, cardinals = _groups(11)
    kinds = set()
    for env in range(state.num_envs):
        heads = {tuple(int(x) for x in state.body[env, k, 0]) for k in range(num_snakes)}
        assert len(heads) == num_snakes
        assert heads <= corners or heads <= cardinals
        kinds.add("corners" if heads <= corners else "cardinals")
    assert kinds == {"corners", "cardinals"}


def test_more_than_four_snakes_use_chosen_group_first():
    state = fresh(num_envs=100, num_snakes=6)
    corners, cardinals = _groups(11)
    for env in range(state.num_envs):
        heads = [tuple(int(x) for x in state.body[env, k, 0]) for k in range(6)]
        first = set(heads[:4])
        assert first == corners or first == cardinals
        assert len(set(heads)) == 6


def _allowed_diagonals(head, b):
    centre = b // 2
    dist = abs(head[0] - centre) + abs(head[1] - centre)
    out = []
    for dr, dc in DIAGONALS:
        r, c = head[0] + int(dr), head[1] + int(dc)
        if (r in (0, b - 1)) and (c in (0, b - 1)):
            continue
        if abs(r - centre) + abs(c - centre) < dist:
            continue
        out.append((r, c))
    return out


@pytest.mark.parametrize("board_size", [7, 8, 11])
def test_every_start_point_has_an_allowed_diagonal(board_size):
    for p in start_points(board_size):
        assert _allowed_diagonals((int(p[0]), int(p[1])), board_size)


@pytest.mark.parametrize("board_size", [7, 8, 11])
def test_initial_snake_food_is_on_allowed_diagonal(board_size):
    state = fresh(num_envs=200, num_snakes=4, board_size=board_size, seed=3)
    centre = board_size // 2
    for env in range(state.num_envs):
        allowed_all = set()
        for k in range(4):
            head = (int(state.body[env, k, 0, 0]), int(state.body[env, k, 0, 1]))
            allowed = _allowed_diagonals(head, board_size)
            allowed_all |= set(allowed)
            assert any(state.food[env, r, c] for r, c in allowed)
        foods = {(int(r), int(c)) for r, c in np.argwhere(state.food[env])}
        assert foods - {(centre, centre)} <= allowed_all
