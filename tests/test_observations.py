import numpy as np

from snake_rl.env.observations import (
    DANGEROUS_HEADS,
    ENEMY_BODIES,
    ENEMY_HEADS,
    FOOD,
    HEALTH,
    NUM_CHANNELS,
    OWN_AGE,
    OWN_BODY,
    OWN_HEAD,
    build_observations,
)
from snake_rl.env.rules import GameState, reset_games

SNAKE_A = [(3, 3), (4, 3), (5, 3)]
SNAKE_B = [(1, 1), (1, 2), (1, 3), (1, 4)]


def cells(plane):
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(plane))}


def two_snake_state():
    return GameState.from_layout(7, [SNAKE_A, SNAKE_B], food=[(0, 6)], health=[50, 100])


def test_shape_and_dtype():
    state = GameState.empty(5, 3, 11)
    reset_games(state, np.ones(5, bool), np.random.default_rng(0))
    obs = build_observations(state)
    assert obs.shape == (5, 3, NUM_CHANNELS, 11, 11)
    assert obs.dtype == np.float32


def test_channels_from_first_snakes_view():
    obs = build_observations(two_snake_state())[0, 0]
    assert cells(obs[OWN_HEAD]) == {(3, 3)}
    assert cells(obs[OWN_BODY]) == {(4, 3), (5, 3)}
    assert obs[OWN_AGE, 3, 3] == 1.0
    assert np.isclose(obs[OWN_AGE, 4, 3], 2 / 3)
    assert np.isclose(obs[OWN_AGE, 5, 3], 1 / 3)
    assert cells(obs[ENEMY_HEADS]) == {(1, 1)}
    assert cells(obs[ENEMY_BODIES]) == {(1, 2), (1, 3), (1, 4)}
    assert cells(obs[DANGEROUS_HEADS]) == {(1, 1)}  # enemy is longer
    assert cells(obs[FOOD]) == {(0, 6)}
    assert np.all(obs[HEALTH] == 0.5)


def test_shorter_enemy_head_is_not_dangerous():
    obs = build_observations(two_snake_state())[0, 1]
    assert cells(obs[ENEMY_HEADS]) == {(3, 3)}
    assert not obs[DANGEROUS_HEADS].any()
    assert np.all(obs[HEALTH] == 1.0)


def test_observations_are_egocentric():
    swapped = GameState.from_layout(7, [SNAKE_B, SNAKE_A], food=[(0, 6)], health=[100, 50])
    original = build_observations(two_snake_state())
    flipped = build_observations(swapped)
    assert np.array_equal(original[0, 0], flipped[0, 1])
    assert np.array_equal(original[0, 1], flipped[0, 0])


def test_dead_snakes_are_invisible():
    state = two_snake_state()
    state.alive[0, 1] = False
    obs = build_observations(state)
    assert not obs[0, 0, ENEMY_HEADS].any() and not obs[0, 0, ENEMY_BODIES].any()
    assert not obs[0, 1, OWN_HEAD].any() and not obs[0, 1, HEALTH].any()


def test_subset_of_envs():
    state = GameState.empty(8, 2, 11)
    reset_games(state, np.ones(8, bool), np.random.default_rng(3))
    full = build_observations(state)
    assert np.array_equal(build_observations(state, np.array([2, 5])), full[[2, 5]])


def test_window_observation_centres_on_head_and_marks_walls():
    from snake_rl.env.observations import WALL, build_window_observations

    state = two_snake_state()  # 7x7, snake 0 head at (3, 3), snake 1 head at (1, 1)
    full = build_observations(state)
    win = build_window_observations(state, 9)
    assert win.shape == (1, 2, NUM_CHANNELS + 1, 9, 9)
    # Snake 0's head (3, 3) sits at the window centre (4, 4): offset +1 in both axes.
    assert win[0, 0, OWN_HEAD, 4, 4] == 1
    assert np.array_equal(win[0, 0, :NUM_CHANNELS, 1:8, 1:8], full[0, 0])
    assert win[0, 0, WALL, 0].all() and win[0, 0, WALL, :, 8].all()
    assert not win[0, 0, WALL, 1:8, 1:8].any()
    # Snake 1's head (1, 1): window rows/cols 0-2 lie off the board.
    assert win[0, 1, OWN_HEAD, 4, 4] == 1
    assert win[0, 1, WALL, :3].all() and win[0, 1, WALL, :, :3].all()
    assert not win[0, 1, WALL, 3:, 3:].any()
    assert np.array_equal(win[0, 1, :NUM_CHANNELS, 3:, 3:], full[0, 1, :, :6, :6])
