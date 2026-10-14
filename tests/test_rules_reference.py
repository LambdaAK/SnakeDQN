import numpy as np
import pytest

from reference_rules import from_state, ref_step
from snake_rl.env.rules import GameState, action_mask, reset_games, step_games


def assert_game_equal(state, env, game):
    assert int(state.turn[env]) == game.turn
    assert bool(state.over[env]) == game.over
    actual_food = {(int(r), int(c)) for r, c in zip(*state.food[env].nonzero())}
    assert actual_food == game.food
    for k, snake in enumerate(game.snakes):
        length = int(state.length[env, k])
        segments = [tuple(int(v) for v in p) for p in state.body[env, k, :length]]
        assert segments == snake.body, f"env {env} snake {k}"
        assert int(state.health[env, k]) == snake.health
        assert bool(state.alive[env, k]) == snake.alive


def random_actions(state, rng, safe_prob=0.9):
    mask = action_mask(state)
    safe = np.argmax(rng.random(mask.shape) + mask, axis=-1)
    wild = rng.integers(0, 4, size=safe.shape)
    return np.where(rng.random(safe.shape) < safe_prob, safe, wild)


@pytest.mark.parametrize("board_size,num_snakes", [(7, 1), (11, 2), (11, 4), (8, 3)])
def test_vectorized_rules_match_reference(board_size, num_snakes):
    rng = np.random.default_rng(board_size * 10 + num_snakes)
    state = GameState.empty(32, num_snakes, board_size)
    for _ in range(150):
        reset_games(state, state.over, rng)
        games = [from_state(state, env) for env in range(state.num_envs)]
        actions = random_actions(state, rng)
        spawn_u = rng.random((state.num_envs, 2))
        step_games(state, actions, spawn_u)
        for env, game in enumerate(games):
            ref_step(game, actions[env], spawn_u[env])
            assert_game_equal(state, env, game)


def test_batched_step_equals_individual_steps():
    rng = np.random.default_rng(7)
    state = GameState.empty(256, 2, 11)
    for _ in range(60):
        reset_games(state, state.over, rng)
        singles = [state.take([env]) for env in range(state.num_envs)]
        actions = random_actions(state, rng)
        spawn_u = rng.random((state.num_envs, 2))
        step_games(state, actions, spawn_u)
        for env, single in enumerate(singles):
            step_games(single, actions[env:env + 1], spawn_u[env:env + 1])
            game = from_state(single, 0)
            assert_game_equal(state, env, game)
