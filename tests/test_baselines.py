import numpy as np
import pytest

from snake_rl.baselines import (
    BOT_NAMES,
    GreedyBot,
    HamiltonianBot,
    RandomBot,
    SafeRandomBot,
    hamiltonian_cycle,
    make_bot,
)
from snake_rl.env import SnakeEnv
from snake_rl.env.rules import GameState, action_mask


def test_random_bot_returns_valid_actions():
    env = SnakeEnv(32, 2, seed=0)
    env.reset()
    actions = RandomBot(0)(env.state, 1)
    assert actions.shape == (32,)
    assert ((actions >= 0) & (actions <= 3)).all()


def test_safe_random_bot_respects_mask():
    env = SnakeEnv(32, 2, seed=0)
    env.reset()
    bot = SafeRandomBot(0)
    for _ in range(50):
        mask = action_mask(env.state)[:, 0]
        actions = bot(env.state, 0)
        has_safe = mask.any(axis=1)
        assert mask[np.arange(32), actions][has_safe].all()
        env.step(np.stack([actions, bot(env.state, 1)], axis=1))


def test_greedy_bot_heads_to_food():
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(0, 3)])
    assert GreedyBot(0)(state, 0)[0] == 0
    state = GameState.from_layout(7, [[(3, 3), (4, 3), (5, 3)]], food=[(3, 6)])
    assert GreedyBot(0)(state, 0)[0] == 3


def test_greedy_bot_avoids_immediate_death():
    # Food straight behind the neck: the only way to it is a fatal reversal.
    state = GameState.from_layout(7, [[(0, 3), (1, 3), (2, 3)]], food=[(6, 3)])
    action = GreedyBot(0)(state, 0)[0]
    assert action in (2, 3)


def test_hamiltonian_cycle_covers_board():
    cycle = hamiltonian_cycle(8)
    assert len(cycle) == 64 and len(set(cycle)) == 64
    for (r1, c1), (r2, c2) in zip(cycle, cycle[1:] + cycle[:1]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_hamiltonian_rejects_odd_board():
    with pytest.raises(ValueError):
        HamiltonianBot(11)


def test_hamiltonian_bot_fills_the_board():
    env = SnakeEnv(2, 1, board_size=8, max_health=64, seed=0)
    env.reset()
    bot = HamiltonianBot(8)
    final = np.zeros(2, int)
    done = np.zeros(2, bool)
    while not done.all():
        _, _, _, info = env.step(bot(env.state, 0)[:, None])
        new = info["game_over"] & ~done
        final[new] = info["final_length"][new, 0]
        done |= new
    # All 64 cells are filled; length counts the tail segment duplicated by the final meal.
    assert (final == 8 * 8 + 1).all()


def test_make_bot():
    for name in BOT_NAMES:
        assert make_bot(name, seed=0).name == name
    assert make_bot("hamiltonian", board_size=8).name == "hamiltonian"


def test_make_bot_unknown_name():
    with pytest.raises(ValueError):
        make_bot("nope")
