import numpy as np

from snake_rl.baselines import BOT_NAMES, FloodFillBot, make_bot
from snake_rl.env import SnakeEnv
from snake_rl.env.rules import GameState, action_mask


def test_refuses_to_enter_a_dead_end_pocket_even_for_food():
    # Moving left enters a 2-cell pocket (0,0)-(0,1) holding food; right is open.
    state = GameState.from_layout(
        7, [[(0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (3, 0)]], food=[(0, 0)], health=[10]
    )
    assert FloodFillBot(0)(state, 0)[0] == 3


def test_avoids_head_to_head_with_a_longer_snake():
    state = GameState.from_layout(
        7,
        [[(3, 3), (4, 3), (5, 3)], [(1, 3), (1, 4), (1, 5), (1, 6)]],
        food=[(2, 3)],
        health=[10, 100],
    )
    for seed in range(5):
        assert FloodFillBot(seed)(state, 0)[0] in (2, 3)


def test_only_picks_safe_moves_when_available():
    env = SnakeEnv(16, 2, seed=0)
    env.reset()
    bot = FloodFillBot(0)
    for _ in range(40):
        mask = action_mask(env.state)
        actions = np.stack([bot(env.state, 0), bot(env.state, 1)], axis=1)
        for k in range(2):
            has_safe = mask[:, k].any(axis=1)
            assert mask[np.arange(16), k, actions[:, k]][has_safe].all()
        env.step(actions)


def test_registered():
    assert "flood_fill" in BOT_NAMES
    assert make_bot("flood_fill", seed=0).name == "flood_fill"
