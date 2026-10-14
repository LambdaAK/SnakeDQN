import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from snake_rl.env import SnakeEnv, build_observations
from snake_rl.env.observations import NUM_CHANNELS, OWN_HEAD
from snake_rl.env.rewards import food, winloss

ROOT = Path(__file__).resolve().parents[1]


def test_reset_returns_obs_and_info():
    env = SnakeEnv(4, 2, seed=0)
    obs, info = env.reset()
    assert obs.shape == (4, 2, NUM_CHANNELS, 11, 11)
    assert info["action_mask"].shape == (4, 2, 4)
    assert set(info) >= {"action_mask", "length", "health", "alive", "turn"}


def test_step_before_reset_raises():
    env = SnakeEnv(2, 1, seed=0)
    with pytest.raises(RuntimeError):
        env.step(np.zeros((2, 1), int))


def test_default_reward_depends_on_snake_count():
    assert SnakeEnv(1, 1).reward_fn is food
    assert SnakeEnv(1, 2).reward_fn is winloss


def test_unknown_reward_raises():
    with pytest.raises(ValueError):
        SnakeEnv(1, 1, reward="bogus")


def test_auto_reset_after_game_over():
    env = SnakeEnv(8, 1, seed=0)
    env.reset()
    seen = np.zeros(8, bool)
    for _ in range(12):  # moving up hits the top wall within 10 turns on 11x11
        obs, rewards, dones, info = env.step(np.zeros((8, 1), int))
        for e in np.flatnonzero(info["game_over"] & ~seen):
            assert dones[e, 0] and rewards[e, 0] == -1.0
            assert info["final_turn"][e] > 0 and info["turn"][e] == 0
            assert info["final_obs"][e, 0, OWN_HEAD].sum() == 0  # dead in the final frame
            assert obs[e, 0, OWN_HEAD].sum() == 1                # alive in the new game
            assert np.array_equal(obs[e], build_observations(env.state, [e])[0])
        seen |= info["game_over"]
    assert seen.all()


def test_same_seed_same_trajectory():
    def run():
        env = SnakeEnv(16, 2, seed=123)
        obs, info = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs, _, _, info = env.step(rng.integers(0, 4, size=(16, 2)))
        return obs

    assert np.array_equal(run(), run())


def test_two_snake_game_reports_winner():
    env = SnakeEnv(64, 2, seed=1)
    env.reset()
    rng = np.random.default_rng(1)
    winners = []
    for _ in range(200):
        _, rewards, _, info = env.step(rng.integers(0, 4, size=(64, 2)))
        for e in np.flatnonzero(info["game_over"]):
            w = info["winner"][e]
            winners.append(w)
            if w >= 0:
                assert rewards[e, w] == 1.0
    assert winners and set(winners) <= {-1, 0, 1}


def test_benchmark_script_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark_env.py"),
         "--num-envs", "8", "--steps", "5"],
        capture_output=True, text=True, check=True,
    )
    assert "env steps/sec" in result.stdout


def test_multi_snake_done_and_active_contract():
    n, s = 64, 3
    env = SnakeEnv(n, s, seed=5)
    _, info = env.reset()
    rng = np.random.default_rng(5)
    ended_games = 0
    partial_deaths = 0
    for _ in range(300):
        prev_alive = info["alive"]
        mask = info["action_mask"]
        scores = rng.random((n, s, 4)) + mask * 2.0  # prefer safe moves
        actions = scores.argmax(axis=2)
        _, rewards, dones, info = env.step(actions)
        active = info["active"]
        assert active.shape == (n, s) and active.dtype == bool
        assert not (active & ~prev_alive).any()      # active implies alive before the step
        assert (active == prev_alive).all()          # no game is over between steps (auto-reset)
        assert (rewards[~active] == 0).all()
        assert not dones[~active & ~info["game_over"][:, None]].any()
        over = info["game_over"]
        ended_games += int(over.sum())
        assert dones[over].all()                     # every slot of an ending game is done
        partial_deaths += int((dones & ~over[:, None]).sum())
    assert ended_games > 0 and partial_deaths > 0
