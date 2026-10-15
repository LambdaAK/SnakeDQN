from dataclasses import replace

import numpy as np
import pytest

from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.env import SlitherEnv

# Small arena so random play produces deaths and kills within a few dozen decisions.
CROWDED = replace(SlitherConfig(), arena_radius=120.0, respawn_margin=30.0, respawn_clearance=20.0,
                  natural_food_target=60)


def test_reset_shapes_and_enabled_counts():
    env = SlitherEnv(16, max_snakes=12, min_snakes=6, seed=0)
    obs, info = env.reset()
    assert obs.shape == (16, 12, 263) and obs.dtype == np.float32
    counts = info["enabled"].sum(axis=1)
    assert counts.min() >= 6 and counts.max() <= 12
    assert (info["alive"] == info["enabled"]).all()
    assert not obs[~info["enabled"]].any()
    assert info["action_mask"].shape == (16, 12, 18)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        SlitherEnv(2, seed=0).step(np.zeros((2, 12), int))


def test_rewards_follow_mass_change_and_death():
    env = SlitherEnv(32, max_snakes=8, min_snakes=8, config=CROWDED, seed=1, survival_bonus=0.001)
    obs, info = env.reset()
    rng = np.random.default_rng(1)
    seen_death = False
    for _ in range(60):
        before = info["mass"].copy()
        actions = rng.integers(0, 18, (32, 8))
        obs, rewards, dones, info = env.step(actions)
        active, died = info["active"], info["died"]
        want = np.where(active, 0.01 * (info["mass"] - before) - 1.0 * died + 0.001 * (active & ~died), 0.0)
        np.testing.assert_allclose(rewards, want, atol=1e-6)
        assert (dones == died).all() and not (dones & ~active).any()
        assert not obs[~info["alive"]].any()
        seen_death |= bool(died.any())
    assert seen_death


def test_kill_credit_matches_killer_slots():
    env = SlitherEnv(32, max_snakes=8, min_snakes=8, config=CROWDED, seed=2)
    env.reset()
    rng = np.random.default_rng(2)
    total = 0
    for _ in range(80):
        _, _, _, info = env.step(rng.integers(0, 18, (32, 8)))
        killer = info["killer"]
        expect = np.zeros_like(info["kills"])
        e, k = np.nonzero(killer >= 0)
        np.add.at(expect, (e, killer[e, k]), 1)
        assert (info["kills"] == expect).all()
        total += int(expect.sum())
    assert total > 0


def test_same_seed_same_game():
    def run(seed):
        env = SlitherEnv(4, seed=seed)
        obs, _ = env.reset()
        for t in range(20):
            obs, _, _, _ = env.step(np.full((4, 12), t % 18))
        return obs
    np.testing.assert_array_equal(run(5), run(5))


def test_env_rejects_bad_actions():
    env = SlitherEnv(2, max_snakes=4, min_snakes=4, seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros((2, 3), int))
    with pytest.raises(ValueError):
        env.step(np.full((2, 4), 18))
    with pytest.raises(ValueError):
        env.step(np.full((2, 4), -1))


def test_env_rejects_respawn_faster_than_a_decision():
    with pytest.raises(ValueError):
        SlitherEnv(2, config=replace(SlitherConfig(), respawn_delay=3))
    with pytest.raises(ValueError):
        SlitherEnv(2, max_snakes=4, min_snakes=5)
