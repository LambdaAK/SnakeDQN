import math

import numpy as np
import pytest

from slither_helpers import scene, snake, state_of, straight
from snake_rl.slither.actions import action_mask
from snake_rl.slither.bots import BOT_NAMES, SlitherBot
from snake_rl.slither.env import SlitherEnv
from snake_rl.slither.observations import build_observations


def act(name, snap, slot=0):
    state = state_of(snap)
    return int(SlitherBot(name, seed=0)(build_observations(state), action_mask(state))[0, slot])


def at(q, d):
    a = -math.pi + q * 2 * math.pi / 32
    return d * math.cos(a), d * math.sin(a)


@pytest.mark.parametrize("name", BOT_NAMES)
def test_actions_are_valid_and_respect_the_mask(name):
    env = SlitherEnv(8, seed=0)
    obs, info = env.reset()
    for _ in range(5):
        actions = SlitherBot(name, seed=1)(obs, info["action_mask"])
        assert actions.shape == (8, 12) and actions.dtype == np.int64
        assert actions.min() >= 0 and actions.max() < 18
        assert not ((actions % 2 == 1) & ~info["action_mask"][..., 1]).any()
        obs, _, _, info = env.step(actions)


@pytest.mark.parametrize("name", BOT_NAMES)
def test_turns_hard_away_from_a_wall_ahead(name):
    assert act(name, scene([snake(straight(280.0, 0.0, 0.0, 40))])) // 2 in (0, 8)


def test_forager_turns_toward_food():
    food = [(*at(20, 15.0), 5.0), (*at(20, 18.0), 5.0)]
    action = act("forager", scene([snake(straight(0.0, 0.0, 0.0, 40))], food=food))
    assert action == 8 * 2  # hard turn (sector 20 is centred 45 degrees off the heading), no boost


def test_hunter_boosts_at_a_smaller_head_ahead_and_forager_does_not():
    hx, hy = at(17, 20.0)
    me = snake(straight(0.0, 0.0, 0.0, 100), mass=40.0)
    prey = snake(straight(hx, hy, math.pi / 2, 40), heading=math.pi / 2, mass=10.0)
    assert act("hunter", scene([me, prey])) % 2 == 1
    assert act("forager", scene([me, prey])) % 2 == 0


def test_unknown_bot():
    with pytest.raises(ValueError):
        SlitherBot("nope")
