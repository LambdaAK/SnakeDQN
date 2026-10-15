import math

import numpy as np
import pytest

import slither_reference as ref
from slither_helpers import QUIET, SMALL, random_ticks, scene, snake, state_of, straight
from snake_rl.slither.actions import NUM_ACTIONS, action_mask, decode_actions, steer_bin
from snake_rl.slither.observations import build_observations, obs_size
from snake_rl.slither.rules import snapshot

C = 8
V = QUIET.vision
W = 2.0 * math.pi / QUIET.sectors
R0 = 2.0 + 0.5 * math.sqrt(10.0)


def obs_of(snap):
    return build_observations(state_of(snap))[0]


def at(q, d):
    """A point at distance d along the centre of sector q, for an observer at the origin facing +x."""
    a = -math.pi + q * W
    return d * math.cos(a), d * math.sin(a)


def me():
    return snake(straight(0.0, 0.0, 0.0, 40))


def test_shape_dtype_and_dead_rows_are_zero():
    o = build_observations(state_of(scene([me(), snake([], alive=False)])))
    assert o.shape == (1, 2, obs_size(QUIET)) == (1, 2, 263) and o.dtype == np.float32
    assert not o[0, 1].any()


def test_wall_ahead():
    o = obs_of(scene([snake(straight(280.0, 0.0, 0.0, 40))]))[0]
    assert o[16 * C + 0] == pytest.approx(1 - 20.0 / V, abs=1e-6)  # sector 16 is straight ahead
    assert o[15 * C + 0] == pytest.approx(o[17 * C + 0], abs=1e-6)
    assert o[0 * C + 0] == 0.0
    assert not obs_of(scene([me()]))[0][0:256:C].any()  # far from every wall


def test_enemy_body_size_and_head():
    enemy = snake(straight(20.0, 25.0, math.pi / 2, 40), heading=math.pi / 2)
    o = obs_of(scene([me(), enemy]))[0]
    assert o[16 * C + 1] == pytest.approx(1 - (20.0 - R0) / V, abs=1e-6)
    assert o[16 * C + 2] == pytest.approx(R0 / 15, abs=1e-6)
    assert o[21 * C + 5] == pytest.approx(1 - (math.hypot(20, 25) - R0) / V, abs=1e-6)
    assert o[21 * C + 6] == pytest.approx(0.0, abs=1e-6)
    assert o[21 * C + 7] == pytest.approx(1.0, abs=1e-6)


def test_own_body_skips_the_neck():
    o = obs_of(scene([me()]))[0]
    assert o[0 * C + 3] == pytest.approx(1 - (10.0 - R0) / V, abs=1e-6)
    assert not o[1:256:C].any()  # no enemies


def test_food_is_summed_per_sector():
    pellets = [(*at(24, 10.0), 1.0), (*at(24, 30.0), 2.0), (*at(8, 10.0), 1.0), (*at(8, 70.0), 9.0)]
    o = obs_of(scene([me()], food=pellets))[0]
    assert o[24 * C + 4] == pytest.approx(0.3, abs=1e-6)
    assert o[8 * C + 4] == pytest.approx(0.1, abs=1e-6)  # the far pellet is out of sight


def test_scalars():
    o = obs_of(scene([me(), snake(straight(100.0, 0.0, math.pi / 2, 40), heading=math.pi / 2, mass=16.0)]))
    np.testing.assert_allclose(o[0, 256:], [math.log(10) / math.log(500), R0 / 15, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(o[1, 256:], [math.log(16) / math.log(500), 4.0 / 15, 0, 1, 1 / 3, 0, 1], atol=1e-6)


def _scene_rotated(alpha=0.0, extra_turns=0):
    def rot(p):
        return [p[0] * math.cos(alpha) - p[1] * math.sin(alpha), p[0] * math.sin(alpha) + p[1] * math.cos(alpha)]
    turn = 2 * math.pi * extra_turns
    a = snake([rot(p) for p in straight(10.0, -20.0, 0.4, 40)], heading=0.4 + alpha + turn)
    b = snake([rot(p) for p in straight(35.0, 5.0, 2.5, 60)], heading=2.5 + alpha + turn, mass=20.0)
    food = [(*rot(p), v) for p, v in [((30.0, -10.0), 1.0), ((0.0, 0.0), 2.0), ((-15.0, -40.0), 1.0)]]
    return scene([a, b], food=food)


def test_observation_invariant_to_rotation():
    np.testing.assert_allclose(obs_of(_scene_rotated(0.7)), obs_of(_scene_rotated(0.0)), atol=1e-6)


def test_observation_invariant_to_full_turns_of_heading():
    np.testing.assert_allclose(obs_of(_scene_rotated(0.0, extra_turns=200)), obs_of(_scene_rotated(0.0)),
                               atol=1e-6)


def test_matches_reference():
    from snake_rl.slither.rules import step_tick
    for t, (state, turn, boost, u) in enumerate(random_ticks(SMALL, 3, 8, 120, 5)):
        if t % 10 == 0:
            obs = build_observations(state)
            for e in range(3):
                snap = snapshot(state, e)
                for k in range(8):
                    np.testing.assert_allclose(obs[e, k], ref.observe(snap, k, SMALL), atol=1e-6,
                                               err_msg=f"tick {t} arena {e} snake {k}")
        step_tick(state, turn, boost, u)


def test_actions_decode_and_mask():
    turn, boost = decode_actions(np.array([0, 1, 8, 9, 17]), QUIET)
    np.testing.assert_allclose(turn, [-0.15, -0.15, 0.0, 0.0, 0.15])
    assert boost.tolist() == [False, True, False, True, True]
    state = state_of(scene([snake(straight(0, 0, 0, 40), mass=14.0), snake(straight(0, 50, 0, 40), mass=15.0)]))
    mask = action_mask(state)
    assert mask.shape == (1, 2, NUM_ACTIONS)
    assert mask[0, 0, ::2].all() and not mask[0, 0, 1::2].any() and mask[0, 1].all()


def test_steer_bin():
    assert steer_bin(np.array([0.0, 0.45, -10.0, 0.1125]), QUIET).tolist() == [4, 8, 0, 5]
