"""The batched engine must match the slow reference tick for tick (positions within 1e-9,
discrete events exactly), and batching arenas must not change any result."""

import copy

import numpy as np
import pytest

import slither_reference as ref
from slither_helpers import SMALL, random_ticks
from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.rules import from_snapshots, snapshot, step_tick


def assert_same(got, want, where):
    assert got["tick"] == want["tick"], where
    for k, (a, b) in enumerate(zip(got["snakes"], want["snakes"], strict=True)):
        for key in ("alive", "enabled", "boosting", "boost_ticks", "respawn"):
            assert a[key] == b[key], f"{where} snake {k} {key}: {a[key]} != {b[key]}"
        for key in ("heading", "mass"):
            assert abs(a[key] - b[key]) <= 1e-9, f"{where} snake {k} {key}: {a[key]} != {b[key]}"
        assert len(a["body"]) == len(b["body"]), f"{where} snake {k} body length"
        if a["body"]:
            np.testing.assert_allclose(a["body"], b["body"], rtol=0, atol=1e-9,
                                       err_msg=f"{where} snake {k} body")
    for f, (a, b) in enumerate(zip(got["food"], want["food"], strict=True)):
        assert (a is None) == (b is None), f"{where} food slot {f}"
        if a is not None:
            np.testing.assert_allclose(a[:3], b[:3], rtol=0, atol=1e-9, err_msg=f"{where} food {f}")
            assert a[3] == b[3], f"{where} food {f} natural flag"


@pytest.mark.parametrize("cfg,num_envs,slots,ticks,seed,crowded", [
    (SMALL, 4, 8, 400, 0, True),
    (SlitherConfig(), 2, 12, 300, 1, False),
])
def test_batched_engine_matches_reference(cfg, num_envs, slots, ticks, seed, crowded):
    died = kills = respawns = 0
    eaten = 0.0
    for t, (state, turn, boost, u) in enumerate(random_ticks(cfg, num_envs, slots, ticks, seed)):
        before = [snapshot(state, e) for e in range(num_envs)]
        alive_before = state.alive.copy()
        ev = step_tick(state, turn, boost, u)
        for e in range(num_envs):
            want = copy.deepcopy(before[e])
            rev = ref.step_tick(want, turn[e].tolist(), boost[e].tolist(), u[e].tolist(), cfg)
            where = f"tick {t} arena {e}"
            assert_same(snapshot(state, e), want, where)
            assert ev.died[e].tolist() == rev["died"], where
            assert ev.killer[e].tolist() == rev["killer"], where
            assert ev.kills[e].tolist() == rev["kills"], where
            np.testing.assert_allclose(ev.eaten[e], rev["eaten"], rtol=0, atol=1e-9, err_msg=where)
        died += int(ev.died.sum())
        kills += int(ev.kills.sum())
        eaten += float(ev.eaten.sum())
        respawns += int((~alive_before & state.alive).sum())
    assert eaten > 0
    if crowded:  # make sure the run exercised every rule, not just movement
        assert died > 10 and kills > 0 and respawns > 0


def test_batching_does_not_change_results():
    singles = None
    for state, turn, boost, u in random_ticks(SMALL, 4, 8, 150, 2):
        if singles is None:
            singles = [from_snapshots([snapshot(state, e)], SMALL) for e in range(4)]
        step_tick(state, turn, boost, u)
        for e, single in enumerate(singles):
            step_tick(single, turn[e:e + 1], boost[e:e + 1], u[e:e + 1])
            assert snapshot(single, 0) == snapshot(state, e)
