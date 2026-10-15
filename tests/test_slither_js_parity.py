"""web/slither/rules.js and observe.js must reproduce the Python engine (spec §5): positions
within 1e-9, discrete events exactly, observations within 1e-6. Fixtures use fixed seeds,
so the test is deterministic; a failure is investigated, not retried."""

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from slither_helpers import SMALL, random_ticks
from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.observations import build_observations
from snake_rl.slither.rules import snapshot, step_tick

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def make_cases(cfg, num_envs, slots, ticks, seed, every):
    cases = []
    for t, (state, turn, boost, u) in enumerate(random_ticks(cfg, num_envs, slots, ticks, seed)):
        record = t % every == 0
        if record:
            before = [snapshot(state, e) for e in range(num_envs)]
            obs = build_observations(state)
        ev = step_tick(state, turn, boost, u)
        if record:
            for e in range(num_envs):
                cases.append({
                    "before": before[e], "after": snapshot(state, e),
                    "turns": turn[e].tolist(), "boosts": boost[e].tolist(), "u": u[e].tolist(),
                    "obs": obs[e].astype(float).tolist(),
                    "events": {"died": ev.died[e].tolist(), "killer": ev.killer[e].tolist(),
                               "kills": ev.kills[e].tolist(), "eaten": ev.eaten[e].tolist()},
                })
    return cases


def test_js_matches_python(tmp_path):
    default = SlitherConfig()
    boosty = dataclasses.replace(SMALL, start_mass=40.0)
    fixtures = {
        "defaults": default.to_dict(),
        "groups": [
            {"config": default.to_dict(), "cases": make_cases(default, 3, 12, 240, 0, 8)},
            {"config": SMALL.to_dict(), "cases": make_cases(SMALL, 4, 8, 300, 1, 6)},
            {"config": boosty.to_dict(), "cases": make_cases(boosty, 4, 8, 240, 2, 6)},
        ],
    }
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(fixtures))
    result = subprocess.run(["node", str(ROOT / "web" / "slither" / "test" / "parity.mjs"), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert " 0 failures" in result.stdout
    assert result.stdout.startswith("450 cases")
