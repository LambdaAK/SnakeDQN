"""The browser demo's rules (web/rules.js) must match the Python rules exactly."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from snake_rl.env.observations import build_observations
from snake_rl.env.rules import GameState, action_mask, reset_games, step_games

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def snapshot(state, env):
    snakes = []
    for k in range(state.num_snakes):
        length = int(state.length[env, k])
        snakes.append({
            "body": [[int(r), int(c)] for r, c in state.body[env, k, :length]],
            "health": int(state.health[env, k]),
            "alive": bool(state.alive[env, k]),
        })
    return {
        "board": state.board_size, "maxHealth": state.max_health,
        "snakes": snakes, "food": [int(i) for i in np.flatnonzero(state.food[env])],
        "turn": int(state.turn[env]), "over": bool(state.over[env]),
    }


def make_cases(board_size, num_snakes, steps, seed):
    rng = np.random.default_rng(seed)
    state = GameState.empty(16, num_snakes, board_size, max_health=30)
    cases = []
    for _ in range(steps):
        reset_games(state, state.over, rng)
        obs = build_observations(state)
        mask = action_mask(state)
        before = [snapshot(state, e) for e in range(state.num_envs)]
        safe = np.argmax(rng.random(mask.shape) + mask, axis=-1)
        actions = np.where(rng.random(safe.shape) < 0.9, safe, rng.integers(0, 4, safe.shape))
        spawn_u = rng.random((state.num_envs, 2))
        step_games(state, actions, spawn_u)
        for e in range(state.num_envs):
            cases.append({
                "before": before[e], "after": snapshot(state, e),
                "actions": [int(a) for a in actions[e]],
                "u0": float(spawn_u[e, 0]), "u1": float(spawn_u[e, 1]),
                "mask": mask[e].tolist(),
                "obs": [np.round(obs[e, k].ravel(), 6).tolist() for k in range(num_snakes)],
            })
    return cases


def test_js_rules_match_python(tmp_path):
    cases = (make_cases(11, 2, 120, 0) + make_cases(7, 1, 80, 1)
             + make_cases(11, 4, 80, 2) + make_cases(8, 3, 60, 3))
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(cases))
    result = subprocess.run(["node", str(ROOT / "web" / "test" / "parity.mjs"), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
