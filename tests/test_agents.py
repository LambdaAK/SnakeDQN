import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from snake_rl.agents import NeuralPolicy, PolicyNet, load_policy, masked_logits  # noqa: E402
from snake_rl.env import SnakeEnv  # noqa: E402
from snake_rl.env.rules import GameState, action_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_policy_net_shapes():
    logits, value = PolicyNet()(torch.zeros(5, 8, 11, 11))
    assert logits.shape == (5, 4) and value.shape == (5,)


def test_masked_logits_forbids_unsafe_moves_but_keeps_all_when_none_safe():
    logits = torch.zeros(2, 4)
    mask = torch.tensor([[True, False, False, True], [False, False, False, False]])
    out = masked_logits(logits, mask)
    assert (out[0, [1, 2]] < -1e8).all() and (out[0, [0, 3]] == 0).all()
    assert (out[1] == 0).all()


def test_neural_policy_only_picks_safe_moves():
    env = SnakeEnv(32, 2, seed=0)
    env.reset()
    policy = NeuralPolicy(PolicyNet())
    for _ in range(20):
        mask = action_mask(env.state)[:, 0]
        actions = policy(env.state, 0)
        has_safe = mask.any(axis=1)
        assert mask[np.arange(32), actions][has_safe].all()
        env.step(np.stack([actions, policy(env.state, 1)], axis=1))


def test_training_smoke_run_writes_loadable_checkpoint(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_selfplay.py"), "--run-dir", str(tmp_path),
         "--num-envs", "8", "--rollout", "8", "--iterations", "2", "--eval-every", "1",
         "--eval-games", "4", "--minibatch", "32", "--threads", "1"],
        check=True, capture_output=True, text=True,
    )
    assert (tmp_path / "metrics.csv").read_text().count("\n") == 3
    model = load_policy(tmp_path / "best.pt")
    state = GameState.from_layout(11, [[(5, 5), (6, 5), (7, 5)], [(1, 1), (1, 2), (1, 3)]])
    assert NeuralPolicy(model)(state, 0).shape == (1,)
