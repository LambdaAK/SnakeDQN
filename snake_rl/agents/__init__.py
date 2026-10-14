"""Learned agents (requires the `train` dependency group: `uv sync --group train`)."""

from snake_rl.agents.model import PolicyNet, load_policy, masked_logits
from snake_rl.agents.policy import NeuralPolicy
