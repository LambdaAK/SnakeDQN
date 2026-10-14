"""SnakeEnv: many Battlesnake games stepped together, with auto-reset."""

from __future__ import annotations

import numpy as np

from snake_rl.env.observations import build_observations
from snake_rl.env.rewards import get_reward
from snake_rl.env.rules import GameState, action_mask, reset_games, step_games


class SnakeEnv:
    def __init__(self, num_envs: int, num_snakes: int, board_size: int = 11,
                 reward: str | None = None, max_health: int = 100,
                 seed: int | None = None):
        self.state = GameState.empty(num_envs, num_snakes, board_size, max_health)
        self.reward_fn = get_reward(reward or ("winloss" if num_snakes > 1 else "food"))
        self.rng = np.random.default_rng(seed)
        self._needs_reset = True

    @property
    def num_envs(self) -> int:
        return self.state.num_envs

    @property
    def num_snakes(self) -> int:
        return self.state.num_snakes

    @property
    def board_size(self) -> int:
        return self.state.board_size

    def reset(self):
        reset_games(self.state, np.ones(self.num_envs, bool), self.rng)
        self._needs_reset = False
        return build_observations(self.state), self._info()

    def step(self, actions):
        """Advance every game one turn. Games that end are reset immediately; `info`
        keys prefixed `final_` (and `game_over`, `winner`) describe them before the reset.
        When no game ended, `obs` and `info["final_obs"]` are the same array."""
        if self._needs_reset:
            raise RuntimeError("call reset() before step()")
        spawn_u = self.rng.random((self.num_envs, 2))
        events = step_games(self.state, actions, spawn_u)
        rewards = self.reward_fn(events)
        dones = events.died | (events.game_over[:, None] & events.alive_before)
        obs = build_observations(self.state)
        info = {
            "game_over": events.game_over,
            "winner": events.winner,
            "final_length": self.state.length.copy(),
            "final_turn": self.state.turn.copy(),
            "final_obs": obs,
        }
        ended = np.flatnonzero(events.game_over)
        if ended.size:
            reset_games(self.state, events.game_over, self.rng)
            obs = obs.copy()
            obs[ended] = build_observations(self.state, ended)
        info.update(self._info())
        return obs, rewards, dones, info

    def _info(self) -> dict:
        return {
            "action_mask": action_mask(self.state),
            "length": self.state.length.copy(),
            "health": self.state.health.copy(),
            "alive": self.state.alive.copy(),
            "turn": self.state.turn.copy(),
        }
