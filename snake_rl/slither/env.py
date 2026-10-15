"""SlitherEnv: many endless slither arenas stepped together, one decision (3 ticks) per step."""

from __future__ import annotations

import numpy as np

from snake_rl.slither.actions import NUM_ACTIONS, action_mask, decode_actions
from snake_rl.slither.config import SlitherConfig, u_layout
from snake_rl.slither.observations import build_observations, obs_size
from snake_rl.slither.rules import SlitherState, reset_arenas, step_tick


class SlitherEnv:
    def __init__(self, num_envs: int, max_snakes: int = 12, min_snakes: int = 6,
                 config: SlitherConfig | None = None, seed: int | None = None,
                 mass_scale: float = 0.01, death_penalty: float = 1.0, survival_bonus: float = 0.0):
        self.cfg = config or SlitherConfig()
        if not 1 <= min_snakes <= max_snakes:
            raise ValueError(f"need 1 <= min_snakes <= max_snakes, got {min_snakes}, {max_snakes}")
        if self.cfg.respawn_delay <= self.cfg.ticks_per_decision:
            raise ValueError("respawn_delay must exceed ticks_per_decision, or a snake could die "
                             "and respawn within one decision")
        self.state = SlitherState.empty(num_envs, max_snakes, self.cfg)
        self.min_snakes, self.max_snakes = min_snakes, max_snakes
        self.mass_scale, self.death_penalty, self.survival_bonus = mass_scale, death_penalty, survival_bonus
        self.rng = np.random.default_rng(seed)
        self.u_size = u_layout(max_snakes, self.cfg)["size"]
        self._needs_reset = True

    @property
    def num_envs(self) -> int:
        return self.state.num_envs

    @property
    def num_snakes(self) -> int:
        return self.state.num_snakes

    @property
    def obs_size(self) -> int:
        return obs_size(self.cfg)

    def reset(self):
        """Fresh arenas with min_snakes..max_snakes enabled slots each (fixed from now on)."""
        counts = self.rng.integers(self.min_snakes, self.max_snakes + 1, self.num_envs)
        reset_arenas(self.state, np.arange(self.num_snakes) < counts[:, None], self.rng)
        self._needs_reset = False
        zeros = np.zeros((self.num_envs, self.num_snakes))
        info = self._info(self.state.alive.copy(), zeros.astype(bool), np.full(zeros.shape, -1, np.int32),
                          zeros.astype(np.int32))
        return build_observations(self.state), info

    def step(self, actions):
        """One decision. Arenas are endless: dead snakes respawn inside the rules, and
        `info["active"]` marks the slots whose transition is real (alive at the start)."""
        if self._needs_reset:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, self.num_snakes):
            raise ValueError(f"actions must have shape {(self.num_envs, self.num_snakes)}, got {actions.shape}")
        if actions.min() < 0 or actions.max() >= NUM_ACTIONS:
            raise ValueError(f"actions must be in [0, {NUM_ACTIONS})")
        turn, boost = decode_actions(actions, self.cfg)
        active = self.state.alive.copy()
        mass_before = self.state.mass.copy()
        died = np.zeros(active.shape, bool)
        killer = np.full(active.shape, -1, np.int32)
        kills = np.zeros(active.shape, np.int32)
        for _ in range(self.cfg.ticks_per_decision):
            ev = step_tick(self.state, turn, boost, self.rng.random((self.num_envs, self.u_size)))
            # Only slots alive at the start are real transitions. A slot that respawned mid-decision
            # is padding, so its death (and any kill it made as a victim) is not counted.
            victim = ev.died & active
            killer = np.where(victim, ev.killer, killer)
            died |= victim
            ke, ki = np.nonzero(victim & (ev.killer >= 0))
            np.add.at(kills, (ke, ev.killer[ke, ki]), 1)
        gained = np.where(active, self.state.mass - mass_before, 0.0)
        rewards = (self.mass_scale * gained - self.death_penalty * died
                   + self.survival_bonus * (active & ~died)).astype(np.float32)
        return build_observations(self.state), rewards, died.copy(), self._info(active, died, killer, kills)

    def _info(self, active, died, killer, kills) -> dict:
        return {
            "active": active,
            "alive": self.state.alive.copy(),
            "enabled": self.state.enabled.copy(),
            "action_mask": action_mask(self.state),
            "mass": self.state.mass.copy(),
            "died": died,
            "killer": killer,
            "kills": kills,
        }
