"""Evaluate a policy in slot 0 against copies of a scripted bot (spec §4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from snake_rl.eval.harness import bootstrap_mean_ci
from snake_rl.slither.bots import SlitherBot
from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.env import SlitherEnv


@dataclass
class SlitherEvalResult:
    policy: str
    bot: str
    arenas: int
    decisions: int
    mean_mass: float                     # slot 0's mass averaged over time (0 while dead)
    mean_mass_ci: tuple[float, float]
    mass_share: float                    # slot 0's share of all live snake mass; fair = 1/snakes
    mass_share_ci: tuple[float, float]
    kills: int
    deaths: int
    kills_per_death: float
    kills_per_death_ci: tuple[float, float]
    mean_life: float                     # decisions per completed life

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio_ci(num, den, rng, n_resamples=2000):
    """95% bootstrap interval (over arenas) of sum(num) / sum(den)."""
    idx = rng.integers(0, num.size, (n_resamples, num.size))
    ratios = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_vs_bot(policy, bot: str, arenas: int = 64, decisions: int = 1000, warmup: int = 100,
                    seed: int = 0, cfg: SlitherConfig | None = None, snakes: int = 8,
                    name: str = "policy") -> SlitherEvalResult:
    """Slot 0 plays `policy` (callable(obs, mask) -> actions over leading dims); the other
    slots play the bot named `bot`. Metrics are measured over `decisions` after `warmup`."""
    env = SlitherEnv(arenas, max_snakes=snakes, min_snakes=snakes, config=cfg, seed=seed)
    bots = SlitherBot(bot, seed=seed + 1, cfg=env.cfg)
    obs, info = env.reset()
    mass_sum, total_sum = np.zeros(arenas), np.zeros(arenas)
    kills, deaths = np.zeros(arenas, np.int64), np.zeros(arenas, np.int64)
    life = np.zeros(arenas, np.int64)
    life_sum = life_count = 0
    for t in range(warmup + decisions):
        mask = info["action_mask"]
        actions = bots(obs, mask)
        actions[:, 0] = policy(obs[:, 0], mask[:, 0])
        obs, _, dones, info = env.step(actions)
        if t < warmup:
            continue
        mass = np.where(info["alive"], info["mass"], 0.0)
        mass_sum += mass[:, 0]
        total_sum += mass.sum(axis=1)
        kills += info["kills"][:, 0]
        deaths += dones[:, 0]
        life += info["active"][:, 0]
        ended = dones[:, 0]
        life_sum += int(life[ended].sum())
        life_count += int(ended.sum())
        life[ended] = 0
    rng = np.random.default_rng(seed)
    per_arena_mass = mass_sum / decisions
    share = mass_sum / np.maximum(total_sum, 1e-9)
    return SlitherEvalResult(
        policy=name, bot=bot, arenas=arenas, decisions=decisions,
        mean_mass=float(per_arena_mass.mean()), mean_mass_ci=bootstrap_mean_ci(per_arena_mass, rng),
        mass_share=float(share.mean()), mass_share_ci=bootstrap_mean_ci(share, rng),
        kills=int(kills.sum()), deaths=int(deaths.sum()),
        kills_per_death=float(kills.sum() / max(int(deaths.sum()), 1)),
        kills_per_death_ci=_ratio_ci(kills, deaths, rng),
        mean_life=float(life_sum / max(life_count, 1)),
    )
