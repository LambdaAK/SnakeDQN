"""Play matches between policies and summarize results with confidence intervals."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from snake_rl.env import SnakeEnv
from snake_rl.env.rules import START_LENGTH, GameState

Policy = Callable[[GameState, int], np.ndarray]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values, rng: np.random.Generator,
                      n_resamples: int = 2000) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean."""
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(n_resamples, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


@dataclass
class SeatResult:
    name: str
    wins: int
    draws: int
    losses: int
    win_rate: float | None  # None for single-snake runs, which have no opponent
    win_rate_ci: tuple[float, float] | None
    mean_length: float


@dataclass
class ScoreResult:
    mean: float
    median: float
    max: int
    ci: tuple[float, float]


@dataclass
class MatchResult:
    n_games: int
    board_size: int
    seed: int
    mean_turns: float
    seats: list[SeatResult]
    score: ScoreResult | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def format_table(self) -> str:
        lines = [f"{self.n_games} games on {self.board_size}x{self.board_size}, "
                 f"seed {self.seed}, mean game length {self.mean_turns:.1f} turns"]
        if self.score is not None:
            lo, hi = self.score.ci
            lines.append(f"{self.seats[0].name}: score mean {self.score.mean:.2f} "
                         f"(95% CI {lo:.2f}-{hi:.2f}), median {self.score.median:.1f}, "
                         f"max {self.score.max}")
            return "\n".join(lines)
        lines.append(f"{'seat':<14}{'wins':>6}{'draws':>7}{'losses':>8}"
                     f"{'win rate':>10}{'95% CI':>16}{'mean len':>10}")
        for seat in self.seats:
            lo, hi = seat.win_rate_ci
            ci = f"{lo:.3f}-{hi:.3f}"
            lines.append(f"{seat.name:<14}{seat.wins:>6}{seat.draws:>7}{seat.losses:>8}"
                         f"{seat.win_rate:>10.3f}{ci:>16}{seat.mean_length:>10.1f}")
        return "\n".join(lines)


def run_matches(policies: Sequence[Policy], n_games: int, seed: int,
                board_size: int = 11, max_health: int = 100, min_food: int = 1,
                names: Sequence[str] | None = None) -> MatchResult:
    """Play n_games games, one per env, with policies[i] controlling snake i.

    `seed` fixes the games; reproducibility also requires the caller to seed the policies."""
    if not policies:
        raise ValueError("need at least one policy")
    if n_games < 1:
        raise ValueError(f"n_games must be >= 1, got {n_games}")
    s = len(policies)
    if names is not None and len(names) != len(policies):
        raise ValueError(f"got {len(names)} names for {len(policies)} policies")
    names = list(names) if names else [getattr(p, "name", f"seat{i}") for i, p in enumerate(policies)]
    env = SnakeEnv(n_games, s, board_size, max_health=max_health, seed=seed, min_food=min_food)
    env.reset()

    recorded = np.zeros(n_games, bool)
    winner = np.full(n_games, -1)
    drew = np.zeros((n_games, s), bool)
    final_length = np.zeros((n_games, s))
    turns = np.zeros(n_games)
    while not recorded.all():
        actions = np.stack([policy(env.state, i) for i, policy in enumerate(policies)], axis=1)
        _, _, _, info = env.step(actions)
        new = info["game_over"] & ~recorded
        if new.any():
            winner[new] = info["winner"][new]
            drew[new] = info["active"][new] & (info["winner"][new] < 0)[:, None]
            final_length[new] = info["final_length"][new]
            turns[new] = info["final_turn"][new]
            recorded |= new

    seats = []
    for i, name in enumerate(names):
        if s == 1:
            wins = draws = losses = 0
        else:
            wins = int((winner == i).sum())
            draws = int(drew[:, i].sum())
            losses = n_games - wins - draws
        rate = None if s == 1 else wins / n_games
        ci = None if s == 1 else wilson_interval(wins, n_games)
        seats.append(SeatResult(name, wins, draws, losses, rate, ci,
                                float(final_length[:, i].mean())))

    score = None
    if s == 1:
        scores = final_length[:, 0] - START_LENGTH
        score = ScoreResult(float(scores.mean()), float(np.median(scores)), int(scores.max()),
                            bootstrap_mean_ci(scores, np.random.default_rng(seed)))
    return MatchResult(n_games, board_size, seed, float(turns.mean()), seats, score)
