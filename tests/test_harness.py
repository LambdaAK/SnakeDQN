import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from snake_rl.baselines import GreedyBot, RandomBot
from snake_rl.env import SnakeEnv
from snake_rl.eval import bootstrap_mean_ci, run_matches, wilson_interval

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_bootstrap_ci_brackets_mean():
    values = np.arange(100)
    lo, hi = bootstrap_mean_ci(values, np.random.default_rng(0))
    assert lo < values.mean() < hi


def test_two_player_counts_add_up_and_are_reproducible():
    def run():
        return run_matches([RandomBot(1), RandomBot(2)], n_games=50, seed=0)

    result = run()
    for seat in result.seats:
        assert seat.wins + seat.draws + seat.losses == 50
    assert result.seats[0].wins + result.seats[1].wins + result.seats[0].draws <= 50
    assert result.seats[0].draws == result.seats[1].draws
    assert result.score is None
    assert run().to_dict() == result.to_dict()


def test_single_snake_reports_score():
    result = run_matches([GreedyBot(0)], n_games=20, seed=0)
    score = result.score
    assert score is not None
    assert score.max >= score.median >= 0
    assert score.ci[0] <= score.mean <= score.ci[1]


def test_json_and_table(tmp_path):
    result = run_matches([GreedyBot(0), RandomBot(0)], n_games=10, seed=0)
    path = tmp_path / "result.json"
    result.to_json(path)
    data = json.loads(path.read_text())
    assert data["n_games"] == 10 and len(data["seats"]) == 2
    table = result.format_table()
    assert "greedy" in table and "random" in table


def test_evaluate_cli(tmp_path):
    out = tmp_path / "cli.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "greedy", "random",
         "--games", "10", "--seed", "0", "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(out.read_text())["seats"][0]["name"] == "greedy"


def test_evaluate_cli_rejects_unknown_bot():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "nope", "random"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "invalid choice" in result.stderr


class Up:
    """Always moves up (direction 0: row - 1).

    Outcomes are determined by the start rows: the game ends at the first wall death,
    and snakes moving straight up cannot reach each other's bodies before that.
    """

    name = "up"

    def __call__(self, state, snake):
        return np.zeros(state.num_envs, dtype=np.int64)


def test_draw_rule_and_exactly_once_counting():
    n, seed = 40, 0
    # Same construction and reset as run_matches, so the start cells are identical.
    env = SnakeEnv(n, 2, seed=seed)
    env.reset()
    rows = env.state.body[:, :, 0, 0].astype(int)  # (N, 2) head rows
    # A head at row r moves up and leaves the board on turn r + 1.
    death = rows + 1
    first_out = death.min(axis=1)
    equal = rows[:, 0] == rows[:, 1]
    assert equal.any() and (~equal).any(), "test data must contain draws and decisive games"

    result = run_matches([Up(), Up()], n_games=n, seed=seed)
    s0, s1 = result.seats

    exp_draws = int(equal.sum())
    exp_wins0 = int((rows[:, 0] > rows[:, 1]).sum())  # seat 1 dies first -> seat 0 wins
    exp_wins1 = int((rows[:, 1] > rows[:, 0]).sum())
    assert (s0.wins, s0.draws, s0.losses) == (exp_wins0, exp_draws, n - exp_wins0 - exp_draws)
    assert (s1.wins, s1.draws, s1.losses) == (exp_wins1, exp_draws, n - exp_wins1 - exp_draws)
    # Each game is counted exactly once: the game length is the first death turn only.
    assert np.isclose(result.mean_turns, first_out.mean())
