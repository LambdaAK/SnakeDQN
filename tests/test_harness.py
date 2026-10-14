import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from snake_rl.baselines import GreedyBot, RandomBot
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
