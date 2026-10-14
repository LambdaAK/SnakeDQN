import pytest

from snake_rl.baselines import make_bot
from snake_rl.eval import run_matches


@pytest.mark.parametrize(
    "stronger,weaker",
    [("safe_random", "random"), ("greedy", "safe_random"), ("flood_fill", "greedy")],
)
def test_baseline_ladder(stronger, weaker):
    result = run_matches(
        [make_bot(stronger, seed=1), make_bot(weaker, seed=2)], n_games=500, seed=0
    )
    assert result.seats[0].win_rate > 0.6, result.format_table()
