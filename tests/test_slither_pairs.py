import numpy as np

from snake_rl.slither import rules


def test_grid_pair_search_finds_exactly_the_dense_pairs():
    rng = np.random.default_rng(0)
    n, s, p = 5, 7, 4000
    pe = rng.integers(0, n, p)
    px, py = rng.uniform(-300, 300, p), rng.uniform(-300, 300, p)
    heads = rng.uniform(-300, 300, (n, s, 2))
    live = rng.random((n, s)) < 0.8
    for reach in (5.0, 27.0, 75.0):
        want = list(zip(*rules._near_pairs_dense(pe, px, py, heads, live, reach)))
        got = list(zip(*rules._near_pairs_grid(pe, px, py, heads, live, reach)))
        assert got == want  # same pairs in the same (point, slot) order
    assert rules.near_pairs is rules._near_pairs_grid
