# Snake RL

Multi-agent Snake as a reinforcement-learning testbed. The end goal is agents trained by
**self-play** against past versions of themselves, ranked by Elo and playable in the
browser. This repository is being built in stages:

1. **Foundation** (done): a fast, tested, batched Battlesnake environment, scripted
   baseline bots and an evaluation harness.
2. Algorithm benchmark: DQN through Rainbow components, and PPO, compared across seeds.
3. Self-play league with Elo tracking.
4. Browser demo with the trained agents.
5. Write-up of the results.

## Quick start

```bash
uv run pytest                                          # full test suite
uv run python scripts/play.py --opponent flood_fill    # play against a bot (WASD)
uv run python scripts/evaluate.py flood_fill greedy    # bot vs bot, 500 games
uv run python scripts/benchmark_env.py                 # environment throughput
```

## Environment

`SnakeEnv` steps many games at once, with every game's state held in NumPy arrays:

```python
from snake_rl.env import SnakeEnv

env = SnakeEnv(num_envs=1024, num_snakes=2)       # standard 11x11 Battlesnake
obs, info = env.reset()                           # obs: (1024, 2, 8, 11, 11)
obs, rewards, dones, info = env.step(actions)     # actions: (1024, 2) in {up, down, left, right}
```

- **Rules:** standard Battlesnake: simultaneous moves, health 100 that ticks down each turn,
  growth on eating, head-to-head collisions won by the longer snake, food spawning.
- **Observations:** egocentric 8-channel grids. These show own head, body and body age,
  enemy heads and bodies, dangerous enemy heads, food, and health. No hand-made features.
- **Speed:** about 214,000 env steps/sec on an Apple Silicon CPU (1024 games, 2 snakes).
- **Correctness:** each rule has a unit test; the batched rules are checked step-by-step
  against a slow reference implementation, and batched results match one-game-at-a-time
  results exactly.

## Baselines (500 games each, 11x11)

| Matchup | Win rate of first bot |
|---|---|
| flood_fill vs greedy | 74.0% |
| greedy vs safe_random | 80.2% |
| safe_random vs random | 99.6% |

The Hamiltonian-cycle bot fills an 8x8 board every game, a reference point for perfect
single-snake play. Raw results, with 95% confidence intervals, are in `results/`.

## Layout

```
snake_rl/env/        rules, observations, rewards, SnakeEnv, renderer
snake_rl/baselines/  random, safe_random, greedy, flood_fill, hamiltonian bots
snake_rl/eval/       match runner with confidence intervals
scripts/             play, evaluate, benchmark
tests/               pytest suite (run in CI)
```
