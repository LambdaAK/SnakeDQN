"""Measure SlitherEnv throughput in snake decisions per second (spec §5 target: 20,000).

  uv run python scripts/benchmark_slither.py
  uv run python scripts/benchmark_slither.py --num-envs 256 --decisions 100

Every slot is driven by the forager bot, so bodies grow to realistic lengths during the
warm-up; only env.step is timed.
"""

import argparse
import time

from snake_rl.slither.bots import SlitherBot
from snake_rl.slither.env import SlitherEnv


def main(argv=None) -> float:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--max-snakes", type=int, default=12)
    p.add_argument("--min-snakes", type=int, default=6)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--decisions", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    env = SlitherEnv(args.num_envs, args.max_snakes, args.min_snakes, seed=args.seed)
    bot = SlitherBot("forager", seed=args.seed, cfg=env.cfg)
    obs, info = env.reset()
    for _ in range(args.warmup):
        obs, _, _, info = env.step(bot(obs, info["action_mask"]))
    snakes, spent = 0, 0.0
    for _ in range(args.decisions):
        actions = bot(obs, info["action_mask"])
        start = time.perf_counter()
        obs, _, _, info = env.step(actions)
        spent += time.perf_counter() - start
        snakes += int(info["active"].sum())
    alive = info["alive"]
    rate = snakes / spent
    print(f"{args.num_envs} arenas, {alive.sum(axis=1).mean():.1f} live snakes each, "
          f"mean mass {info['mass'][alive].mean():.1f}, mean length {env.state.length[alive].mean():.0f} points")
    print(f"{rate:,.0f} snake decisions/s (env only, {1000 * spent / args.decisions:.1f} ms per step)")
    return rate


if __name__ == "__main__":
    main()
