"""Measure SnakeEnv throughput (one env step = one turn of one game)."""

import argparse
import time

import numpy as np

from snake_rl.env import SnakeEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-snakes", type=int, default=2)
    parser.add_argument("--board", type=int, default=11)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    env = SnakeEnv(args.num_envs, args.num_snakes, args.board, seed=args.seed)
    _, info = env.reset()
    rng = np.random.default_rng(args.seed)

    def safe_random(mask):
        return np.argmax(rng.random(mask.shape) + mask, axis=-1)

    for _ in range(min(20, args.steps)):  # warm-up
        _, _, _, info = env.step(safe_random(info["action_mask"]))
    start = time.perf_counter()
    for _ in range(args.steps):
        _, _, _, info = env.step(safe_random(info["action_mask"]))
    elapsed = time.perf_counter() - start
    rate = args.steps * args.num_envs / elapsed
    print(f"{rate:,.0f} env steps/sec "
          f"(num_envs={args.num_envs}, num_snakes={args.num_snakes}, board={args.board})")


if __name__ == "__main__":
    main()
