"""Evaluate a trained checkpoint against every scripted bot and save the results.

  uv run --group train python scripts/final_eval.py runs/selfplay/best.pt --games 1000
  uv run --group train python scripts/final_eval.py runs/arena/best.pt --board 19 \
      --num-snakes 6 --min-food 6 --games 300 --out results/ppo_arena.json

With more than two snakes the network takes seat 0 and every other seat is the same
bot, so an even contest would win 1 / num_snakes of the games.
"""

import argparse
import json
from pathlib import Path

from snake_rl.agents import NeuralPolicy, load_policy
from snake_rl.baselines import make_bot
from snake_rl.eval import run_matches


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default="results/ppo_selfplay.json")
    p.add_argument("--board", type=int, default=11)
    p.add_argument("--num-snakes", type=int, default=2)
    p.add_argument("--min-food", type=int, default=1)
    args = p.parse_args(argv)

    policy = NeuralPolicy(load_policy(args.checkpoint), name="ppo")
    summary = {"games": args.games, "seed": args.seed, "board": args.board,
               "num_snakes": args.num_snakes, "min_food": args.min_food}
    matches = {}
    for bot in ("flood_fill", "greedy", "safe_random"):
        bots = [make_bot(bot, seed=args.seed + 1 + k) for k in range(args.num_snakes - 1)]
        result = run_matches([policy, *bots], args.games, args.seed, board_size=args.board,
                             min_food=args.min_food)
        print(result.format_table(), "\n")
        seat = result.seats[0]
        summary[f"win_vs_{bot}"] = seat.win_rate
        summary[f"draw_vs_{bot}"] = seat.draws / args.games
        summary[f"ci_vs_{bot}"] = seat.win_rate_ci
        matches[bot] = result.to_dict()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "matches": matches}, indent=2) + "\n")
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out} and {summary_path}")


if __name__ == "__main__":
    main()
