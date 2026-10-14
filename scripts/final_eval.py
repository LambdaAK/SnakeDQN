"""Evaluate a trained checkpoint against every scripted bot and save the results.

  uv run --group train python scripts/final_eval.py runs/selfplay/best.pt --games 1000
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
    args = p.parse_args(argv)

    policy = NeuralPolicy(load_policy(args.checkpoint), name="ppo")
    summary = {"games": args.games, "seed": args.seed}
    matches = {}
    for bot in ("flood_fill", "greedy", "safe_random"):
        result = run_matches([policy, make_bot(bot, seed=args.seed + 1)], args.games, args.seed)
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
