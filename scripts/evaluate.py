"""Play matches between baseline bots and report win rates or scores.

Examples:
  uv run python scripts/evaluate.py flood_fill greedy --games 500
  uv run python scripts/evaluate.py hamiltonian --board 8 --max-health 64 --games 20
"""

import argparse

from snake_rl.baselines import BOT_NAMES, make_bot
from snake_rl.eval import run_matches


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate baseline bots against each other.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("seats", nargs="+", choices=list(BOT_NAMES) + ["hamiltonian"],
                        help="one bot per snake; give one name for single-snake mode")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--board", type=int, default=11)
    parser.add_argument("--max-health", type=int, default=100)
    parser.add_argument("--out", help="write the result as JSON to this path")
    args = parser.parse_args(argv)

    policies = [make_bot(name, seed=args.seed + i + 1, board_size=args.board)
                for i, name in enumerate(args.seats)]
    result = run_matches(policies, args.games, args.seed, board_size=args.board,
                         max_health=args.max_health)
    print(result.format_table())
    if args.out:
        result.to_json(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
