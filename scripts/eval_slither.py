"""Final evaluation of a slither policy against each scripted bot (spec §4).

  uv run --group train python scripts/eval_slither.py runs/slither/best.pt --arenas 300 --decisions 3000
  uv run python scripts/eval_slither.py bot:hunter --bots wander forager --arenas 100

The policy plays slot 0 of 8; the other 7 slots are copies of the bot. Writes JSON with
mass share (fair share is 1/8), mean mass, kills per death and mean life, with 95% CIs.
"""

import argparse
import json
from pathlib import Path

from snake_rl.slither.bots import BOT_NAMES, SlitherBot
from snake_rl.slither.eval import evaluate_vs_bot

SNAKES = 8


def make_policy(spec, seed):
    if spec.startswith("bot:"):
        return SlitherBot(spec[4:], seed=seed)
    from snake_rl.slither.model import SlitherPolicy, load_slither_policy  # needs torch
    return SlitherPolicy(load_slither_policy(spec))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy", help="checkpoint path, or bot:<name>")
    p.add_argument("--bots", nargs="+", default=list(BOT_NAMES), choices=BOT_NAMES)
    p.add_argument("--arenas", type=int, default=300)
    p.add_argument("--decisions", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default="results/slither_eval.json")
    args = p.parse_args(argv)

    results = {}
    for i, bot in enumerate(args.bots):
        r = evaluate_vs_bot(make_policy(args.policy, args.seed + 100 + i), bot, arenas=args.arenas,
                            decisions=args.decisions, warmup=args.warmup, seed=args.seed + i,
                            snakes=SNAKES, name=args.policy)
        results[bot] = r.to_dict()
        print(f"vs 7x {bot:8s} mass share {r.mass_share:.3f} [{r.mass_share_ci[0]:.3f}, {r.mass_share_ci[1]:.3f}] "
              f"(fair {1 / SNAKES:.3f})  mass {r.mean_mass:.1f}  kills/death {r.kills_per_death:.2f} "
              f"[{r.kills_per_death_ci[0]:.2f}, {r.kills_per_death_ci[1]:.2f}]  life {r.mean_life:.0f} decisions",
              flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": args.policy, "snakes": SNAKES, "results": results}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
