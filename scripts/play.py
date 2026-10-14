"""Play Snake in the terminal with WASD, alone or against a bot.

  uv run python scripts/play.py
  uv run python scripts/play.py --opponent flood_fill
"""

import argparse
import sys
import termios
import tty

import numpy as np

from snake_rl.baselines import BOT_NAMES, make_bot
from snake_rl.env import START_LENGTH, SnakeEnv, render

KEYS = {"w": 0, "s": 1, "a": 2, "d": 3}


def read_key() -> str:
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:  # not a terminal
        try:
            return (input("move (w/a/s/d, q to quit): ").strip().lower() or "q")[0]
        except EOFError:
            return "q"
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1).lower()
        return "q" if key in ("", "\x03", "\x04") else key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play Snake with WASD.")
    parser.add_argument("--opponent", choices=BOT_NAMES, help="bot to play against")
    parser.add_argument("--board", type=int, default=11)
    parser.add_argument("--max-health", type=int, default=100)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    num_snakes = 2 if args.opponent else 1
    env = SnakeEnv(1, num_snakes, args.board, max_health=args.max_health, seed=args.seed)
    env.reset()
    bot = make_bot(args.opponent, seed=args.seed) if args.opponent else None
    action = 3
    while True:
        print("\033[2J\033[H" + render(env.state), flush=True)
        print("You are snake 0. WASD to move, Q to quit.")
        key = read_key()
        if key == "q":
            return
        action = KEYS.get(key, action)
        row = [action] + ([int(bot(env.state, 1)[0])] if bot else [])
        _, _, _, info = env.step(np.array([row]))
        if info["game_over"][0]:
            if bot is None:
                print(f"Game over. Score: {info['final_length'][0, 0] - START_LENGTH}")
            else:
                winner = info["winner"][0]
                print({0: "You win!", 1: f"{args.opponent} wins."}.get(winner, "Draw."))
            return


if __name__ == "__main__":
    main()
