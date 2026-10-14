"""Render a self-play game of a trained checkpoint to an animated GIF for the README.

  uv run --group train python scripts/render_gif.py runs/selfplay/best.pt --out docs/selfplay.gif
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from snake_rl.agents import NeuralPolicy, load_policy
from snake_rl.env import SnakeEnv

COLORS = [(245, 158, 11), (56, 189, 248)]
BG = [(14, 20, 38), (16, 23, 43)]
FOOD = (244, 63, 94)


def draw(state, cell=36):
    b = state.board_size
    img = Image.new("RGB", (b * cell, b * cell))
    d = ImageDraw.Draw(img)
    for r in range(b):
        for c in range(b):
            d.rectangle([c * cell, r * cell, (c + 1) * cell, (r + 1) * cell], fill=BG[(r + c) % 2])
    for r, c in zip(*np.nonzero(state.food[0])):
        pad = cell * 0.27
        d.ellipse([c * cell + pad, r * cell + pad, (c + 1) * cell - pad, (r + 1) * cell - pad], fill=FOOD)
    for k in range(state.num_snakes):
        if not state.alive[0, k]:
            continue
        length = int(state.length[0, k])
        segs = [tuple(int(v) for v in p) for p in state.body[0, k, :length]]
        w = cell * 0.66
        for (r1, c1), (r2, c2) in zip(segs[1:], segs[:-1]):
            x0 = min(c1, c2) * cell + (cell - w) / 2
            y0 = min(r1, r2) * cell + (cell - w) / 2
            d.rounded_rectangle([x0, y0, x0 + abs(c1 - c2) * cell + w, y0 + abs(r1 - r2) * cell + w],
                                radius=w * 0.35, fill=COLORS[k])
        r, c = segs[0]
        s = cell * 0.82
        x0, y0 = c * cell + (cell - s) / 2, r * cell + (cell - s) / 2
        d.rounded_rectangle([x0, y0, x0 + s, y0 + s], radius=s * 0.32, fill=COLORS[k])
    return img


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--out", default="docs/selfplay.gif")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--min-turns", type=int, default=80, help="re-roll games shorter than this")
    args = p.parse_args(argv)

    model = load_policy(args.checkpoint)
    for seed in range(args.seed, args.seed + 50):
        policy = NeuralPolicy(model, deterministic=False, seed=seed)
        env = SnakeEnv(1, 2, seed=seed)
        env.reset()
        frames = [draw(env.state)]
        while True:
            actions = np.stack([policy(env.state, 0), policy(env.state, 1)], axis=1)
            snapshot = env.state.take([0])
            _, _, _, info = env.step(actions)
            if info["game_over"][0]:
                break
            frames.append(draw(env.state))
        frames.append(draw(snapshot))
        if len(frames) >= args.min_turns:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:] + [frames[-1]] * 8,
                   duration=90, loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames (seed {seed})")


if __name__ == "__main__":
    main()
