"""Generate the figures and data tables used by report.tex.

  uv run --group train python docs/report/make_figures.py
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from snake_rl.agents import NeuralPolicy, load_policy
from snake_rl.env import SnakeEnv
from snake_rl.env.observations import (NUM_CHANNELS, WALL, build_observations,
                                       build_window_observations)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG, DATA = OUT / "figures", OUT / "data"

COLORS = [(245, 158, 11), (56, 189, 248), (167, 139, 250), (52, 211, 153),
          (244, 114, 182), (250, 204, 21), (251, 113, 133), (34, 211, 238)]
BG = [(14, 20, 38), (16, 23, 43)]
FOOD = (244, 63, 94)


# ---------------------------------------------------------------- curves
def curves(metrics_csv, prefix, every=5):
    rows = list(csv.DictReader(open(metrics_csv)))
    with open(DATA / f"{prefix}_eval.csv", "w") as f:
        f.write("steps,flood,greedy\n")
        for r in rows:
            if r["win_vs_flood_fill"]:
                f.write(f"{int(r['env_steps']) / 1e6:.4f},{float(r['win_vs_flood_fill']) * 100:.2f},"
                        f"{float(r['win_vs_greedy']) * 100:.2f}\n")
    with open(DATA / f"{prefix}_train.csv", "w") as f:
        f.write("steps,turns,entropy,kl\n")
        window = []
        for i, r in enumerate(rows):
            if r["mean_game_turns"]:
                window.append(float(r["mean_game_turns"]))
            if i % every == 0 and window:
                f.write(f"{int(r['env_steps']) / 1e6:.4f},{np.mean(window[-every:]):.1f},"
                        f"{float(r['entropy']):.4f},{float(r['approx_kl']):.5f}\n")


# ---------------------------------------------------------------- boards
def draw_board(state, env=0, cell=40, dead_alpha=0.0):
    b = state.board_size
    img = Image.new("RGB", (b * cell, b * cell))
    d = ImageDraw.Draw(img)
    for r in range(b):
        for c in range(b):
            d.rectangle([c * cell, r * cell, (c + 1) * cell, (r + 1) * cell], fill=BG[(r + c) % 2])
    for r, c in zip(*np.nonzero(state.food[env])):
        pad = cell * 0.28
        d.ellipse([c * cell + pad, r * cell + pad, (c + 1) * cell - pad, (r + 1) * cell - pad], fill=FOOD)
    for k in range(state.num_snakes):
        if not state.alive[env, k]:
            continue
        segs = [tuple(int(v) for v in p) for p in state.body[env, k, :int(state.length[env, k])]]
        w = cell * 0.64
        color = COLORS[k % len(COLORS)]
        pts = [((c + 0.5) * cell, (r + 0.5) * cell) for r, c in segs]
        for (x1, y1), (x2, y2) in zip(pts[1:], pts[:-1]):
            d.line([(x1, y1), (x2, y2)], fill=color, width=int(w))
        for x, y in pts:
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=color)
        hx, hy = pts[0]
        d.ellipse([hx - cell * 0.4, hy - cell * 0.4, hx + cell * 0.4, hy + cell * 0.4], fill=color)
    return img


def play_to(model, num_snakes, board, turn, seed, min_food=1, stochastic=True):
    """Play one game with the network in every seat; return the state at `turn`
    (re-rolling seeds until a game lasts that long with everyone alive)."""
    for s in range(seed, seed + 200):
        policy = NeuralPolicy(model, deterministic=not stochastic, seed=s)
        env = SnakeEnv(1, num_snakes, board, seed=s, min_food=min_food)
        env.reset()
        ok = True
        for _ in range(turn):
            actions = np.stack([policy(env.state, k) for k in range(num_snakes)], axis=1)
            _, _, _, info = env.step(actions)
            if info["game_over"][0] or not env.state.alive[0].all():
                ok = False
                break
        if ok:
            return env.state
    raise RuntimeError("no game reached the requested turn")


# ---------------------------------------------------------------- observations
def plane_image(plane, cell=14, color=(166, 25, 46), wall=False):
    h, w = plane.shape
    img = Image.new("RGB", (w * cell + 1, h * cell + 1), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for r in range(h):
        for c in range(w):
            v = float(plane[r, c])
            if wall:
                fill = (110, 104, 100) if v > 0.5 else (244, 242, 240)
            else:
                fill = tuple(int(255 - (255 - ch) * v) for ch in color) if v > 0 else (250, 249, 248)
            d.rectangle([c * cell, r * cell, (c + 1) * cell, (r + 1) * cell], fill=fill,
                        outline=(207, 201, 196))
    return img


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    curves(ROOT / "results" / "ppo_selfplay_training.csv", "duel")
    curves(ROOT / "results" / "ppo_arena_training.csv", "arena")

    duel = load_policy(ROOT / "runs" / "selfplay" / "best.pt")
    arena = load_policy(ROOT / "runs" / "arena" / "latest.pt")

    duel_state = play_to(duel, 2, 11, 45, seed=11)
    draw_board(duel_state, cell=44).save(FIG / "duel_board.png")
    arena_state = play_to(arena, 6, 19, 120, seed=5, min_food=6)
    draw_board(arena_state, cell=30).save(FIG / "arena_board.png")

    # The eight observation planes of the duel position, from snake 0's point of view.
    obs = build_observations(duel_state)[0, 0]
    names = ["own_head", "own_body", "own_age", "enemy_heads", "enemy_bodies",
             "danger", "food", "health"]
    for ch, name in enumerate(names):
        plane_image(obs[ch]).save(FIG / f"obs_{name}.png")

    # The arena window of snake 0: board context and wall plane.
    win = build_window_observations(arena_state, 17)[0, 0]
    occupancy = np.clip(win[1] + win[4] + 0.6 * win[0] + 0.6 * win[3], 0, 1)
    plane_image(occupancy, cell=12).save(FIG / "window_bodies.png")
    plane_image(win[6], cell=12, color=(244, 63, 94)).save(FIG / "window_food.png")
    plane_image(win[WALL], cell=12, wall=True).save(FIG / "window_wall.png")

    summary = {
        "duel_params": sum(p.numel() for p in duel.parameters()),
        "arena_params": sum(p.numel() for p in arena.parameters()),
        "duel_turn": int(duel_state.turn[0]), "arena_turn": int(arena_state.turn[0]),
        "obs_channels": NUM_CHANNELS,
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
