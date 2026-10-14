"""PPO self-play training on 11x11 two-snake Battlesnake.

Each env seats the learner as snake 0. Snake 1 is the learner too (pure self-play,
both seats produce training data) or, with probability --pool-prob, a frozen past
snapshot. Progress is measured against the scripted flood-fill and greedy bots.

  uv run --group train python scripts/train_selfplay.py --hours 3
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from snake_rl.agents import NeuralPolicy, PolicyNet, masked_logits
from snake_rl.baselines import make_bot
from snake_rl.env import SnakeEnv
from snake_rl.env.observations import NUM_CHANNELS
from snake_rl.eval import run_matches


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/selfplay")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--iterations", type=int, default=100_000)
    p.add_argument("--hours", type=float, default=None, help="stop after this much wall time")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--food-bonus", type=float, default=0.05,
                   help="shaping reward per food, on top of +1 win / -1 loss")
    p.add_argument("--pool-prob", type=float, default=0.3)
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def evaluate(model, games, seed):
    policy = NeuralPolicy(model, name="ppo")
    out = {}
    for bot in ("flood_fill", "greedy"):
        result = run_matches([policy, make_bot(bot, seed=seed + 1)], games, seed)
        seat = result.seats[0]
        out[f"win_vs_{bot}"] = seat.win_rate
        out[f"draw_vs_{bot}"] = seat.draws / games
        out[f"len_vs_{bot}"] = seat.mean_length
    return out


def save_checkpoint(path, model, iteration, env_steps, metrics=None):
    torch.save({"config": model.config(), "model": model.state_dict(),
                "iteration": iteration, "env_steps": env_steps, "metrics": metrics or {}}, path)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    n, s, b, c, t_len = args.num_envs, 2, 11, NUM_CHANNELS, args.rollout
    env = SnakeEnv(n, s, b, reward="winloss", seed=args.seed)
    model = PolicyNet(b).to(device)
    opponent = PolicyNet(b).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    pool = [copy.deepcopy(model.state_dict())]

    obs, info = env.reset()
    use_pool = rng.random(n) < args.pool_prob

    obs_buf = np.zeros((t_len, n, s, c, b, b), np.float16)
    mask_buf = np.zeros((t_len, n, s, 4), bool)
    act_buf = np.zeros((t_len, n, s), np.int64)
    logp_buf = np.zeros((t_len, n, s), np.float32)
    val_buf = np.zeros((t_len, n, s), np.float32)
    rew_buf = np.zeros((t_len, n, s), np.float32)
    done_buf = np.zeros((t_len, n, s), bool)
    valid_buf = np.zeros((t_len, n, s), bool)

    metrics_path = run_dir / "metrics.csv"
    fields = ["iteration", "env_steps", "hours", "samples", "games", "pool_win_rate",
              "mean_game_turns", "policy_loss", "value_loss", "entropy", "approx_kl",
              "win_vs_flood_fill", "draw_vs_flood_fill", "len_vs_flood_fill",
              "win_vs_greedy", "draw_vs_greedy", "len_vs_greedy"]
    with metrics_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    start = time.time()
    env_steps = 0
    best = -1.0
    for iteration in range(1, args.iterations + 1):
        opponent.load_state_dict(pool[rng.integers(len(pool))])
        games = pool_games = pool_wins = 0
        game_turns = []

        model.eval()
        for t in range(t_len):
            learner = np.ones((n, s), bool)
            learner[:, 1] = ~use_pool
            mask = info["action_mask"]
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.reshape(n * s, c, b, b)).to(device)
                mask_t = torch.from_numpy(mask.reshape(n * s, 4)).to(device)
                logits, value = model(obs_t)
                dist = Categorical(logits=masked_logits(logits, mask_t))
                action = dist.sample()
                logp = dist.log_prob(action)
                actions = action.cpu().numpy().reshape(n, s)
                pool_envs = np.flatnonzero(use_pool)
                if pool_envs.size:
                    o_logits, _ = opponent(torch.from_numpy(obs[pool_envs, 1]).to(device))
                    o_mask = torch.from_numpy(mask[pool_envs, 1]).to(device)
                    o_action = Categorical(logits=masked_logits(o_logits, o_mask)).sample()
                    actions[pool_envs, 1] = o_action.cpu().numpy()

            obs_buf[t] = obs
            mask_buf[t] = mask
            act_buf[t] = actions
            logp_buf[t] = logp.cpu().numpy().reshape(n, s)
            val_buf[t] = value.cpu().numpy().reshape(n, s)

            prev_length = info["length"]
            obs, rewards, dones, info = env.step(actions)
            ate = (info["final_length"] > prev_length) & info["active"]
            rew_buf[t] = rewards + args.food_bonus * ate
            done_buf[t] = dones
            valid_buf[t] = info["active"] & learner
            env_steps += n

            ended = np.flatnonzero(info["game_over"])
            if ended.size:
                games += ended.size
                game_turns.extend(info["final_turn"][ended].tolist())
                pooled = ended[use_pool[ended]]
                pool_games += pooled.size
                pool_wins += int((info["winner"][pooled] == 0).sum())
                use_pool[ended] = rng.random(ended.size) < args.pool_prob

        with torch.no_grad():
            _, next_value = model(torch.from_numpy(obs.reshape(n * s, c, b, b)).to(device))
            next_value = next_value.cpu().numpy().reshape(n, s)
        adv = np.zeros_like(rew_buf)
        last = np.zeros((n, s), np.float32)
        for t in reversed(range(t_len)):
            nv = next_value if t == t_len - 1 else val_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * nv * nonterminal - val_buf[t]
            last = delta + args.gamma * args.lam * nonterminal * last
            adv[t] = last
        returns = adv + val_buf

        valid = valid_buf.reshape(-1)
        b_obs = torch.from_numpy(obs_buf.reshape(-1, c, b, b)[valid].astype(np.float32)).to(device)
        b_mask = torch.from_numpy(mask_buf.reshape(-1, 4)[valid]).to(device)
        b_act = torch.from_numpy(act_buf.reshape(-1)[valid]).to(device)
        b_logp = torch.from_numpy(logp_buf.reshape(-1)[valid]).to(device)
        b_adv = torch.from_numpy(adv.reshape(-1)[valid]).to(device)
        b_ret = torch.from_numpy(returns.reshape(-1)[valid]).to(device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        model.train()
        samples = b_obs.shape[0]
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        for _ in range(args.epochs):
            perm = torch.randperm(samples, device=device)
            for lo in range(0, samples, args.minibatch):
                idx = perm[lo:lo + args.minibatch]
                logits, value = model(b_obs[idx])
                dist = Categorical(logits=masked_logits(logits, b_mask[idx]))
                logp = dist.log_prob(b_act[idx])
                ratio = torch.exp(logp - b_logp[idx])
                pg = -torch.min(ratio * b_adv[idx],
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[idx]).mean()
                v_loss = 0.5 * ((value - b_ret[idx]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                with torch.no_grad():
                    stats["approx_kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
                stats["policy_loss"].append(pg.item())
                stats["value_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.item())

        hours = (time.time() - start) / 3600
        row = {
            "iteration": iteration, "env_steps": env_steps, "hours": round(hours, 4),
            "samples": samples, "games": games,
            "pool_win_rate": round(pool_wins / pool_games, 4) if pool_games else "",
            "mean_game_turns": round(float(np.mean(game_turns)), 1) if game_turns else "",
            **{k: round(float(np.mean(v)), 5) for k, v in stats.items()},
        }

        if iteration % args.snapshot_every == 0:
            pool.append(copy.deepcopy(model.state_dict()))
            pool = pool[-args.pool_size:]

        stop = args.hours is not None and hours >= args.hours
        if iteration % args.eval_every == 0 or stop or iteration == args.iterations:
            model.eval()
            row.update({k: round(v, 4) for k, v in
                        evaluate(model, args.eval_games, seed=iteration).items()})
            save_checkpoint(run_dir / "checkpoints" / f"iter_{iteration:05d}.pt",
                            model, iteration, env_steps, row)
            save_checkpoint(run_dir / "latest.pt", model, iteration, env_steps, row)
            score = row["win_vs_flood_fill"] + 0.5 * row["draw_vs_flood_fill"]
            if score > best:
                best = score
                save_checkpoint(run_dir / "best.pt", model, iteration, env_steps, row)
            print(f"[eval] iter {iteration} steps {env_steps:,} "
                  f"vs flood_fill win {row['win_vs_flood_fill']:.3f} "
                  f"vs greedy win {row['win_vs_greedy']:.3f}", flush=True)

        with metrics_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"iter {iteration} steps {env_steps:,} {hours:.2f}h samples {samples} "
              f"games {games} pool_win {row['pool_win_rate']} turns {row['mean_game_turns']} "
              f"ent {row['entropy']:.3f} kl {row['approx_kl']:.4f}", flush=True)
        if stop:
            break


if __name__ == "__main__":
    main()
