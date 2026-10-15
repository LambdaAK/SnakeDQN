"""PPO self-play for the slither mode (spec docs/superpowers/specs/2026-10-15-slither-mode-design.md §4).

Every arena has 6-12 snakes. Each life is played either by the learner or, with
probability --pool-prob, by a frozen snapshot from the opponent pool; only learner lives
produce training samples. A slot's owner is redrawn when its snake dies.

  uv run --group train python scripts/train_slither.py --device mps --hours 2.5
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

from snake_rl.slither.actions import NUM_ACTIONS
from snake_rl.slither.env import SlitherEnv
from snake_rl.slither.eval import evaluate_vs_bot
from snake_rl.slither.model import SlitherNet, SlitherPolicy, masked_logits

EVAL_BOTS = ("forager", "hunter")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/slither")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--max-snakes", type=int, default=12)
    p.add_argument("--min-snakes", type=int, default=6)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--iterations", type=int, default=100_000)
    p.add_argument("--hours", type=float, default=None, help="stop after this much wall time")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--mass-scale", type=float, default=0.1)
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--survival-bonus", type=float, default=0.0)
    p.add_argument("--pool-prob", type=float, default=0.3)
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-arenas", type=int, default=32)
    p.add_argument("--eval-decisions", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def evaluate(model, args, seed):
    """Learner in slot 0 against 7 copies of each evaluation bot."""
    policy = SlitherPolicy(model)
    out = {}
    for bot in EVAL_BOTS:
        r = evaluate_vs_bot(policy, bot, arenas=args.eval_arenas, decisions=args.eval_decisions,
                            warmup=min(100, args.eval_decisions), seed=seed)
        out[f"mass_share_vs_{bot}"] = r.mass_share
        out[f"kpd_vs_{bot}"] = r.kills_per_death
        out[f"mass_vs_{bot}"] = r.mean_mass
    return out


def save_checkpoint(path, model, iteration, env_steps, metrics=None):
    torch.save({"config": model.config(), "model": model.state_dict(), "iteration": iteration,
                "env_steps": env_steps, "metrics": metrics or {}}, path)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    n, s, t_len = args.num_envs, args.max_snakes, args.rollout
    env = SlitherEnv(n, max_snakes=s, min_snakes=args.min_snakes, seed=args.seed,
                     survival_bonus=args.survival_bonus, mass_scale=args.mass_scale,
                     death_penalty=args.death_penalty)
    d = env.obs_size
    model = SlitherNet(d, args.hidden).to(device)
    opponent = SlitherNet(d, args.hidden).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    pool = [copy.deepcopy(model.state_dict())]

    obs, info = env.reset()
    learner = rng.random((n, s)) >= args.pool_prob

    obs_buf = np.zeros((t_len, n, s, d), np.float16)
    mask_buf = np.zeros((t_len, n, s, NUM_ACTIONS), bool)
    act_buf = np.zeros((t_len, n, s), np.int64)
    logp_buf = np.zeros((t_len, n, s), np.float32)
    val_buf = np.zeros((t_len, n, s), np.float32)
    rew_buf = np.zeros((t_len, n, s), np.float32)
    done_buf = np.zeros((t_len, n, s), bool)
    valid_buf = np.zeros((t_len, n, s), bool)

    metrics_path = run_dir / "metrics.csv"
    fields = ["iteration", "env_steps", "hours", "samples", "deaths", "kills", "mean_reward",
              "mean_mass", "policy_loss", "value_loss", "entropy", "approx_kl"]
    fields += [f"{m}_vs_{bot}" for bot in EVAL_BOTS for m in ("mass_share", "kpd", "mass")]
    with metrics_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    start = time.time()
    env_steps = 0
    best = -1.0
    for iteration in range(1, args.iterations + 1):
        opponent.load_state_dict(pool[rng.integers(len(pool))])
        deaths = kills = 0

        model.eval()
        for t in range(t_len):
            mask = info["action_mask"]
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.reshape(n * s, d)).to(device)
                mask_t = torch.from_numpy(mask.reshape(n * s, NUM_ACTIONS)).to(device)
                logits, value = model(obs_t)
                dist = Categorical(logits=masked_logits(logits, mask_t))
                action = dist.sample()
                logp = dist.log_prob(action)
                actions = action.cpu().numpy().reshape(n, s)
                opp = np.flatnonzero(~learner.reshape(-1))
                if opp.size:
                    opp_t = torch.from_numpy(opp).to(device)
                    o_logits, _ = opponent(obs_t[opp_t])
                    o_action = Categorical(logits=masked_logits(o_logits, mask_t[opp_t])).sample()
                    actions.reshape(-1)[opp] = o_action.cpu().numpy()

            obs_buf[t] = obs
            mask_buf[t] = mask
            act_buf[t] = actions
            logp_buf[t] = logp.cpu().numpy().reshape(n, s)
            val_buf[t] = value.cpu().numpy().reshape(n, s)

            obs, rewards, dones, info = env.step(actions)
            rew_buf[t] = rewards
            done_buf[t] = dones
            valid_buf[t] = info["active"] & learner
            deaths += int((dones & learner).sum())
            kills += int((info["kills"] * learner).sum())
            env_steps += n
            died = np.flatnonzero(dones.reshape(-1))
            if died.size:  # each new life gets a fresh owner
                learner.reshape(-1)[died] = rng.random(died.size) >= args.pool_prob

        with torch.no_grad():
            _, next_value = model(torch.from_numpy(obs.reshape(n * s, d)).to(device))
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
        b_obs = torch.from_numpy(obs_buf.reshape(-1, d)[valid].astype(np.float32)).to(device)
        b_mask = torch.from_numpy(mask_buf.reshape(-1, NUM_ACTIONS)[valid]).to(device)
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
        valid_rew = rew_buf[valid_buf]
        row = {
            "iteration": iteration, "env_steps": env_steps, "hours": round(hours, 4),
            "samples": samples, "deaths": deaths, "kills": kills,
            "mean_reward": round(float(valid_rew.mean()), 5) if valid_rew.size else "",
            "mean_mass": round(float(info["mass"][info["alive"] & learner].mean()), 2)
            if (info["alive"] & learner).any() else "",
            **{k: round(float(np.mean(v)), 5) for k, v in stats.items()},
        }

        if iteration % args.snapshot_every == 0:
            pool.append(copy.deepcopy(model.state_dict()))
            pool = pool[-args.pool_size:]

        stop = args.hours is not None and hours >= args.hours
        if iteration % args.eval_every == 0 or stop or iteration == args.iterations:
            model.eval()
            row.update({k: round(v, 4) for k, v in evaluate(model, args, iteration).items()})
            save_checkpoint(run_dir / "checkpoints" / f"iter_{iteration:05d}.pt", model, iteration, env_steps, row)
            save_checkpoint(run_dir / "latest.pt", model, iteration, env_steps, row)
            if row["mass_share_vs_hunter"] > best:
                best = row["mass_share_vs_hunter"]
                save_checkpoint(run_dir / "best.pt", model, iteration, env_steps, row)
            print(f"[eval] iter {iteration} steps {env_steps:,} mass share vs hunter "
                  f"{row['mass_share_vs_hunter']:.3f} vs forager {row['mass_share_vs_forager']:.3f} "
                  f"kpd vs hunter {row['kpd_vs_hunter']:.2f}", flush=True)

        with metrics_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"iter {iteration} steps {env_steps:,} {hours:.2f}h samples {samples} deaths {deaths} "
              f"kills {kills} reward {row['mean_reward']} mass {row['mean_mass']} "
              f"ent {row['entropy']:.3f} kl {row['approx_kl']:.4f}", flush=True)
        if stop:
            break


if __name__ == "__main__":
    main()
