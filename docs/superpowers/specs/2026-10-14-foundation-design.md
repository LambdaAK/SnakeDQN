# Foundation: Vectorized Multi-Snake Environment — Design

Date: 2026-10-14
Status: Draft, awaiting review

## Context and goal

The project is being repositioned from a single-snake DQN hobby project into a
portfolio piece whose headline is **multi-agent self-play Snake** (snakes trained
against past versions of themselves, ranked by Elo, playable in a browser demo).
The primary audience is ML hiring: a visitor should be hooked in 30–60 seconds by
an interactive demo, and an engineer digging deeper should find clean, tested code
and rigorous benchmarks.

The overall work is split into five sub-projects, each with its own spec → plan →
implementation cycle:

1. **Foundation** (this document)
2. Algorithm benchmark — DQN → Rainbow components, PPO, on single-snake mode, multi-seed
3. Self-play league — PPO self-play, opponent pool, Elo tracking
4. Browser demo — TypeScript rules port + ONNX agents, value/probability overlays
5. Write-up and stretch goals — results page, live Battlesnake ladder bot, MCTS

Constraints:

- Stack: Python, NumPy, PyTorch (no JAX).
- Compute: the user's Mac (CPU). Cloud GPU (Thunder Compute) only if it becomes
  clearly necessary in later sub-projects.
- Rules follow standard Battlesnake so a live ladder bot remains possible later.

## Scope

In scope: repo restructure, a vectorized multi-snake environment, observations,
reward functions, scripted baseline bots, an evaluation harness, terminal play,
tests and CI, and repo cleanup.

Out of scope: training code, neural networks, Elo, the browser demo.

## 1. Repository layout

The project becomes an installable package configured by `pyproject.toml`.

```
snake_rl/
  env/
    rules.py         # batched Battlesnake rules, pure NumPy
    observations.py  # egocentric grid observation builder
    rewards.py       # pluggable reward functions
    render.py        # terminal renderer (adapted from the current _render)
  baselines/         # random, safe-random, greedy, flood-fill, hamiltonian bots
  eval/              # match runner and statistics
scripts/
  play.py            # human play via WASD (solo or against a bot)
  evaluate.py        # CLI wrapper around the evaluation harness
  benchmark_env.py   # prints env steps/sec
tests/               # pytest suite
legacy/              # current code, kept temporarily (see Definition of done)
```

Repo hygiene:

- Add a `.gitignore` covering `__pycache__/`, `.DS_Store`, checkpoints (`*.pth`,
  `checkpoints/`), and run outputs.
- Remove the committed `__pycache__/`, `.DS_Store`, and `agents/*.pth` from version
  control. The checkpoint files stay on disk locally.
- Move the existing modules (`snake_game.py`, `dqn_agent.py`, `reinforce_agent.py`,
  `train.py`, `run_trained_model.py`, `monitor_training.py`, `test_dqn.py`,
  `colors.py`, and the two `.txt` specs) into `legacy/`.

## 2. Environment API

A single class steps many games at once with all state held in NumPy arrays and
no Python loop over individual games:

```python
env = SnakeEnv(num_envs: int, num_snakes: int, board_size: int = 11,
               reward: str = "winloss", seed: int | None = None)
obs, info = env.reset()
obs, rewards, dones, info = env.step(actions)
```

- `obs`: `float32` array of shape `(num_envs, num_snakes, C, H, W)`.
- `actions`: `int` array of shape `(num_envs, num_snakes)`, values 0–3 meaning
  up, down, left, right (absolute directions, matching the Battlesnake API).
  Actions for already-dead snakes are ignored.
- `rewards`: `float32` array `(num_envs, num_snakes)`.
- `dones`: `bool` array `(num_envs, num_snakes)`, true on the step a snake dies or
  its game ends.
- A game ends when at most one snake remains alive (for `num_snakes == 1`, when
  the snake dies). Finished games auto-reset; the terminal observation is
  returned in `info["final_obs"]` for bootstrapping.
- `info["action_mask"]`: `bool` array `(num_envs, num_snakes, 4)` of moves that do
  not cause immediate death from walls or bodies. Agents may use or ignore it.
- `info` also carries per-game `turn`, per-snake `length`, `health`, `alive`, and
  per-game `winner` on the step a game ends (`-1` for a draw).
- Single-snake mode is `num_snakes=1`; it replaces the legacy game and gives
  sub-project 2 the same code path as self-play.

### Rules (standard Battlesnake)

- Board 11×11 by default. Snakes start at length 3 (all three segments stacked on
  the start cell) with health 100, at symmetric start positions.
- Each turn, all snakes move simultaneously. Health decreases by 1 per turn.
- Eating food restores health to 100 and grows the snake by one segment (the tail
  does not move on the following turn).
- A snake dies if it moves off the board, into any snake body segment (including
  its own neck — reversing is fatal), or its health reaches 0.
- Moving into a cell occupied by a tail that is moving away this turn is safe,
  unless that snake ate on the previous turn (its tail stays).
- Head-to-head: the shorter snake dies; equal lengths both die.
- Food: at least one piece is always on the board; additionally each turn has a
  15% chance of spawning one food on a random free cell. Initial food follows
  Battlesnake defaults (one piece near each snake plus one in the center).
- No hazards.

### Observation

Egocentric from each snake's perspective (snake *i*'s own snake is always
"self"). Channels, each `H × W`:

1. own head
2. own body
3. own body age (segment index / length, so the tail position is known)
4. enemy heads
5. enemy bodies
6. enemy heads of snakes at least as long as self (dangerous heads)
7. food
8. own health / 100, broadcast over the grid

The channel count `C` is a module constant. No hand-engineered features (A*,
dead-end flags) appear in the observation. A wrapper that reproduces the legacy
33-feature vector for single-snake mode is optional and deferred to sub-project 2,
where it is needed for the engineered-vs-learned comparison.

### Performance target

At least 200,000 environment steps per second (one step = one turn of one game)
at `num_envs=1024`, `num_snakes=2`, on the user's Mac CPU, measured by
`scripts/benchmark_env.py`.

## 3. Rewards

`rewards.py` defines named reward functions with the signature
`fn(events) -> float32[num_envs, num_snakes]`, where `events` is a struct of
per-step arrays (food eaten, died, won, game over). The env selects one by name.

- `"winloss"` (default for `num_snakes > 1`): +1 to the winner when the game ends,
  −1 on death, 0 otherwise. If all remaining snakes die on the same step, all of
  them receive 0.
- `"food"` (default for `num_snakes == 1`): +1 per food eaten, −1 on death.

The legacy ±0.3 distance shaping is dropped. Additional shaping terms are
experiment options for later sub-projects, not part of the rules.

## 4. Baseline bots

Each bot is a callable `policy(state, snake_index) -> actions[num_envs]` operating
on the whole batch, with the same interface the evaluation harness uses for
learned agents later.

1. **Random** — uniform over the 4 actions.
2. **Safe-random** — uniform over actions allowed by `action_mask` (any action if
   none are allowed).
3. **Greedy** — among safe actions, move toward the nearest food by Manhattan
   distance.
4. **Flood-fill heuristic** — for each safe action, compute reachable free area by
   flood fill; avoid cells an equal-or-longer enemy head could reach next turn;
   seek food when health is low or when not the longest snake; otherwise prefer
   the action with the largest reachable area.
5. **Hamiltonian** (single-snake only) — follows a fixed Hamiltonian cycle; never
   dies and eventually fills the board. Serves as the perfect-play reference.

Bots may loop over games internally in Python where vectorization is impractical
(flood fill); they are evaluation tools, not the training hot path.

## 5. Evaluation harness

```python
run_matches(policies: list[Policy], n_games: int, seed: int,
            board_size: int = 11) -> MatchResult
```

- Plays `n_games` games with the given policies seated as snakes (one policy per
  seat), using the vectorized env.
- `MatchResult` reports per policy: wins, draws, losses, win rate with a 95%
  Wilson confidence interval, mean final length, mean game length in turns.
- For single-snake runs, it reports mean / median / max score with a 95% bootstrap
  confidence interval.
- Results serialize to JSON; a summary table prints to the terminal.
- All randomness flows from the `seed` argument, so runs are reproducible.

Scripts:

- `scripts/evaluate.py` — CLI: choose bots or checkpoints per seat, number of
  games, seed, output path.
- `scripts/play.py` — terminal play with WASD, solo or against a chosen bot, using
  the adapted renderer.

## 6. Testing

pytest, run in GitHub Actions CI on every push.

- **Rule tests** on small handcrafted boards, one per rule: wall collision, self
  collision, other-body collision, neck reversal, head-to-head (longer, shorter,
  equal), eating (growth and health reset), starvation, moving into a vacating
  tail (safe) versus a tail that just ate (fatal), food spawning, auto-reset.
- **Batch consistency**: stepping 1024 games batched yields results identical to
  stepping each game individually with the same seeds.
- **Reference oracle**: a slow, readable pure-Python implementation of the rules
  in `tests/`; randomized games must match the vectorized env step for step.
- **Observation tests**: shapes, channel contents on known boards, egocentric
  symmetry (swapping two snakes swaps their observations).
- **Reward tests**: values for win, loss, draw, food, death.
- **Bot tests**: safe-random never takes an action excluded by the mask when an
  allowed one exists; greedy beats safe-random and flood-fill beats greedy at more
  than 60% win rate over 500 games each.
- **Benchmark**: `scripts/benchmark_env.py` prints steps/sec. It is not a pass/fail
  test.

## Definition of done

1. All tests pass in CI.
2. `SnakeEnv` meets the 200k steps/sec target on the user's Mac.
3. The harness confirms flood-fill > greedy > safe-random > random, with results
   saved as JSON.
4. `scripts/play.py` works solo and against a bot.
5. Package layout, `.gitignore`, and cleanup are in place; the README is rewritten
   around the new direction.
6. `legacy/` is deleted once the new single-snake mode has replaced it.
