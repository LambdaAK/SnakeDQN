# Slither Mode: Continuous snake.io-style Self-Play — Design

Date: 2026-10-15
Status: Draft, awaiting review

## Context and goal

The project already has a finished grid Battlesnake mode, made up of:

- a batched NumPy engine verified against a reference implementation;
- PPO self-play networks for duels and an 8-snake arena;
- a browser demo in `web/` that runs the networks with ONNX Runtime Web and a JS rules
  port with parity tests.

It is a portfolio piece aimed at ML hiring.

This sub-project adds a **second, flashier game mode** in the style of slither.io:

- a circular arena;
- continuous movement and steering;
- snakes that grow by eating pellets and collapse into food when they die.

Agents learn it purely by self-play, as in the grid mode, and run live in the browser,
where a visitor can watch them or join with the mouse.

**Hard constraint: purely additive.** Nothing that exists today is deleted, renamed or
rewritten:

- the grid engine, models, scripts, tests, web demo, report and docs;
- the generated assets (`web/model.onnx`, `web/model_arena.onnx`, stats JSON).

New code lives in new modules. Edits to existing files are limited to:

- adding a link from the current demo page to the new one;
- a new README section;
- new `[dependency-groups]` entries only if one is ever needed (none are expected).

Decisions made during brainstorming:

| Question | Decision |
|---|---|
| Browser experience | Watch an AI arena **and** join it as a human (mouse steering) |
| Crowd size | 6–12 snakes per arena (12 slots, a subset enabled) |
| Objective | Endless with respawn; reward = mass gained, penalty on death |
| Boost | In v1 (steering bins × {cruise, boost}) |
| Web entry | Separate page `web/slither/`, linked from the existing header |
| Simulator | NumPy float64 on CPU, fixed-spacing trail bodies; network on MPS |
| Elo | Dropped for v1; mass share and kills per death vs bots instead |

Success means:

- a verified simulator with JS parity;
- a trained network whose time-averaged **mass share against 7 `hunter` bots is
  significantly above the fair share of 1/8**, with the 95% CI excluding 0.125 (the
  stretch target is 2× fair share);
- a smooth browser page deployed with the existing Pages workflow.

## 1. Repository layout (all new)

```
snake_rl/slither/
  __init__.py
  config.py        # SlitherConfig dataclass: every constant below, one place
  rules.py         # batched state + tick physics, pure NumPy float64
  observations.py  # sector observation builder
  env.py           # SlitherEnv: decisions, rewards, info, active masks
  bots.py          # wander, forager, hunter (batched, observation-driven)
  model.py         # MLP actor-critic + load helper
  eval.py          # net-vs-bots evaluation with bootstrap CIs
scripts/
  train_slither.py
  eval_slither.py
  benchmark_slither.py
  build_slither_assets.py   # -> web/slither/model.onnx, web/slither/stats.json
tests/
  slither_reference.py      # slow, loop-based reference rules
  test_slither_rules.py     # one test per rule
  test_slither_reference.py # batched == reference, batched == single-arena
  test_slither_observations.py
  test_slither_env.py
  test_slither_bots.py
  test_slither_js_parity.py
web/slither/
  index.html  page.js  slither-demo.js  slither-demo.css  rules.js  observe.js  render.js
  model.onnx  stats.json
  test/parity.mjs
```

`snake_rl/eval/harness.py` is reused (imported, not modified) for `bootstrap_mean_ci`.
`snake_rl/agents/model.py` is not modified. The slither MLP is its own class.

## 2. Game rules

All constants live in `SlitherConfig`. The values below are the v1 defaults. Units are
world units, and time is counted in physics ticks.

| Constant | Value | Meaning |
|---|---|---|
| `arena_radius` | 300 | circular arena centred at the origin |
| `tick_hz` | 30 | physics rate |
| `ticks_per_decision` | 3 | agent acts at 10 Hz |
| `max_turn` | 0.15 rad/tick | turn cap (≈4.5 rad/s) |
| `start_mass` | 10 | mass on (re)spawn |
| `max_mass` | 500 | cap |
| `length(m)` | `floor(20 + 2·m)` points | body length; capacity 1024 points |
| `radius(m)` | `2 + 0.5·sqrt(m)` | 3.6 at spawn, ≈13 at the cap |
| `boost_min_mass` | 15 | boost is unavailable below this |
| `boost_cost` | 0.2 mass/tick | |
| `boost_drop_every` | 5 ticks | drops one pellet of value 0.8 behind the tail |
| `eat_margin` | 3 | pellet eaten if `dist(head, pellet) < radius + eat_margin` |
| `food_capacity` | 1024 slots | per arena |
| `natural_food_target` | 200 | natural pellets kept topped up (≤2 spawned per tick), value 1 |
| `death_drop_fraction` | 0.8 | a dead snake's mass × 0.8 becomes pellets |
| `death_drop_max_pellets` | 64 | spread evenly along the body, jittered ±2 units |
| `respawn_delay` | 30 ticks | 1 s |
| `respawn_candidates` | 16 | random points in radius `R − 60` |
| `respawn_clearance` | 40 | from every live head |
| `max_snakes` | 12 | slots per arena |

**Bodies.** Each snake is a ring buffer of points spaced exactly 1 unit apart (one
cruise step), with index 0 at the head.

**Tick order.** Every tick, for every live snake:

1. **Steer.** `heading += clip(turn, −max_turn, +max_turn)`. Here `turn` is a float, so
   the agent's bins and a human's mouse share one code path.
2. **Move and trim.** Push 1 new head point, or 2 if boosting, 1 unit apart along
   `heading`.
   - Boosting means boost was requested and `mass ≥ boost_min_mass`.
   - A boosting snake pays `boost_cost` mass.
   - Then the body is trimmed: `length = min(length + points pushed, length(mass))`.
     A growing body therefore gains at most the points pushed this tick, because the
     tail stays put; a shrinking body has its tail cut.
   - Every `boost_drop_every` ticks of continuous boosting it drops a pellet at its
     (trimmed) tail point.
3. **Deaths**, all decided on positions after the move:
   - **Wall:** `|head| + radius > arena_radius`.
   - **Head-on:** `dist(head_i, head_j) < r_i + r_j`. Both die.
   - **Body hit:** `dist(head_i, p) < r_i + r_j` for a point `p` of snake `j ≠ i`, where
     `p` is not `j`'s head point. Snake `i` dies and `j` is credited with the kill.
   - There is no self-collision.
   - All deaths in a tick are simultaneous: a snake that dies this tick still kills
     others with its body this tick.
4. **Eat** (surviving snakes only). Each pellet within reach goes to the nearest eligible
   head, with ties going to the lower slot. Each snake's eaten values are summed in
   pellet-slot order, then `mass = min(mass + eaten, max_mass)`.
5. **Death drops.**
   - `n = min(ceil(mass/2), 64)` pellets of value `0.8·mass/n` are placed at body points
     `floor(q·length/n)`, plus jitter of ±2 units.
   - The snake's slot goes to the dead state (empty body, mass kept until respawn), with
     a respawn timer of `respawn_delay`.
6. **Respawn.** Slots that were dead at the start of the tick count down; for each slot,
   in slot order, whose timer reaches 0:
   - Use the first of the 16 candidate points whose distance to every live head is at
     least `respawn_clearance`. If none qualifies, take the candidate with the largest
     minimum distance.
   - The heading is random, and the body is laid out straight behind the head at
     `start_mass`.
7. **Pellets.** If natural pellets < target, spawn up to 2, uniformly in the disc of
   radius `R − 10`. Then all of this tick's new pellets are placed into free food slots
   in ascending slot order, in the order boost drops, death drops, natural pellets
   (each in slot order). When the slots run out, the excess is discarded. New pellets
   can't be eaten before the next tick.

**Determinism.** All randomness enters through a per-tick uniform array
`u: (num_envs, U)` with a documented fixed layout (respawn candidates and headings,
drop jitter, food spawn). `step_ticks(state, turn, boost, u)` is a pure function of its
inputs, so the reference implementation and the JS port can replay it exactly.
`SlitherEnv` draws `u` from its RNG.

**Slots.** `state.enabled (num_envs, 12)` marks which slots take part in an arena.
`SlitherEnv` samples 6–12 enabled slots per arena at construction, and they stay fixed.
Disabled slots are never alive.

**Collision cost.** The broad phase keeps a bounding circle per snake (head plus its
length) and skips head–snake pairs whose circles don't overlap. The narrow phase checks
only live points of the remaining pairs, chunked over arenas to bound memory. A spatial
hash is added only if `benchmark_slither.py` misses the throughput target in §5.

## 3. Agent interface

**Decision.** One decision holds for `ticks_per_decision = 3` ticks.

**Actions.** 18 discrete actions: `steer_bin × 2 + boost`.

- The 9 steering bins map to `turn = max_turn × {−1, −¾, −½, −¼, 0, ¼, ½, ¾, 1}` per
  tick.
- `action_mask (num_envs, 12, 18)` forbids boost actions when `mass < boost_min_mass`.
- A Gaussian continuous-steering head is a later experiment; the rules already take a
  float `turn`.

**Observation.** 263 float32 values, rotated to the snake's heading and computed at
decision time from 32 angular sectors of 11.25° each.

The vision radius is `V = 60`. A sector's distance features are encoded as `1 − d/V` for
the nearest object in that sector within `V`, and 0 if there is none. The 8 channels per
sector are:

1. **wall:** the analytic distance from the head to the arena boundary along the
   sector's centre ray;
2. **enemy body:** the nearest point of any other live snake, measured to its surface
   (`d − r_j`);
3. **enemy size:** the largest radius among enemy points seen in the sector, `/ 15`;
4. **own body:** points from index 10 onward, skipping the neck;
5. **food:** total pellet value in the sector within `V`, `/ 10`, clipped at 1;
6. **enemy head:** the nearest other head;
7. and 8. **the enemy head's heading** relative to our heading, as `cos` and `sin`. Both
   are 0 when there is no enemy head.

Then 7 scalars:

- `log(mass)/log(max_mass)`;
- `radius/15`;
- whether boosting last decision;
- whether boost is available;
- `|head|/R`;
- the bearing to the arena centre as `cos` and `sin`.

Sector index is `floor((atan2(rel) + π) / (2π/32)) mod 32`, after rotating into the
heading frame. Points are binned by angle, so the cost is O(heads × nearby points), not
O(rays × points). Dead or disabled slots get an all-zero observation.

**Reward.**

- Per decision: `0.01 × (mass change over the 3 ticks)`, including boost costs, food and
  kills eaten.
- On the decision where the snake dies: `−1` added.
- A `--survival-bonus` flag (default 0) adds a constant reward per decision alive.
- `gamma = 0.995` (≈20 s horizon at 10 Hz) and `lam = 0.95`.

**Episodes.** One life is one episode: `dones` is true on the decision a snake dies.
Arenas never reset. While a slot is dead or disabled it is padding, and
`info["active"]` is false. This is the same convention as `SnakeEnv`, so the PPO loop's
masking carries over unchanged.

**`SlitherEnv` API.** This mirrors `SnakeEnv`:

```python
env = SlitherEnv(num_envs, max_snakes=12, min_snakes=6, config=SlitherConfig(), seed=0)
obs, info = env.reset()           # obs (N, 12, 263) float32
obs, rewards, dones, info = env.step(actions)   # actions (N, 12) int in [0, 18)
```

`info` contains:

- `active`, `alive`, `enabled`, `action_mask`, `mass`;
- `kills` and `deaths` this decision, per slot;
- `killer`, the slot credited with each death, or −1.

**Network** (`snake_rl/slither/model.py`). `SlitherNet` is an MLP: 263 → 256 → 256 (ReLU),
with a policy head (18) and a value head (1). It has `config()` and `load_slither_policy`,
like `PolicyNet`. The masking reuses the same `masked_logits` semantics (copied, not
imported, to keep the modules independent). The flat-vector interface is deliberately
algorithm-agnostic, so the planned DQN/Rainbow comparison can use it as-is.

## 4. Training and evaluation

**`scripts/train_slither.py`** is a new file modelled on `train_selfplay.py`: the same
PPO loss, GAE, minibatch epochs, gradient clipping, metrics CSV and checkpointing
layout under `runs/slither/`.

- **Opponent pool, per slot:** each enabled slot is assigned to the learner or to a
  frozen snapshot from the pool.
  - The snapshot share is `--pool-prob` (default 0.3).
  - Assignments are redrawn whenever a slot's snake respawns.
  - Only learner slots produce training samples: `valid = active & learner`.
- **Defaults:**
  - `--num-envs 256`, `--rollout 128`;
  - `--hidden 256`, `--lr 2.5e-4`, `--ent-coef 0.01`;
  - `--device mps`, `--hours 2.5`.
- **Every 25 iterations:**
  - evaluate against `forager` and `hunter` (§4 evaluation);
  - save `iter_XXXXX.pt` and `latest.pt`;
  - keep `best.pt` by mass share vs `hunter`.

**Scripted bots** (`snake_rl/slither/bots.py`) read the same observation, so they are
batched and cheap, and portable to JS if ever wanted.

- `wander`: keeps its current turn with small random changes, and steers away from any
  sector whose wall, enemy-body or enemy-head reading exceeds a danger threshold. It
  never boosts.
- `forager`: picks the safe sector (all danger readings below the threshold) with the
  most food and steers toward it. When nothing is safe, it steers toward the least
  dangerous sector. It never boosts.
- `hunter`: a `forager` that also hunts. When an enemy head is within 30 units ahead and
  to one side and its own mass exceeds the enemy's, it boosts and steers across the
  enemy's path, if the sectors on that side are safe.

The danger threshold and hunt geometry are tuned so that `hunter` beats `forager` beats
`wander` on mass share. A test asserts this ordering.

**Evaluation** (`snake_rl/slither/eval.py`, CLI `scripts/eval_slither.py`).

- **Setup:** each evaluation arena has 8 enabled slots, with slot 0 the network
  (deterministic, argmax) and slots 1–7 one bot type. It runs for 3,000 decisions
  (5 min of game time) after a 100-decision warm-up.
- **Per arena, the following are measured:**
  - the network's time-averaged mass;
  - its time-averaged **mass share**, i.e. its mass / total snake mass;
  - kills, deaths and mean life length (in decisions).
- **Reporting:** means across arenas with 95% bootstrap CIs (`bootstrap_mean_ci`), and
  kills per death as a ratio of totals with a bootstrap CI over arenas.
- **Arena counts:** 64 during training and 300 for the final evaluation. Results are
  written to `results/slither_eval.json`.

**`scripts/build_slither_assets.py`:**

- exports `best.pt` to `web/slither/model.onnx` (input `obs (batch, 263)`, outputs
  `logits (batch, 18)` and `value (batch)`), verified against PyTorch with onnxruntime,
  as `build_web_assets.py` does;
- writes `web/slither/stats.json` with the final evaluation and the training curve.

## 5. Verification and performance

The grid engine's standard applies.

- **Rule unit tests** (`test_slither_rules.py`) cover:
  - spacing and straight-line movement;
  - the turn cap;
  - boost speed, cost, drops and the minimum-mass gate;
  - wall death;
  - body hit with kill credit;
  - head-on double death;
  - no self-collision;
  - eating, including nearest-head tie-breaking;
  - growth and shrink trimming;
  - death drops (count, value and the capacity limit);
  - respawn timer and clearance;
  - natural food top-up;
  - disabled slots staying dead.
- **Reference match:** `tests/slither_reference.py` is a plain-Python, per-arena,
  per-snake, loop-based implementation of §2. On random games with random turns and
  boosts, the batched engine must match it for at least 2,000 ticks: positions within
  1e-9, and discrete events (deaths, eats, kills, respawns, food slots) exactly.
- **Batched vs single:** stepping N arenas together gives the same result as stepping
  each alone.
- **Observation tests:** hand-built scenes with a known wall, enemy, food and head in
  known sectors, and with rotation invariance.
- **JS parity** (`test_slither_js_parity.py` and `web/slither/test/parity.mjs`):
  - Python generates single-tick cases: the state before, `turn`, `boost` and `u`, the
    state after, and the observations.
  - `rules.js` and `observe.js` must reproduce them, with positions within 1e-9 and the
    observations within 1e-6.
  - Both sides use float64 with the same operation order. JS numbers are float64, and
    the Python side does not use `np.float32` anywhere in the rules.
  - A last-bit (one-ulp) difference between NumPy's and V8's `sin`/`cos`/`atan2` could
    in principle flip a discrete outcome, but only for a distance within about 1e-15 of
    a threshold or an angle within about 1e-15 of a sector boundary. Across the whole
    fixture set that has a probability of roughly 1e-7, and fixtures use fixed seeds, so
    the test is deterministic. Cases are therefore not filtered; a failure is
    investigated, not retried.
  - The test is skipped when Node is unavailable, like the existing parity test.
- **Existing suite:** must stay green. No existing test is modified.
- **Throughput target:** `scripts/benchmark_slither.py` must report **≥ 20,000 snake
  decisions per second** (environment only, 256 arenas, about 9 snakes each, CPU, M3
  Max). If the target is missed, add a spatial hash (uniform grid by sort) to the
  collision and observation narrow phases.

## 6. Browser (`web/slither/`)

**Component and page.** The grid demo is now an embeddable component
(`web/snake-demo.js`: `mountSnakeDemo(root, {base, autoLoad, ortCdn})`, styles scoped to
`.snake-demo` and read from the host page's `--color-*` tokens). Slither follows the same
pattern so it can be embedded on alexkozik.com too:

- `web/slither/slither-demo.js` exports `mountSlitherDemo(root, {base, autoLoad, ortCdn})`.
  It loads ONNX Runtime on demand, can wait for a click before downloading the network,
  follows the host's `.dark` class and pauses while scrolled off-screen.
- `web/slither/slither-demo.css` is scoped to `.slither-demo`.
- `web/slither/index.html` and `page.js` are a thin standalone shell that reuses
  `../style.css` for the page chrome (fonts, palette, light-mode default, theme toggle).

The existing `web/index.html` gets a single "Slither →" link in its header. The new page
links back. Nothing else in the existing `web/` changes. The current Pages workflow
already deploys all of `web/`.

**Modules:**

- `rules.js`: a port of §2, with the same state layout and the same `u` layout.
  - `stepTick(state, turn, boost, u)`.
  - In live play `u` comes from `Math.random`, filled into a `Float64Array` of the
    documented size.
- `observe.js`: a port of §3's observation and action mask.
- `render.js`: Canvas 2D rendering:
  - a camera that follows a snake and zooms out slightly as it grows;
  - a background grid and a soft arena boundary;
  - pellets as glowing dots sized by value;
  - bodies as smooth tapered strokes through the trail points in per-snake colours,
    with eyes;
  - boosting snakes get a brighter outline;
  - head positions interpolated between physics ticks by the accumulator fraction, for
    smooth 60 fps motion.
- `slither-demo.js`: the component, with the main loop and UI described below.

**Loop.** A fixed-timestep accumulator steps physics at 30 Hz. Every 3 ticks, all AI
snakes run through one batched `InferenceSession.run` on an `obs (k, 263)` tensor.
Sampling uses the softmax with temperature 1 by default, with a "deterministic" toggle.

**Modes:**

- **Watch:** 10 AI snakes. Click a snake, or a leaderboard entry, to follow it.
- **Play:** the human takes one slot.
  - The mouse position sets a target heading, and each tick the human's `turn` is the
    angle to the target clipped to `±max_turn`. That is the same cap the AI has, without
    the 9-bin quantisation.
  - Holding the mouse button or Space boosts.
  - On death, an overlay shows final mass, kills and time alive, with "Play again".
- **Panel:** the leaderboard (top 5 by mass) and a minimap. For the followed AI snake,
  its 9-bin steering distribution is drawn as a fan, alongside its boost probability and
  value estimate. This mirrors the grid demo's probability panel.
- **Touch:** a touch sets the target heading. On touch devices a Boost button appears.
- **Embed:** `?embed=1` on the standalone page hides the page chrome, as on the main page.

## 7. Out of scope (possible follow-ups)

- A Gaussian continuous-action policy and the comparison with the discrete bins.
- Elo or league ratings among checkpoints.
- Arenas larger than 12 snakes, or a spatial hash unless the benchmark demands one.
- A LaTeX report chapter on the slither mode.
- Human-vs-AI leaderboards or persistence.

## Definition of done

1. All tests pass, both the new ones and the existing suite, unmodified.
2. The benchmark meets the throughput target, and the result is recorded in the README.
3. A trained network, with a final evaluation in `results/slither_eval.json` that meets
   the success criterion in the context section. If the criterion is not met after
   reasonable tuning, the results are reported honestly and the spec is revisited.
4. `web/slither/` works locally (Watch and Play) and is linked from the existing page.
5. The README gains a "Slither mode" section: results table, how it works, how to run.
   No existing README content is removed.
