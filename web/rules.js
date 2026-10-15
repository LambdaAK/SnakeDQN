// Battlesnake rules for one game, ported from snake_rl/env/rules.py and
// snake_rl/env/observations.py. Same turn order (move, health, feed, spawn food,
// eliminate) and the same food-spawn draw, so Python fixtures replay exactly.

export const START_LENGTH = 3;
export const MIN_FOOD = 1;
export const SPAWN_CHANCE = 0.15;
export const NUM_CHANNELS = 8;
// Action index -> [dRow, dCol]: up, down, left, right.
export const DIRECTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const DIAGONALS = [[-1, -1], [-1, 1], [1, -1], [1, 1]];

export function newGame(boardSize, numSnakes, random = Math.random, maxHealth = 100, minFood = MIN_FOOD) {
  const b = boardSize;
  const a = 1, m = Math.floor(b / 2), z = b - 2;
  const corners = [[a, a], [a, z], [z, a], [z, z]];
  const cardinals = [[a, m], [m, a], [m, z], [z, m]];
  const [first, second] = random() < 0.5 ? [corners, cardinals] : [cardinals, corners];
  const starts = [...shuffle(first, random), ...shuffle(second, random)].slice(0, numSnakes);
  const snakes = starts.map(([r, c]) => ({
    body: Array.from({ length: START_LENGTH }, () => [r, c]),
    health: maxHealth,
    alive: true,
  }));
  const food = new Uint8Array(b * b);
  const centre = m;
  food[centre * b + centre] = 1;
  for (const [r, c] of starts) {
    const startDist = Math.abs(r - centre) + Math.abs(c - centre);
    const allowed = DIAGONALS.map(([dr, dc]) => [r + dr, c + dc]).filter(([fr, fc]) => {
      const corner = (fr === 0 || fr === b - 1) && (fc === 0 || fc === b - 1);
      return !corner && Math.abs(fr - centre) + Math.abs(fc - centre) >= startDist;
    });
    const [fr, fc] = allowed[Math.floor(random() * allowed.length)];
    food[fr * b + fc] = 1;
  }
  return { boardSize: b, maxHealth, minFood, snakes, food, turn: 0, over: false, winner: -1 };
}

function shuffle(items, random) {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

const onBoard = (b, [r, c]) => r >= 0 && r < b && c >= 0 && c < b;

// Advance one turn in place. actions: one 0-3 per snake (ignored for dead snakes).
// u0 decides the 15% food spawn, u1 picks which free cell. Returns events.
export function step(game, actions, u0 = Math.random(), u1 = Math.random()) {
  const events = { ate: game.snakes.map(() => false), died: game.snakes.map(() => false),
                   gameOver: false, winner: -1 };
  if (game.over) return events;
  const b = game.boardSize;
  const movers = game.snakes.map((s, i) => (s.alive ? i : -1)).filter((i) => i >= 0);

  for (const i of movers) {
    const snake = game.snakes[i];
    const [dr, dc] = DIRECTIONS[actions[i]];
    const head = [snake.body[0][0] + dr, snake.body[0][1] + dc];
    snake.body = [head, ...snake.body.slice(0, -1)];
    snake.health -= 1;
  }

  const eaten = [];
  for (const i of movers) {
    const snake = game.snakes[i];
    const head = snake.body[0];
    if (onBoard(b, head) && game.food[head[0] * b + head[1]]) {
      snake.health = game.maxHealth;
      snake.body.push(snake.body[snake.body.length - 1].slice());
      eaten.push(head[0] * b + head[1]);
      events.ate[i] = true;
    }
  }
  for (const cell of eaten) game.food[cell] = 0;

  let foodCount = 0;
  for (const f of game.food) foodCount += f;
  if (foodCount < (game.minFood ?? MIN_FOOD) || u0 < SPAWN_CHANCE) {
    const occupied = new Uint8Array(b * b);
    for (const i of movers) {
      for (const seg of game.snakes[i].body) if (onBoard(b, seg)) occupied[seg[0] * b + seg[1]] = 1;
    }
    const free = [];
    for (let cell = 0; cell < b * b; cell++) if (!game.food[cell] && !occupied[cell]) free.push(cell);
    if (free.length) game.food[free[Math.floor(u1 * free.length)]] = 1;
  }

  const survivors = movers.filter((i) => game.snakes[i].health > 0 && onBoard(b, game.snakes[i].body[0]));
  const bodyCells = new Set();
  for (const i of survivors) {
    for (const seg of game.snakes[i].body.slice(1)) bodyCells.add(seg[0] * b + seg[1]);
  }
  const eliminated = new Set(movers.filter((i) => !survivors.includes(i)));
  for (const i of survivors) {
    const head = game.snakes[i].body[0];
    const cell = head[0] * b + head[1];
    if (bodyCells.has(cell)) eliminated.add(i);
    for (const k of survivors) {
      const other = game.snakes[k];
      if (k !== i && other.body[0][0] === head[0] && other.body[0][1] === head[1]
          && other.body.length >= game.snakes[i].body.length) eliminated.add(i);
    }
  }
  for (const i of eliminated) {
    game.snakes[i].alive = false;
    events.died[i] = true;
  }

  game.turn += 1;
  const alive = game.snakes.map((s, i) => (s.alive ? i : -1)).filter((i) => i >= 0);
  game.over = game.snakes.length === 1 ? alive.length === 0 : alive.length <= 1;
  if (game.over) {
    game.winner = game.snakes.length > 1 && alive.length === 1 ? alive[0] : -1;
    events.gameOver = true;
    events.winner = game.winner;
  }
  return events;
}

// Moves for snake k that don't immediately hit a wall or a body (tails assumed to move).
export function actionMask(game, k) {
  const b = game.boardSize;
  const snake = game.snakes[k];
  if (!snake.alive || game.over) return [false, false, false, false];
  const blocked = new Uint8Array(b * b);
  for (const s of game.snakes) {
    if (!s.alive) continue;
    for (const seg of s.body.slice(0, -1)) blocked[seg[0] * b + seg[1]] = 1;
  }
  return DIRECTIONS.map(([dr, dc]) => {
    const r = snake.body[0][0] + dr, c = snake.body[0][1] + dc;
    return onBoard(b, [r, c]) && !blocked[r * b + c];
  });
}

// Egocentric observation for snake k: Float32Array(NUM_CHANNELS * B * B), channel-major.
// Channels: own head, own body, own body age, enemy heads, enemy bodies, dangerous
// enemy heads (at least as long), food, own health / max.
export function observation(game, k) {
  const b = game.boardSize, plane = b * b;
  const out = new Float32Array(NUM_CHANNELS * plane);
  const me = game.snakes[k];
  game.snakes.forEach((snake, i) => {
    if (!snake.alive) return;
    const own = i === k;
    const len = snake.body.length;
    const dangerous = !own && len >= me.body.length;
    // Tail-first, so where segments share a cell the one nearest the head wins.
    for (let j = len - 1; j >= 0; j--) {
      const [r, c] = snake.body[j];
      const cell = r * b + c;
      if (own) {
        if (j === 0) out[0 * plane + cell] = 1;
        else out[1 * plane + cell] = 1;
        out[2 * plane + cell] = (len - j) / len;
      } else if (j === 0) {
        out[3 * plane + cell] = 1;
        if (dangerous) out[5 * plane + cell] = 1;
      } else {
        out[4 * plane + cell] = 1;
      }
    }
  });
  for (let cell = 0; cell < plane; cell++) out[6 * plane + cell] = game.food[cell];
  const health = me.alive ? me.health / game.maxHealth : 0;
  out.fill(health, 7 * plane, 8 * plane);
  return out;
}

// Greedy bot: among safe moves, step toward the nearest food.
export function greedyAction(game, k, random = Math.random) {
  const b = game.boardSize;
  const mask = actionMask(game, k);
  const head = game.snakes[k].body[0];
  let best = -1, bestScore = -Infinity;
  DIRECTIONS.forEach(([dr, dc], a) => {
    const r = head[0] + dr, c = head[1] + dc;
    let nearest = 4 * b;
    for (let cell = 0; cell < b * b; cell++) {
      if (game.food[cell]) {
        nearest = Math.min(nearest, Math.abs(r - Math.floor(cell / b)) + Math.abs(c - (cell % b)));
      }
    }
    const noise = random();
    const score = mask[a] ? -nearest + 0.5 * noise : -1e6 + noise;
    if (score > bestScore) { bestScore = score; best = a; }
  });
  return best;
}

// Head-centred window (see build_window_observations in Python): NUM_CHANNELS planes
// cropped around snake k's head, plus a wall plane that is 1 outside the board.
export function windowObservation(game, k, window) {
  const b = game.boardSize, plane = b * b, wPlane = window * window, r = Math.floor(window / 2);
  const full = observation(game, k);
  const out = new Float32Array((NUM_CHANNELS + 1) * wPlane);
  const [hr0, hc0] = game.snakes[k].body[0];
  const hr = Math.min(Math.max(hr0, 0), b - 1), hc = Math.min(Math.max(hc0, 0), b - 1);
  for (let i = 0; i < window; i++) {
    for (let j = 0; j < window; j++) {
      const row = hr - r + i, col = hc - r + j;
      const w = i * window + j;
      if (row < 0 || row >= b || col < 0 || col >= b) {
        out[NUM_CHANNELS * wPlane + w] = 1;
        continue;
      }
      for (let ch = 0; ch < NUM_CHANNELS; ch++) out[ch * wPlane + w] = full[ch * plane + row * b + col];
    }
  }
  return out;
}
