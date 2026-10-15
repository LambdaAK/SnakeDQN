// Slither rules for the browser: a line-by-line port of tests/slither_reference.py
// (spec §2). tests/test_slither_js_parity.py replays Python-generated ticks through this
// file, so keep every expression in the same form and order as the Python.
//
// A game is a plain object in the snapshot format of snake_rl.slither.rules.snapshot:
//   { slots, tick, snakes: [{ body: [[x, y], ...] head first, heading, mass, alive,
//     enabled, boosting, boost_ticks, respawn }], food: [null | [x, y, value, natural]] }

export const CONFIG = Object.freeze({
  arena_radius: 300, tick_hz: 30, ticks_per_decision: 3, max_turn: 0.15,
  start_mass: 10, max_mass: 500, base_length: 20, length_per_mass: 2,
  base_radius: 2, radius_per_sqrt_mass: 0.5, body_capacity: 1024,
  boost_min_mass: 15, boost_cost: 0.2, boost_drop_every: 5, boost_drop_value: 0.8,
  eat_margin: 3, food_capacity: 1024, natural_food_target: 200, natural_food_per_tick: 2,
  natural_food_value: 1, food_spawn_margin: 10, death_drop_fraction: 0.8,
  death_drop_max_pellets: 64, death_drop_jitter: 2, respawn_delay: 30,
  respawn_candidates: 16, respawn_margin: 60, respawn_clearance: 40,
  vision: 60, sectors: 32, neck: 10,
});

export const NUM_ACTIONS = 18;

export const targetLength = (mass, cfg) => Math.floor(cfg.base_length + cfg.length_per_mass * mass);
export const bodyRadius = (mass, cfg) => cfg.base_radius + cfg.radius_per_sqrt_mass * Math.sqrt(mass);

export function uLayout(slots, cfg) {
  const drops = slots * (2 * cfg.respawn_candidates + 1);
  const food = drops + slots * cfg.death_drop_max_pellets * 2;
  return { respawn: 0, drops, food, size: food + 2 * cfg.natural_food_per_tick };
}

export function randomU(slots, cfg, random = Math.random) {
  const u = new Float64Array(uLayout(slots, cfg).size);
  for (let i = 0; i < u.length; i++) u[i] = random();
  return u;
}

// action = steer * 2 + boost; steering bin b turns max_turn * (b - 4) / 4 per tick.
export function decodeAction(action, cfg) {
  const steer = Math.floor(action / 2);
  return { turn: cfg.max_turn * (steer - 4) / 4, boost: action % 2 === 1 };
}

export function stepTick(g, turns, boosts, u, cfg) {
  const snakes = g.snakes, food = g.food, count = snakes.length;
  const lay = uLayout(count, cfg);
  const alive = snakes.map((s) => s.alive);

  // 1. steer
  snakes.forEach((s, k) => {
    if (alive[k]) s.heading = s.heading + Math.min(Math.max(turns[k], -cfg.max_turn), cfg.max_turn);
  });

  // 2. boost, move, trim
  const boostDrops = [];
  snakes.forEach((s, k) => {
    const b = alive[k] && Boolean(boosts[k]) && s.mass >= cfg.boost_min_mass;
    if (b) s.mass = s.mass - cfg.boost_cost;
    s.boost_ticks = b ? s.boost_ticks + 1 : 0;
    s.boosting = b;
    if (!alive[k]) return;
    for (let step = 0; step < (b ? 2 : 1); step++) {
      const [hx, hy] = s.body[0];
      s.body.unshift([hx + Math.cos(s.heading), hy + Math.sin(s.heading)]);
    }
    s.body.length = Math.min(s.body.length, targetLength(s.mass, cfg));
    if (b && s.boost_ticks % cfg.boost_drop_every === 0) {
      const [tx, ty] = s.body[s.body.length - 1];
      boostDrops.push([tx, ty, cfg.boost_drop_value, false]);
    }
  });

  // 3. deaths
  const radius = snakes.map((s) => bodyRadius(s.mass, cfg));
  const died = Array(count).fill(false), killer = Array(count).fill(-1), kills = Array(count).fill(0);
  for (let i = 0; i < count; i++) {
    if (!alive[i]) continue;
    const [hx, hy] = snakes[i].body[0];
    const wall = Math.sqrt(hx * hx + hy * hy) + radius[i] > cfg.arena_radius;
    let headOn = false, first = -1;
    for (let j = 0; j < count; j++) {
      if (j === i || !alive[j]) continue;
      const t = radius[i] + radius[j];
      const [ox, oy] = snakes[j].body[0];
      let dx = hx - ox, dy = hy - oy;
      if (dx * dx + dy * dy < t * t) headOn = true;
      if (first < 0) {
        const body = snakes[j].body;
        for (let p = 1; p < body.length; p++) {
          dx = body[p][0] - hx;
          dy = body[p][1] - hy;
          if (dx * dx + dy * dy < t * t) { first = j; break; }
        }
      }
    }
    died[i] = wall || headOn || first >= 0;
    killer[i] = first >= 0 && !headOn ? first : -1;
  }
  for (const j of killer) if (j >= 0) kills[j] += 1;

  // 4. eat
  const eaten = Array(count).fill(0);
  for (let f = 0; f < food.length; f++) {
    const item = food[f];
    if (item === null) continue;
    let best = -1, bestD2 = Infinity;
    for (let k = 0; k < count; k++) {
      if (!alive[k] || died[k]) continue;
      const [hx, hy] = snakes[k].body[0];
      const dx = item[0] - hx, dy = item[1] - hy;
      const d2 = dx * dx + dy * dy;
      const reach = radius[k] + cfg.eat_margin;
      if (d2 < reach * reach && d2 < bestD2) { best = k; bestD2 = d2; }
    }
    if (best >= 0) {
      eaten[best] += item[2];
      food[f] = null;
    }
  }
  snakes.forEach((s, k) => { s.mass = Math.min(s.mass + eaten[k], cfg.max_mass); });

  // 5. death drops
  const deathDrops = [];
  const most = cfg.death_drop_max_pellets, jitter = 2.0 * cfg.death_drop_jitter;
  snakes.forEach((s, k) => {
    if (!died[k]) return;
    const length = s.body.length, m = s.mass;
    const n = Math.max(Math.min(Math.ceil(m / 2), most), 1);
    const value = cfg.death_drop_fraction * m / n;
    for (let q = 0; q < n; q++) {
      const [px, py] = s.body[Math.floor((q * length) / n)];
      const base = lay.drops + (k * most + q) * 2;
      deathDrops.push([px + (u[base] - 0.5) * jitter, py + (u[base + 1] - 0.5) * jitter, value, false]);
    }
    Object.assign(s, { alive: false, body: [], respawn: cfg.respawn_delay, boosting: false, boost_ticks: 0 });
  });

  // 6. respawn slots that were dead when the tick started
  snakes.forEach((s, k) => {
    if (!s.enabled || alive[k]) return;
    s.respawn -= 1;
    if (s.respawn <= 0) respawnSlot(g, k, u, cfg);
  });

  // 7. new pellets: boost drops, death drops, natural food, into the lowest free slots
  let natural = 0;
  for (const item of food) if (item !== null && item[3]) natural++;
  const need = Math.max(0, Math.min(cfg.natural_food_target - natural, cfg.natural_food_per_tick));
  const fresh = [...boostDrops, ...deathDrops];
  for (let q = 0; q < need; q++) {
    const rr = (cfg.arena_radius - cfg.food_spawn_margin) * Math.sqrt(u[lay.food + 2 * q]);
    const th = 2.0 * Math.PI * u[lay.food + 2 * q + 1];
    fresh.push([rr * Math.cos(th), rr * Math.sin(th), cfg.natural_food_value, true]);
  }
  let next = 0;
  for (let f = 0; f < food.length && next < fresh.length; f++) if (food[f] === null) food[f] = fresh[next++];
  g.tick += 1;
  return { died, killer, kills, eaten };
}

// Place slot k at the first candidate clear of every live head, else the farthest one.
export function respawnSlot(g, k, u, cfg) {
  const snakes = g.snakes, c = cfg.respawn_candidates;
  const base = uLayout(snakes.length, cfg).respawn + k * (2 * c + 1);
  const clearance2 = cfg.respawn_clearance * cfg.respawn_clearance;
  let pick = -1, far = -1, farD2 = -Infinity;
  const candidates = [];
  for (let q = 0; q < c; q++) {
    const rr = (cfg.arena_radius - cfg.respawn_margin) * Math.sqrt(u[base + 2 * q]);
    const th = 2.0 * Math.PI * u[base + 2 * q + 1];
    const x = rr * Math.cos(th), y = rr * Math.sin(th);
    let nearest = Infinity;
    for (const other of snakes) {
      if (!other.alive) continue;
      const dx = x - other.body[0][0], dy = y - other.body[0][1];
      nearest = Math.min(nearest, dx * dx + dy * dy);
    }
    candidates.push([x, y]);
    if (pick < 0 && nearest >= clearance2) pick = q;
    if (nearest > farD2) { far = q; farD2 = nearest; }
  }
  if (pick < 0) pick = far;
  const [x, y] = candidates[pick];
  const h = 2.0 * Math.PI * u[base + 2 * c];
  const cosH = Math.cos(h), sinH = Math.sin(h);
  const body = [];
  for (let i = 0; i < targetLength(cfg.start_mass, cfg); i++) body.push([x - i * cosH, y - i * sinH]);
  Object.assign(snakes[k], { body, heading: h, mass: cfg.start_mass, alive: true, boosting: false, boost_ticks: 0, respawn: 0 });
}

// A fresh arena: the first `enabledCount` slots spawn one by one, natural food is filled.
export function newArena(slots, enabledCount, cfg = CONFIG, random = Math.random) {
  const g = { slots, tick: 0, snakes: [], food: Array(cfg.food_capacity).fill(null) };
  for (let k = 0; k < slots; k++) {
    g.snakes.push({ body: [], heading: 0, mass: cfg.start_mass, alive: false, enabled: k < enabledCount,
                    boosting: false, boost_ticks: 0, respawn: 0 });
  }
  const u = randomU(slots, cfg, random);
  for (let k = 0; k < enabledCount; k++) respawnSlot(g, k, u, cfg);
  for (let f = 0; f < cfg.natural_food_target; f++) {
    const rr = (cfg.arena_radius - cfg.food_spawn_margin) * Math.sqrt(random());
    const th = 2.0 * Math.PI * random();
    g.food[f] = [rr * Math.cos(th), rr * Math.sin(th), cfg.natural_food_value, true];
  }
  return g;
}
