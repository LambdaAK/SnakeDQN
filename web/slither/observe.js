// The network's observation (spec §3), a line-by-line port of tests/slither_reference.observe.
// 32 head-centred sectors x 8 channels, then 7 scalars; checked by tests/test_slither_js_parity.py.
import { bodyRadius, NUM_ACTIONS } from "./rules.js";

export const OBS_CHANNELS = 8;
export const NUM_SCALARS = 7;
export const obsSize = (cfg) => cfg.sectors * OBS_CHANNELS + NUM_SCALARS;

export function observe(g, k, cfg) {
  const K = cfg.sectors, V = cfg.vision, width = 2.0 * Math.PI / K;
  const out = new Float64Array(obsSize(cfg));
  const snakes = g.snakes, me = snakes[k];
  if (!me.alive) return out;
  const [hx, hy] = me.body[0];
  const h = me.heading, c = Math.cos(h), sn = Math.sin(h);
  const radius = snakes.map((s) => bodyRadius(s.mass, cfg));
  const close = (d) => Math.min(Math.max(1.0 - d / V, 0.0), 1.0);
  const sector = (rx, ry) => {
    const x = rx * c + ry * sn, y = ry * c - rx * sn;
    return ((Math.floor((Math.atan2(y, x) + Math.PI + width / 2) / width) % K) + K) % K;
  };

  for (let q = 0; q < K; q++) {
    const th = h + (-Math.PI + q * width);
    const dx = Math.cos(th), dy = Math.sin(th);
    const b = hx * dx + hy * dy;
    const cc = hx * hx + hy * hy - cfg.arena_radius * cfg.arena_radius;
    out[q * 8] = close(-b + Math.sqrt(Math.max(b * b - cc, 0.0)));
  }

  snakes.forEach((other, j) => {
    if (!other.alive) return;
    other.body.forEach(([px, py], idx) => {
      if (j === k && idx < cfg.neck) return;
      const rx = px - hx, ry = py - hy;
      const v = close(Math.sqrt(rx * rx + ry * ry) - radius[j]);
      if (v <= 0) return;
      const q = sector(rx, ry);
      if (j === k) {
        out[q * 8 + 3] = Math.max(out[q * 8 + 3], v);
      } else {
        out[q * 8 + 1] = Math.max(out[q * 8 + 1], v);
        out[q * 8 + 2] = Math.max(out[q * 8 + 2], radius[j] / 15.0);
      }
    });
  });

  const food = new Float64Array(K);
  for (const item of g.food) {
    if (item === null) continue;
    const rx = item[0] - hx, ry = item[1] - hy;
    if (Math.sqrt(rx * rx + ry * ry) < V) food[sector(rx, ry)] += item[2];
  }
  for (let q = 0; q < K; q++) out[q * 8 + 4] = Math.min(food[q] / 10.0, 1.0);

  snakes.forEach((other, j) => {
    if (j === k || !other.alive) return;
    const [ox, oy] = other.body[0];
    const rx = ox - hx, ry = oy - hy;
    const v = close(Math.sqrt(rx * rx + ry * ry) - radius[j]);
    if (v <= 0) return;
    const q = sector(rx, ry);
    if (v > out[q * 8 + 5]) {
      out[q * 8 + 5] = v;
      out[q * 8 + 6] = Math.cos(other.heading - h);
      out[q * 8 + 7] = Math.sin(other.heading - h);
    }
  });

  const base = K * 8;
  out[base] = Math.log(me.mass) / Math.log(cfg.max_mass);
  out[base + 1] = radius[k] / 15.0;
  out[base + 2] = me.boosting ? 1.0 : 0.0;
  out[base + 3] = me.mass >= cfg.boost_min_mass ? 1.0 : 0.0;
  const dist = Math.sqrt(hx * hx + hy * hy);
  out[base + 4] = dist / cfg.arena_radius;
  if (dist > 1e-9) {
    const bx = -hx / dist, by = -hy / dist;
    out[base + 5] = bx * c + by * sn;
    out[base + 6] = by * c - bx * sn;
  } else {
    out[base + 5] = 1.0;
  }
  return out;
}

// Boost actions (odd indices) need boost_min_mass; steering is always allowed.
export function actionMask(g, k, cfg) {
  const canBoost = g.snakes[k].mass >= cfg.boost_min_mass;
  return Array.from({ length: NUM_ACTIONS }, (_, a) => a % 2 === 0 || canBoost);
}
