// Replays Python-generated slither ticks through rules.js and observe.js and checks every field.
// Usage: node web/slither/test/parity.mjs fixtures.json
import { readFileSync } from "node:fs";
import { CONFIG, stepTick } from "../rules.js";
import { observe } from "../observe.js";

const { defaults, groups } = JSON.parse(readFileSync(process.argv[2], "utf8"));
let failures = 0;
let total = 0;
const near = (a, b, tol) => Math.abs(a - b) <= tol;

function fail(where, what, got, want) {
  failures++;
  if (failures <= 10) console.error(`${where}: ${what}\n  got  ${JSON.stringify(got)}\n  want ${JSON.stringify(want)}`);
}

// CONFIG must equal the Python defaults, key for key.
for (const key of new Set([...Object.keys(CONFIG), ...Object.keys(defaults)])) {
  if (CONFIG[key] !== defaults[key]) fail("CONFIG", key, CONFIG[key], defaults[key]);
}

function compareGame(where, got, want) {
  if (got.tick !== want.tick) fail(where, "tick", got.tick, want.tick);
  got.snakes.forEach((s, k) => {
    const w = want.snakes[k];
    for (const key of ["alive", "enabled", "boosting", "boost_ticks", "respawn"]) {
      if (s[key] !== w[key]) fail(where, `snake ${k} ${key}`, s[key], w[key]);
    }
    for (const key of ["heading", "mass"]) {
      if (!near(s[key], w[key], 1e-9)) fail(where, `snake ${k} ${key}`, s[key], w[key]);
    }
    if (s.body.length !== w.body.length) {
      fail(where, `snake ${k} body length`, s.body.length, w.body.length);
      return;
    }
    for (let p = 0; p < s.body.length; p++) {
      if (!near(s.body[p][0], w.body[p][0], 1e-9) || !near(s.body[p][1], w.body[p][1], 1e-9)) {
        fail(where, `snake ${k} point ${p}`, s.body[p], w.body[p]);
        break;
      }
    }
  });
  got.food.forEach((item, f) => {
    const w = want.food[f];
    if ((item === null) !== (w === null)) return fail(where, `food ${f}`, item, w);
    if (item !== null && (!near(item[0], w[0], 1e-9) || !near(item[1], w[1], 1e-9)
        || !near(item[2], w[2], 1e-9) || item[3] !== w[3])) fail(where, `food ${f}`, item, w);
  });
}

groups.forEach((group, g) => {
  const cfg = group.config;
  group.cases.forEach((c, i) => {
    const where = `group ${g} case ${i}`;
    total++;
    const game = structuredClone(c.before);
    game.snakes.forEach((_, k) => {
      const obs = observe(game, k, cfg);
      const want = c.obs[k];
      for (let j = 0; j < want.length; j++) {
        if (!near(obs[j], want[j], 1e-6)) { fail(where, `obs snake ${k} index ${j}`, obs[j], want[j]); break; }
      }
    });
    const ev = stepTick(game, c.turns, c.boosts, c.u, cfg);
    for (const key of ["died", "killer", "kills"]) {
      if (JSON.stringify(ev[key]) !== JSON.stringify(c.events[key])) fail(where, key, ev[key], c.events[key]);
    }
    ev.eaten.forEach((v, k) => { if (!near(v, c.events.eaten[k], 1e-9)) fail(where, `eaten ${k}`, v, c.events.eaten[k]); });
    compareGame(where, game, c.after);
  });
});

console.log(`${total} cases, ${failures} failures`);
process.exit(failures ? 1 : 0);
