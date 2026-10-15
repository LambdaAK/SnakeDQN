// Replays Python-generated transitions through rules.js and checks every field.
// Usage: node web/test/parity.mjs fixtures.json
import { readFileSync } from "node:fs";
import { actionMask, observation, step, windowObservation } from "../rules.js";

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
let failures = 0;

function fromFixture(g) {
  const food = new Uint8Array(g.board * g.board);
  for (const cell of g.food) food[cell] = 1;
  return {
    boardSize: g.board, maxHealth: g.maxHealth, minFood: g.minFood, turn: g.turn, over: g.over, winner: -1,
    snakes: g.snakes.map((s) => ({ body: s.body.map((p) => p.slice()), health: s.health, alive: s.alive })),
    food,
  };
}

function fail(i, what, got, want) {
  failures++;
  if (failures <= 10) console.error(`case ${i}: ${what}\n  got  ${JSON.stringify(got)}\n  want ${JSON.stringify(want)}`);
}

cases.forEach((c, i) => {
  const game = fromFixture(c.before);
  c.before.snakes.forEach((_, k) => {
    const mask = actionMask(game, k);
    if (JSON.stringify(mask) !== JSON.stringify(c.mask[k])) fail(i, `mask snake ${k}`, mask, c.mask[k]);
    const obs = c.window ? windowObservation(game, k, c.window) : observation(game, k);
    const want = c.obs[k];
    for (let j = 0; j < want.length; j++) {
      if (Math.abs(obs[j] - want[j]) > 1e-5) { fail(i, `obs snake ${k} index ${j}`, obs[j], want[j]); break; }
    }
  });
  step(game, c.actions, c.u0, c.u1);
  const after = c.after;
  const food = [...game.food.keys()].filter((cell) => game.food[cell]);
  if (JSON.stringify(food) !== JSON.stringify(after.food)) fail(i, "food", food, after.food);
  if (game.turn !== after.turn || game.over !== after.over) fail(i, "turn/over", [game.turn, game.over], [after.turn, after.over]);
  game.snakes.forEach((s, k) => {
    const want = after.snakes[k];
    const got = { body: s.body, health: s.health, alive: s.alive };
    if (JSON.stringify(got) !== JSON.stringify(want)) fail(i, `snake ${k}`, got, want);
  });
});

console.log(`${cases.length} cases, ${failures} failures`);
process.exit(failures ? 1 : 0);
