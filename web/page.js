// The standalone demo page: theme toggle, the game component, and training results.
import { mountSnakeDemo } from "./snake-demo.js?v=15";

const $ = (id) => document.getElementById(id);

if (new URLSearchParams(location.search).has("embed")) document.body.classList.add("embed");

$("theme").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch {}
});

mountSnakeDemo($("demo"));

async function loadStats() {
  const get = async (url) => { try { return await (await fetch(url)).json(); } catch { return null; } };
  const [duel, arena] = await Promise.all([get("stats.json?v=2"), get("stats_arena.json?v=2")]);
  const tiles = [];
  if (duel?.final_eval) {
    tiles.push([`${Math.round(100 * duel.final_eval.win_vs_flood_fill)}%`, "wins vs the best scripted bot"]);
    tiles.push([`${Math.round(100 * duel.final_eval.win_vs_greedy)}%`, "wins vs a greedy bot"]);
  }
  if (arena?.final_eval) {
    tiles.push([`${Math.round(100 * arena.final_eval.win_vs_flood_fill)}%`, "arena wins vs 5 bots (fair: 17%)"]);
  }
  $("results").innerHTML = tiles.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  if (duel?.curve) drawChart(duel.curve);
}

function drawChart(points) {
  const W = 720, H = 220, L = 40, R = 8, T = 10, B = 30;
  const maxX = points[points.length - 1].env_steps;
  const x = (v) => L + ((W - L - R) * v) / maxX;
  const y = (v) => T + (H - T - B) * (1 - v);
  const line = (key) => points.map((p, i) => `${i ? "L" : "M"}${x(p.env_steps).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");
  let grid = "";
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    grid += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/>`;
    grid += `<text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  }
  for (let k = 0; k <= 5; k++) {
    const v = (maxX * k) / 5;
    grid += `<text x="${x(v)}" y="${H - 10}" text-anchor="middle">${(v / 1e6).toFixed(0)}M</text>`;
  }
  $("chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Win rate over training">
      ${grid}
      <path d="${line("win_vs_greedy")}" fill="none" stroke="var(--color-fg-faint)" stroke-width="2"/>
      <path d="${line("win_vs_flood_fill")}" fill="none" stroke="var(--color-accent)" stroke-width="2"/>
    </svg>
    <div class="legend"><span>Duel win rate during training</span>
      <span style="--c:var(--color-accent)">vs best scripted bot</span>
      <span style="--c:var(--color-fg-faint)">vs greedy bot</span></div>`;
}

loadStats();
