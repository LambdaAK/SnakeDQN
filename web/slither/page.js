// The standalone Slither page: theme toggle, the arena component, and training results.
import { mountSlitherDemo } from "./slither-demo.js?v=1";

const $ = (id) => document.getElementById(id);

if (new URLSearchParams(location.search).has("embed")) document.body.classList.add("embed");

$("theme").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch {}
});

mountSlitherDemo($("demo"));

(async () => {
  let stats = null;
  try { stats = await (await fetch("stats.json?v=1")).json(); } catch {}
  const evals = stats?.final_eval;
  const tiles = [];
  if (evals?.hunter) {
    tiles.push([`${Math.round(100 * evals.hunter.mass_share)}%`, "of all mass vs 7 hunter bots (fair: 12.5%)"]);
    tiles.push([evals.hunter.kills_per_death.toFixed(1), "kills per death vs hunter bots"]);
  }
  if (evals?.forager) tiles.push([`${Math.round(100 * evals.forager.mass_share)}%`, "of all mass vs 7 forager bots"]);
  $("results").innerHTML = tiles.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
})();
