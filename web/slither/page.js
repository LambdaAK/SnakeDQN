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
  const pct = (v) => (Number.isFinite(v) ? `${Math.round(100 * v)}%` : null);
  const hunterShare = pct(evals?.hunter?.mass_share), foragerShare = pct(evals?.forager?.mass_share);
  const kpd = evals?.hunter?.kills_per_death;
  if (hunterShare) tiles.push([hunterShare, "of all mass vs 7 hunter bots (fair: 12.5%)"]);
  if (Number.isFinite(kpd)) tiles.push([kpd.toFixed(1), "kills per death vs hunter bots"]);
  if (foragerShare) tiles.push([foragerShare, "of all mass vs 7 forager bots"]);
  $("results").innerHTML = tiles.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
})();
