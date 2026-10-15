// The standalone Slither page: theme toggle and the arena component.
import { mountSlitherDemo } from "./slither-demo.js?v=3";

const $ = (id) => document.getElementById(id);

if (new URLSearchParams(location.search).has("embed")) document.body.classList.add("embed");

$("theme").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch {}
});

mountSlitherDemo($("demo"));
