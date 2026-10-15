// The standalone demo page: theme toggle and the game component.
import { mountSnakeDemo } from "./snake-demo.js?v=17";

const $ = (id) => document.getElementById(id);

if (new URLSearchParams(location.search).has("embed")) document.body.classList.add("embed");

$("theme").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch {}
});

mountSnakeDemo($("demo"));
