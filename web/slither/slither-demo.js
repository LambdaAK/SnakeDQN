// Slither self-play demo as a mountable component.
//
//   import { mountSlitherDemo } from "./slither-demo.js";
//   mountSlitherDemo(document.querySelector("#slither"), { base: "/snake/slither/", autoLoad: false });
//
// Everything runs in the browser: the rules (rules.js, a checked port of the Python
// engine), the observation (observe.js), the trained network (ONNX Runtime Web) and the
// rendering (render.js). Styles live in slither-demo.css, scoped to .slither-demo; colours
// come from the host page's --color-* tokens when it has them.

import { CONFIG, decodeAction, newArena, randomU, respawnSlot, stepTick } from "./rules.js";
import { actionMask, observe, obsSize } from "./observe.js";
import { COLORS, NAMES, drawArena } from "./render.js?v=3";

const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL = { file: "model.onnx", mb: 0.6 };
const SLOTS = 12;
const SNAKES = 10;        // enabled slots; in Play mode slot 0 is you
const HUMAN = 0;
const MAX_CATCH_UP = 8;   // physics ticks per advance() call, times the speed
const SPEEDS = [1, 2, 3, 5];
const LEAD_TICKS = 2;     // the simulation stays this many ticks ahead of what is drawn

const TEMPLATE = `
  <div class="sl-tabs" role="tablist">
    <button type="button" role="tab" data-mode="watch">Watch</button>
    <button type="button" role="tab" data-mode="play">Play</button>
  </div>
  <div class="sl-game">
    <div class="sl-board">
      <canvas class="sl-arena" data-sl="arena" aria-label="Slither arena"></canvas>
      <div class="sl-overlay" data-sl="overlay" hidden></div>
      <button type="button" class="sl-boost" data-sl="boost" hidden>Boost</button>
    </div>
    <div class="sl-side">
      <div><p class="sl-lbl">Leaderboard</p><ol class="sl-leaders" data-sl="leaders"></ol></div>
      <div class="sl-pills">
        <button type="button" data-sl="restart">Restart</button>
        <button type="button" data-sl="pause">Pause</button>
        <button type="button" data-sl="speed">Speed 1×</button>
      </div>
      <p class="sl-meta" data-sl="hint"></p>
    </div>
  </div>`;

let ortPromise = null;
function loadOrt(src) {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  ortPromise ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = () => resolve(globalThis.ort);
    script.onerror = () => { ortPromise = null; script.remove(); reject(new Error("couldn't load ONNX Runtime")); };
    document.head.appendChild(script);
  });
  return ortPromise;
}

/**
 * Build the demo inside `root`.
 * @param {HTMLElement} root
 * @param {{ base?: string, autoLoad?: boolean, ortCdn?: string, mode?: "watch" | "play" }} [options]
 *   base: URL prefix of model.onnx (default: same folder as the page).
 *   autoLoad: download the network immediately (default) or wait for a click.
 */
export function mountSlitherDemo(root, options = {}) {
  const cfg = CONFIG;
  const base = options.base ?? "";
  const ortCdn = options.ortCdn ?? ORT_CDN;
  const tickMs = 1000 / cfg.tick_hz;
  const coarse = matchMedia("(pointer: coarse)").matches;
  root.classList.add("slither-demo");
  root.innerHTML = TEMPLATE;

  const $ = (key) => root.querySelector(`[data-sl="${key}"]`);
  const canvas = $("arena");
  const ctx = canvas.getContext("2d");
  const overlay = $("overlay");
  const tabs = [...root.querySelectorAll(".sl-tabs button")];

  let mode = options.mode ?? "watch";
  let session = null;
  let started = false, paused = false, visible = true, busy = false;
  let theme = {}, dpr = 1, camera = null;
  let game, plan, follow;
  let snaps = [], simTick = 0, rt = 0; // recent per-tick states; ticks simulated; render time (ms of game time)
  let shown = null, lastPanel = "", speed = 1;
  let last = performance.now(), frameNo = 0;
  let pointer = null, boostHeld = false;
  let gen = 0, failed = false;
  let you = null; // { kills, ticks } for your current life in Play mode

  // What the renderer needs from one snake, copied so later ticks can't change it.
  const snapSnake = (s) => ({ alive: s.alive, body: s.alive ? s.body.slice() : [], mass: s.mass,
    heading: s.heading, boosting: s.boosting });
  const snapshot = () => ({ t: simTick * tickMs, snakes: game.snakes.map(snapSnake) });
  const followed = () => (mode === "play" && game.snakes[HUMAN].alive ? HUMAN : follow);
  const name = (k) => (mode === "play" && k === HUMAN ? "You" : NAMES[k % NAMES.length]);

  function biggest() {
    let best = -1;
    game.snakes.forEach((s, k) => {
      if (!s.alive) return;
      if (best < 0 || s.mass > game.snakes[best].mass) best = k;
    });
    return Math.max(best, 0);
  }

  function newGame() {
    gen++;
    boostHeld = false;
    game = newArena(SLOTS, SNAKES, cfg);
    simTick = 0;
    rt = 0;
    snaps = [snapshot()];
    plan = Array.from({ length: SLOTS }, () => ({ turn: 0, boost: false }));
    follow = mode === "play" ? HUMAN : biggest();
    you = mode === "play" ? { kills: 0, ticks: 0 } : null;
    camera = null;
    if (session) hideOverlay();
  }

  // ---------- simulation ----------

  async function decide() {
    const ai = [];
    game.snakes.forEach((s, k) => { if (s.alive && !(mode === "play" && k === HUMAN)) ai.push(k); });
    if (!ai.length || !session) return; // nobody to steer: no inference call
    const D = obsSize(cfg);
    const data = new Float32Array(ai.length * D);
    ai.forEach((k, r) => data.set(observe(game, k, cfg), r * D));
    const g = gen;
    const out = await session.run({ obs: new globalThis.ort.Tensor("float32", data, [ai.length, D]) });
    if (g !== gen) return; // the game was restarted while the network was thinking
    const logits = out.logits.data;
    ai.forEach((k, r) => {
      const probs = maskedSoftmax(Array.from(logits.subarray(r * 18, r * 18 + 18)), actionMask(game, k, cfg));
      const action = sample(probs);
      plan[k] = decodeAction(action, cfg);
    });
  }

  function humanTurn() {
    const s = game.snakes[HUMAN];
    if (!pointer || !camera) return 0;
    const [hx, hy] = s.body[0];
    const wx = camera.x + (pointer.x - canvas.width / 2) / camera.scale;
    const wy = camera.y + (pointer.y - canvas.height / 2) / camera.scale;
    if (Math.hypot(wx - hx, wy - hy) < 2) return 0;
    let d = Math.atan2(wy - hy, wx - hx) - s.heading;
    d = Math.atan2(Math.sin(d), Math.cos(d)); // wrap to (-pi, pi]
    return Math.max(-cfg.max_turn, Math.min(cfg.max_turn, d));
  }

  async function tick() {
    const g = gen;
    if (game.tick % cfg.ticks_per_decision === 0) await decide();
    if (g !== gen) return false;
    const turns = plan.map((p) => p.turn), boosts = plan.map((p) => p.boost);
    if (mode === "play" && game.snakes[HUMAN].alive) {
      turns[HUMAN] = humanTurn();
      boosts[HUMAN] = boostHeld;
    }
    const ev = stepTick(game, turns, boosts, randomU(SLOTS, cfg), cfg);
    if (mode === "play" && you) {
      if (game.snakes[HUMAN].alive || ev.died[HUMAN]) {
        you.ticks += 1;
        you.kills += ev.kills[HUMAN];
      }
      if (ev.died[HUMAN]) onHumanDeath();
    }
    simTick++;
    snaps.push(snapshot());
    if (!game.snakes[follow]?.alive) follow = biggest();
    return true;
  }

  function onHumanDeath() {
    const s = game.snakes[HUMAN];
    s.enabled = false; // you respawn only when you ask to
    const secs = Math.round(you.ticks / cfg.tick_hz);
    showOverlay(`You reached mass ${Math.round(s.mass)}`, `${you.kills} kill${you.kills === 1 ? "" : "s"} · ${secs}s alive`,
      { label: "Play again", run: respawnYou });
  }

  function respawnYou() {
    const s = game.snakes[HUMAN];
    s.enabled = true;
    respawnSlot(game, HUMAN, randomU(SLOTS, cfg), cfg);
    snaps[snaps.length - 1].snakes[HUMAN] = snapSnake(s);
    you = { kills: 0, ticks: 0 };
    hideOverlay();
  }

  // Run physics until the simulation is LEAD_TICKS ahead of the render clock. Inference makes
  // some ticks slow, but the render clock only reads finished states, so drawing stays smooth.
  async function advance() {
    let n = 0;
    while (simTick * tickMs < rt + LEAD_TICKS * tickMs && n < MAX_CATCH_UP * speed) {
      if (!(await tick())) break;
      n++;
    }
  }

  // The arena at render time `rt`: the two bracketing tick states, blended. Head and tail
  // are interpolated (the body in between moves rigidly one point per tick).
  function interpolated() {
    while (snaps.length > 2 && snaps[1].t <= rt) snaps.shift();
    const A = snaps[0], B = snaps[snaps.length > 1 ? 1 : 0];
    const f = B.t > A.t ? Math.max(0, Math.min(1, (rt - A.t) / (B.t - A.t))) : 1;
    const lerp = (p, q) => [p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f];
    return { food: game.food, snakes: B.snakes.map((sb, k) => {
      const sa = A.snakes[k];
      if (!sb.alive || !sa.alive || sa === sb || sb.body.length < 2 || sa.body.length < 2) return sb;
      const body = sb.body.slice();
      body[0] = lerp(sa.body[0], sb.body[0]);
      body[body.length - 1] = lerp(sa.body[sa.body.length - 1], sb.body[body.length - 1]);
      return { ...sb, body };
    }) };
  }

  function frame(now) {
    requestAnimationFrame(frame); // first, so a drawing error can never stop the loop
    const dt = Math.min(now - last, 250);
    last = now;
    const running = started && session && !paused && visible && !failed;
    if (running) {
      rt += dt * speed;
      if (!busy) {
        busy = true;
        advance().catch((err) => {
          failed = true;
          showOverlay("Something went wrong", String(err.message ?? err),
            { label: "Retry", run: () => { failed = false; hideOverlay(); newGame(); } });
        })
          .finally(() => { busy = false; });
      }
    }
    // never draw past the newest finished state (a slow device just plays slower)
    rt = Math.min(rt, snaps[snaps.length - 1].t);
    shown = interpolated();
    camera = drawArena(ctx, shown, cfg, {
      dt, follow: followed(), human: mode === "play" ? HUMAN : -1, theme, dpr,
    }, camera);
    if (++frameNo % 6 === 0) updatePanel();
  }

  // ---------- panel ----------

  function updatePanel() {
    const top = game.snakes.map((s, k) => [k, s]).filter(([, s]) => s.alive)
      .sort((a, b) => b[1].mass - a[1].mass).slice(0, 5);
    const html = top.map(([k, s]) => `<li data-k="${k}" class="${k === followed() ? "sl-on" : ""}">`
      + `<span class="sl-dot" style="background:${COLORS[k % COLORS.length]}"></span><span>${name(k)}</span>`
      + `<span class="sl-mass">${Math.round(s.mass)}</span></li>`).join("");
    if (html === lastPanel) return; // only touch the DOM when something changed
    lastPanel = html;
    $("leaders").innerHTML = html;
  }

  // ---------- overlay ----------

  function showOverlay(title, sub = "", action = null) {
    overlay.hidden = false;
    overlay.innerHTML = `<div>${title}${sub ? `<small>${sub}</small>` : ""}</div>`;
    if (action) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sl-start";
      btn.textContent = action.label;
      btn.addEventListener("click", action.run);
      overlay.firstElementChild.appendChild(btn);
    }
  }
  function hideOverlay() { overlay.hidden = true; overlay.innerHTML = ""; }

  // ---------- network and modes ----------

  async function loadModel() {
    if (session) return true;
    showOverlay("Loading…");
    try {
      const ort = await loadOrt(`${ortCdn}ort.min.js`);
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = ortCdn;
      session = await ort.InferenceSession.create(base + MODEL.file);
      hideOverlay();
      return true;
    } catch (err) {
      showOverlay("Couldn't load the network", String(err.message ?? err), { label: "Retry", run: () => start() });
      return false;
    }
  }

  function selectMode(next) {
    mode = next;
    boostHeld = false;
    tabs.forEach((b) => b.classList.toggle("sl-active", b.dataset.mode === mode));
    root.classList.toggle("sl-playing", mode === "play");
    $("boost").hidden = !(mode === "play" && coarse);
    $("hint").textContent = mode === "play"
      ? (coarse ? "Touch to steer, hold Boost to speed up." : "Steer with the mouse. Hold the button or Space to boost.")
      : "Click a snake to follow it.";
    newGame();
  }

  async function start() {
    started = true;
    if (await loadModel()) selectMode(mode);
  }

  // ---------- input and wiring ----------

  const toCanvas = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
  };
  function pickSnake(p) {
    let best = -1, bestD = 40 * dpr;
    (shown ?? game).snakes.forEach((s, k) => {
      if (!s.alive || !camera) return;
      const x = canvas.width / 2 + (s.body[0][0] - camera.x) * camera.scale;
      const y = canvas.height / 2 + (s.body[0][1] - camera.y) * camera.scale;
      const d = Math.hypot(x - p.x, y - p.y);
      if (d < bestD) { best = k; bestD = d; }
    });
    if (best >= 0) follow = best;
  }

  tabs.forEach((btn) => btn.addEventListener("click", () => { btn.blur(); if (started) selectMode(btn.dataset.mode); }));
  canvas.addEventListener("pointermove", (e) => { pointer = toCanvas(e); });
  canvas.addEventListener("pointerdown", (e) => {
    pointer = toCanvas(e);
    if (mode === "play" && e.pointerType === "mouse" && e.button === 0) boostHeld = true;
    if (mode === "watch") pickSnake(pointer);
  });
  window.addEventListener("pointerup", () => { boostHeld = false; });
  $("boost").addEventListener("pointerdown", (e) => { e.preventDefault(); boostHeld = true; });
  $("boost").addEventListener("pointerleave", () => { boostHeld = false; });
  $("leaders").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-k]");
    if (li && mode === "watch") follow = Number(li.dataset.k);
  });
  // Space boosts only while you're playing and the arena is on screen.
  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || mode !== "play" || !visible || !root.isConnected) return;
    if (e.target instanceof HTMLElement && e.target.closest("input, textarea, select, button, [contenteditable]")) return;
    e.preventDefault();
    boostHeld = true;
  });
  window.addEventListener("blur", () => { boostHeld = false; });
  document.addEventListener("visibilitychange", () => { if (document.hidden) boostHeld = false; });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") boostHeld = false; });
  $("restart").addEventListener("click", () => { if (started) newGame(); });
  $("pause").addEventListener("click", () => {
    paused = !paused;
    $("pause").textContent = paused ? "Resume" : "Pause";
  });
  $("speed").addEventListener("click", () => {
    speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
    $("speed").textContent = `Speed ${speed}×`;
  });

  function readTheme() {
    const css = getComputedStyle(root);
    const v = (n) => css.getPropertyValue(n).trim();
    const board = v("--sl-board"), fg = v("--sl-fg");
    theme = { board, fg, outside: mix(board, fg, 0.06), grid: mix(board, v("--sl-line"), 0.7),
              wall: v("--sl-wall") };
  }
  function resizeCanvas() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
  }

  new ResizeObserver(resizeCanvas).observe(canvas);
  // Follow the host page's light/dark switch (a class or data attribute on <html>).
  new MutationObserver(readTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", readTheme);
  // Pause while scrolled out of view.
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(root);

  readTheme();
  resizeCanvas();
  tabs.forEach((b) => b.classList.toggle("sl-active", b.dataset.mode === mode));
  newGame();
  requestAnimationFrame(frame);
  if (options.autoLoad === false) {
    showOverlay("Ten snakes, all trained by self-play", "", { label: `▶ Load the AI (${MODEL.mb} MB)`, run: () => start() });
  } else {
    start();
  }
  return { selectMode };
}

// ---------- helpers ----------

function sample(probs) {
  let u = Math.random();
  for (let a = 0; a < probs.length; a++) {
    u -= probs[a];
    if (u <= 0) return a;
  }
  return probs.length - 1;
}

function maskedSoftmax(logits, mask) {
  const allowed = mask.some(Boolean) ? mask : mask.map(() => true);
  const max = Math.max(...logits.filter((_, a) => allowed[a]));
  const exp = logits.map((l, a) => (allowed[a] ? Math.exp(l - max) : 0));
  const sum = exp.reduce((x, y) => x + y, 0);
  return exp.map((e) => e / sum);
}

// Blend colour a toward b by t (0..1). Accepts #rrggbb or rgb(...) strings.
function mix(a, b, t) {
  const parse = (c) => (c.startsWith("#")
    ? [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16))
    : (c.match(/\d+(\.\d+)?/g) || [0, 0, 0]).slice(0, 3).map(Number));
  const [x, y] = [parse(a), parse(b)];
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
}
