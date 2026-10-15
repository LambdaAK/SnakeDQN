// Snake self-play demo as a mountable component.
//
//   import { mountSnakeDemo } from "./snake-demo.js";
//   mountSnakeDemo(document.querySelector("#demo"), { base: "/snake/", autoLoad: false });
//
// Everything runs in the browser: the rules (rules.js), the trained networks (ONNX
// Runtime Web) and the rendering. Styles live in snake-demo.css, scoped to .snake-demo;
// colours come from the host page's --color-* tokens when it has them.

import { actionMask, DIRECTIONS, greedyAction, newGame, NUM_CHANNELS, observation, step, windowObservation } from "./rules.js";

const ORT_VERSION = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const COLORS = ["#f59e0b", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#facc15", "#fb7185", "#22d3ee"];
const COLOR_NAMES = ["Orange", "Blue", "Violet", "Green", "Pink", "Yellow", "Coral", "Cyan"];
const KEYS = { ArrowUp: 0, KeyW: 0, ArrowDown: 1, KeyS: 1, ArrowLeft: 2, KeyA: 2, ArrowRight: 3, KeyD: 3 };
const OPPOSITE = [1, 0, 3, 2];
const SPEEDS = [[1, 180], [2, 100], [0.5, 320]]; // [label multiplier, ms per move]
// Two networks: the 1v1 duel network sees the whole 11x11 board; the arena network sees a
// 17x17 window around its own head, so it plays on a large board with many snakes.
const MODELS = {
  duel: { file: "model.onnx", window: 0, mb: 8 },
  arena: { file: "model_arena.onnx", window: 17, mb: 10 },
};
const MODES = {
  watch: { label: "AI vs AI", board: 11, model: "duel", names: ["Orange AI", "Blue AI"], controllers: ["ai", "ai"] },
  play: { label: "You vs AI", board: 11, model: "duel", names: ["AI", "You"], controllers: ["ai", "human"] },
  bot: { label: "AI vs bot", board: 11, model: "duel", names: ["AI", "Greedy bot"], controllers: ["ai", "greedy"] },
  arena: { label: "Arena", board: 19, model: "arena" },
};

const TEMPLATE = `
  <div class="sd-tabs" role="tablist">
    ${Object.entries(MODES).map(([key, m]) => `<button type="button" role="tab" data-mode="${key}">${m.label}</button>`).join("")}
  </div>
  <div class="sd-game">
    <div class="sd-board">
      <canvas aria-label="Snake game board"></canvas>
      <div class="sd-overlay" data-sd="overlay" hidden></div>
    </div>
    <div class="sd-side">
      <div><p class="sd-lbl">Snakes</p><ul class="sd-snakes" data-sd="snakes"></ul></div>
      <div class="sd-pills">
        <button type="button" data-sd="restart">Restart</button>
        <button type="button" data-sd="pause">Pause</button>
        <button type="button" data-sd="speed">Speed 1×</button>
      </div>
      <div class="sd-pills sd-stepper" data-sd="arena-controls" hidden>
        <button type="button" data-sd="fewer" aria-label="Fewer snakes">−</button>
        <span data-sd="arena-count">6 snakes</span>
        <button type="button" data-sd="more" aria-label="More snakes">+</button>
      </div>
      <p class="sd-meta" data-sd="tally"></p>
    </div>
  </div>
  <div class="sd-dpad" data-sd="dpad" hidden>
    <button type="button" data-dir="0" aria-label="Up">↑</button>
    <button type="button" data-dir="2" aria-label="Left">←</button>
    <button type="button" data-dir="1" aria-label="Down">↓</button>
    <button type="button" data-dir="3" aria-label="Right">→</button>
  </div>`;

let ortPromise = null;
function loadOrt(src) {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  ortPromise ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = () => resolve(globalThis.ort);
    script.onerror = () => reject(new Error("couldn't load ONNX Runtime"));
    document.head.appendChild(script);
  });
  return ortPromise;
}

/**
 * Build the demo inside `root`.
 * @param {HTMLElement} root
 * @param {{ base?: string, autoLoad?: boolean, ortCdn?: string, mode?: keyof MODES }} [options]
 *   base: URL prefix of model.onnx / model_arena.onnx (default: same folder as the page).
 *   autoLoad: download the duel network immediately (default) or wait for a click.
 */
export function mountSnakeDemo(root, options = {}) {
  const base = options.base ?? "";
  const ortCdn = options.ortCdn ?? ORT_CDN;
  root.classList.add("snake-demo");
  root.innerHTML = TEMPLATE;

  const $ = (key) => root.querySelector(`[data-sd="${key}"]`);
  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const overlay = $("overlay");
  const tabs = [...root.querySelectorAll(".sd-tabs button")];
  const sessions = {};

  let mode = options.mode ?? "watch";
  let arenaN = 6;
  let speedIdx = 0;
  let theme = {};
  let game = null;
  let prev = null; // state before the latest turn, for animating between turns
  let lastTick = 0; // time the latest turn started animating (fixed cadence, see advance)
  let tickDelay = SPEEDS[0][1];
  let paused = false;
  let pausedAt = 0;
  let visible = true;
  let started = false;
  let restartTimer = null;
  let waitingForHuman = false;
  let waitForReplay = false;
  let humanQueue = [];
  let aiPlan = null; // AI moves for the current state, decided while the last move animates
  let planToken = 0;
  const tally = { watch: [0, 0, 0], play: [0, 0, 0], bot: [0, 0, 0], arena: Array(9).fill(0) };

  const controllers = () => MODES[mode].controllers ?? Array(arenaN).fill("ai");
  const names = () => MODES[mode].names ?? COLOR_NAMES.slice(0, arenaN);
  const session = () => sessions[MODES[mode].model];

  // Board colours follow the host page's theme, read from the component's CSS variables.
  function readTheme() {
    const css = getComputedStyle(root);
    const v = (name) => css.getPropertyValue(name).trim();
    theme = { bg: v("--sd-board"), line: v("--sd-line"), food: v("--sd-food") };
  }

  // ---------- game flow ----------
  // The game is turn-based, exactly like training. The animation loop (frame) owns the
  // clock: it renders every display frame and advances one turn every tickDelay ms, so
  // motion stays continuous. AI moves are computed ahead of time so turns never stall.

  function startGame() {
    clearTimeout(restartTimer);
    const n = controllers().length;
    game = newGame(MODES[mode].board, n, Math.random, 100, mode === "arena" ? n : 1);
    prev = structuredClone(game);
    humanQueue = [];
    waitingForHuman = controllers().includes("human");
    $("dpad").hidden = mode !== "play";
    if (waitingForHuman) showOverlay("You're blue. Press an arrow key or click to start.");
    else hideOverlay();
    lastTick = performance.now();
    planNext();
    updatePanel();
  }

  function planNext() {
    const token = ++planToken;
    aiPlan = null;
    decideAI(game).then((plan) => {
      if (token === planToken) aiPlan = plan;
    });
  }

  async function decideAI(state) {
    const actions = Array(state.snakes.length).fill(0);
    const net = session();
    const { window } = MODELS[MODES[mode].model];
    const ai = controllers().map((c, i) => (c === "ai" && state.snakes[i].alive ? i : -1)).filter((i) => i >= 0);
    if (ai.length && net) {
      const [ch, size] = window ? [NUM_CHANNELS + 1, window] : [NUM_CHANNELS, state.boardSize];
      const plane = ch * size * size;
      const data = new Float32Array(ai.length * plane);
      ai.forEach((i, j) => data.set(window ? windowObservation(state, i, window) : observation(state, i), j * plane));
      const out = await net.run({ obs: new globalThis.ort.Tensor("float32", data, [ai.length, ch, size, size]) });
      ai.forEach((i, j) => {
        const probs = maskedSoftmax(Array.from(out.logits.data.slice(j * 4, j * 4 + 4)), actionMask(state, i));
        // Copies of one deterministic network mirror each other from symmetric starts, so
        // all-AI modes sample from the policy (as in training); other modes play greedily.
        actions[i] = mode === "watch" || mode === "arena" ? sample(probs) : probs.indexOf(Math.max(...probs));
      });
    }
    return { actions };
  }

  function advance(now) {
    if (!session() || paused || !visible || waitingForHuman || game.over || !aiPlan) return;
    if (now - lastTick < tickDelay) return;
    const actions = aiPlan.actions.slice();
    controllers().forEach((c, i) => {
      if (!game.snakes[i].alive) return;
      if (c === "greedy") actions[i] = greedyAction(game, i);
      if (c === "human") actions[i] = humanAction(i); // read at the last moment: most responsive
    });
    prev = structuredClone(game);
    step(game, actions);
    // Keep a fixed cadence; only resynchronise after a long stall (e.g. a background tab).
    lastTick = now - lastTick < 2 * tickDelay ? lastTick + tickDelay : now;
    updatePanel();
    // Let the final move finish animating before announcing the result.
    if (game.over) restartTimer = setTimeout(onGameOver, tickDelay);
    else planNext();
  }

  function humanAction(i) {
    const current = heading(game.snakes[i]);
    while (humanQueue.length) {
      const next = humanQueue.shift();
      if (current < 0 || next !== OPPOSITE[current]) return next;
    }
    return current >= 0 ? current : 0;
  }

  function onGameOver() {
    const w = game.winner;
    const t = tally[mode];
    if (w === -1) t[mode === "arena" ? 8 : 2]++; else t[w]++;
    let title;
    if (w === -1) title = "Draw";
    else if (mode === "play") title = w === 1 ? "You win!" : "The AI wins";
    else title = `${names()[w]} wins`;
    if (mode === "play") {
      showOverlay(title, "Press a direction to play again");
      waitForReplay = true;
    } else {
      showOverlay(title);
      restartTimer = setTimeout(startGame, 1600);
    }
    updatePanel();
  }

  function onDirection(dir) {
    if (mode !== "play" || !session()) return;
    if (waitForReplay) {
      waitForReplay = false;
      startGame();
      humanQueue = [dir];
    } else if (humanQueue.length < 2) {
      humanQueue.push(dir);
    }
    if (waitingForHuman) {
      waitingForHuman = false;
      lastTick = performance.now() - tickDelay; // first move right away
      hideOverlay();
    }
  }

  // ---------- rendering ----------

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.getBoundingClientRect().width;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
  }

  // Draws every animation frame. Game logic advances in whole turns; between turns each
  // snake slides from its previous cells (`prev`) to its current ones (`game`) by t in [0, 1].
  function render(t) {
    const B = game.boardSize, W = canvas.width, cell = W / B;
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, W);
    ctx.fillStyle = theme.line;
    const dot = Math.max(1.5, cell * 0.06);
    for (let r = 0; r < B; r++) {
      for (let c = 0; c < B; c++) ctx.fillRect((c + 0.5) * cell - dot / 2, (r + 0.5) * cell - dot / 2, dot, dot);
    }
    for (let i = 0; i < B * B; i++) {
      const now = game.food[i], before = prev.food[i];
      if (!now && !before) continue;
      let scale = 1;
      if (now && !before) scale = easeOutBack(Math.min(1, t * 1.4)); // new food pops in
      if (!now && before) scale = t < 0.55 ? 1 : Math.max(0, 1 - (t - 0.55) / 0.3); // eaten
      if (scale <= 0) continue;
      ctx.fillStyle = theme.food;
      ctx.beginPath();
      ctx.arc(((i % B) + 0.5) * cell, (Math.floor(i / B) + 0.5) * cell, cell * 0.24 * scale, 0, Math.PI * 2);
      ctx.fill();
    }
    game.snakes.forEach((snake, i) => drawSnake(prev.snakes[i], snake, COLORS[i % COLORS.length], cell, t));
  }

  function drawSnake(before, snake, color, cell, t) {
    const died = before.alive && !snake.alive;
    ctx.globalAlpha = !snake.alive && !died ? 0.25 : died ? 1 - 0.7 * t : 1;
    const points = bodyPoints(before, snake, t).map(([r, c]) => [(c + 0.5) * cell, (r + 0.5) * cell]);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = cell * 0.64;
    // Body: one continuous rounded stroke, a slightly lighter shade toward the tail.
    for (let j = points.length - 1; j > 0; j--) {
      ctx.strokeStyle = mix(color, theme.bg, 0.35 * (j / points.length));
      ctx.beginPath();
      ctx.moveTo(points[j][0], points[j][1]);
      ctx.lineTo(points[j - 1][0], points[j - 1][1]);
      ctx.stroke();
    }
    // Head with eyes looking along the direction of travel.
    const [hx, hy] = points[0];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(hx, hy, cell * 0.4, 0, Math.PI * 2);
    ctx.fill();
    const dir = heading(snake);
    const [dr, dc] = dir >= 0 ? DIRECTIONS[dir] : [-1, 0];
    const fwd = cell * 0.14, side = cell * 0.17;
    ctx.fillStyle = "#0b1020";
    for (const k of [-1, 1]) {
      ctx.beginPath();
      ctx.arc(hx + dc * fwd + dr * side * k, hy + dr * fwd + dc * side * k, cell * 0.075, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function frame(now) {
    if (!root.isConnected) return; // unmounted
    if (game && visible) {
      advance(now);
      render(Math.max(0, Math.min(1, ((paused ? pausedAt : now) - lastTick) / tickDelay)));
    }
    requestAnimationFrame(frame);
  }

  function showOverlay(title, sub = "", action = null) {
    overlay.innerHTML = "";
    const div = document.createElement("div");
    div.textContent = title;
    if (sub) {
      const small = document.createElement("small");
      small.textContent = sub;
      div.appendChild(small);
    }
    if (action) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sd-start";
      btn.textContent = action.label;
      btn.addEventListener("click", action.run);
      div.appendChild(btn);
    }
    overlay.appendChild(div);
    overlay.hidden = false;
  }

  function hideOverlay() { overlay.hidden = true; }

  // ---------- side panel ----------

  function updatePanel() {
    const label = names();
    // In the arena the list is a live leaderboard: alive first, then by length.
    const order = game.snakes.map((_, i) => i);
    if (mode === "arena") {
      order.sort((a, b) => (game.snakes[b].alive - game.snakes[a].alive)
        || (game.snakes[b].body.length - game.snakes[a].body.length));
    }
    $("snakes").innerHTML = order.map((i) => {
      const s = game.snakes[i];
      return `<li class="${s.alive ? "" : "sd-dead"}"><span class="sd-dot" style="background:${COLORS[i % COLORS.length]}"></span>
        <span>${label[i]}</span><span class="sd-len">${s.body.length}</span></li>`;
    }).join("");

    const t = tally[mode];
    $("tally").textContent = mode === "arena"
      ? `Wins: ${label.map((n, i) => `${n} ${t[i]}`).join(" · ")}`
      : `Wins: ${label[0]} ${t[0]} · ${label[1]} ${t[1]}`;
  }

  // ---------- networks ----------

  async function loadModel(name) {
    if (sessions[name]) return true;
    showOverlay("Loading…");
    try {
      const ort = await loadOrt(`${ortCdn}ort.min.js`);
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = ortCdn;
      sessions[name] = await ort.InferenceSession.create(base + MODELS[name].file);
      return true;
    } catch (err) {
      showOverlay("Couldn't load the network", String(err.message ?? err));
      return false;
    }
  }

  async function selectMode(next) {
    mode = next;
    waitForReplay = false;
    tabs.forEach((b) => b.classList.toggle("sd-active", b.dataset.mode === mode));
    $("arena-controls").hidden = mode !== "arena";
    if (!started) {
      // Before the first load, just show the right board.
      game = newGame(MODES[mode].board, controllers().length);
      prev = structuredClone(game);
      updatePanel();
      return;
    }
    if (!(await loadModel(MODES[mode].model))) return;
    startGame();
  }

  async function start() {
    started = true;
    if (await loadModel(MODES[mode].model)) startGame();
  }

  // ---------- wiring ----------

  tabs.forEach((btn) => btn.addEventListener("click", () => { btn.blur(); selectMode(btn.dataset.mode); }));
  const setArenaSize = (n) => {
    arenaN = Math.max(2, Math.min(8, n));
    $("arena-count").textContent = `${arenaN} snakes`;
    if (mode === "arena") selectMode("arena");
  };
  $("fewer").addEventListener("click", () => setArenaSize(arenaN - 1));
  $("more").addEventListener("click", () => setArenaSize(arenaN + 1));
  $("speed").addEventListener("click", () => {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    tickDelay = SPEEDS[speedIdx][1];
    $("speed").textContent = `Speed ${SPEEDS[speedIdx][0]}×`;
  });
  $("restart").addEventListener("click", () => { if (started) { waitForReplay = false; startGame(); } });
  $("pause").addEventListener("click", () => {
    paused = !paused;
    $("pause").textContent = paused ? "Resume" : "Pause";
    if (paused) pausedAt = performance.now();
    else lastTick += performance.now() - pausedAt;
  });
  root.querySelectorAll(".sd-dpad button").forEach((b) => b.addEventListener("click", () => onDirection(Number(b.dataset.dir))));
  // Arrow keys steer only while you're playing and the board is on screen, so they still
  // scroll the page the rest of the time.
  window.addEventListener("keydown", (e) => {
    if (!(e.code in KEYS) || mode !== "play" || !visible || !root.isConnected) return;
    if (e.target instanceof HTMLElement && e.target.closest("input, textarea, select, [contenteditable]")) return;
    e.preventDefault();
    onDirection(KEYS[e.code]);
  });
  let touchStart = null;
  canvas.addEventListener("touchstart", (e) => { touchStart = e.touches[0]; }, { passive: true });
  canvas.addEventListener("touchend", (e) => {
    if (!touchStart || mode !== "play") return;
    const t = e.changedTouches[0];
    const dx = t.clientX - touchStart.clientX, dy = t.clientY - touchStart.clientY;
    if (Math.max(Math.abs(dx), Math.abs(dy)) > 24) onDirection(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 3 : 2) : (dy > 0 ? 1 : 0));
    touchStart = null;
  });
  // Click / tap steering: turn toward the clicked spot, relative to your snake's head.
  canvas.addEventListener("click", (e) => {
    if (mode !== "play" || !game) return;
    const rect = canvas.getBoundingClientRect();
    const cell = rect.width / game.boardSize;
    const [hr, hc] = game.snakes[1].body[0];
    const dx = (e.clientX - rect.left) / cell - (hc + 0.5);
    const dy = (e.clientY - rect.top) / cell - (hr + 0.5);
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 0.5) return;
    onDirection(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 3 : 2) : (dy > 0 ? 1 : 0));
  });

  new ResizeObserver(resizeCanvas).observe(canvas);
  // Follow the host page's light/dark switch (a class or data attribute on <html>).
  new MutationObserver(readTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", readTheme);
  // Pause the game while it's scrolled out of view.
  new IntersectionObserver(([entry]) => {
    const now = performance.now();
    if (!entry.isIntersecting && visible) {
      visible = false;
      if (!paused) pausedAt = now;
    } else if (entry.isIntersecting && !visible) {
      visible = true;
      if (!paused) lastTick += now - pausedAt;
    }
  }).observe(root);

  readTheme();
  resizeCanvas();
  tabs.forEach((b) => b.classList.toggle("sd-active", b.dataset.mode === mode));
  game = newGame(MODES[mode].board, controllers().length);
  prev = structuredClone(game);
  updatePanel();
  requestAnimationFrame(frame);
  if (options.autoLoad === false) {
    showOverlay("Watch two trained networks play", "", {
      label: `▶ Load the AI (${MODELS[MODES[mode].model].mb} MB)`,
      run: () => start(),
    });
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

function heading(snake) {
  const [h, n] = snake.body;
  return DIRECTIONS.findIndex(([r, c]) => r === h[0] - n[0] && c === h[1] - n[1]);
}

const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

function easeOutBack(t) {
  const c = 1.7;
  return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2;
}

// Points along the body at fraction t of the current turn: the head slides from its old
// cell to its new one, the middle sits on cell centres, and the tail slides after it.
function bodyPoints(before, after, t) {
  const a = before.body, b = after.body;
  if (a === b || !before.alive) return b.slice();
  const points = [lerp(a[0], b[0], t)];
  // Keep every current cell, including the one the tail is moving into, so the tail
  // bends through corners instead of cutting across them diagonally.
  for (let j = 1; j < b.length; j++) points.push(b[j]);
  if (b.length > 1) points.push(lerp(a[a.length - 1], b[b.length - 1], t));
  return points;
}

// Blend colour a toward b by t (0..1), so body shading never stacks like alpha does.
// Accepts #rrggbb or rgb(...) strings (computed CSS colours).
function mix(a, b, t) {
  const parse = (c) => (c.startsWith("#")
    ? [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16))
    : (c.match(/\d+(\.\d+)?/g) || [0, 0, 0]).slice(0, 3).map(Number));
  const [x, y] = [parse(a), parse(b)];
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
}
