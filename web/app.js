import { actionMask, DIRECTIONS, greedyAction, newGame, NUM_CHANNELS, observation, step, windowObservation } from "./rules.js?v=12";

const COLORS = ["#f59e0b", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#facc15", "#fb7185", "#22d3ee"];
const COLOR_NAMES = ["Orange", "Blue", "Violet", "Green", "Pink", "Yellow", "Coral", "Cyan"];
// Two networks: the 1v1 duel network sees the whole 11x11 board; the arena network sees a
// 17x17 window around its own head, so it plays on a large board with many snakes.
const MODELS = { duel: { url: "model.onnx?v=2", window: 0 }, arena: { url: "model_arena.onnx?v=2", window: 17 } };
const sessions = {};
const ARROWS = ["↑", "↓", "←", "→"];
const MODES = {
  watch: { board: 11, model: "duel", names: ["Orange AI", "Blue AI"], controllers: ["ai", "ai"] },
  play: { board: 11, model: "duel", names: ["AI", "You"], controllers: ["ai", "human"] },
  bot: { board: 11, model: "duel", names: ["AI", "Greedy bot"], controllers: ["ai", "greedy"] },
  arena: { board: 19, model: "arena" },
};
let arenaN = 6;
const arenaSize = () => arenaN;
const SPEEDS = [[1, 180], [2, 100], [0.5, 320]]; // [label multiplier, ms per move]
let speedIdx = 0;
// Board colours follow the page theme (light paper or dark), read from the CSS tokens.
let theme = {};
function readTheme() {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  theme = { bg: v("--bg"), line: v("--line"), food: v("--food") };
}
const controllers = () => MODES[mode].controllers ?? Array(arenaSize()).fill("ai");
const names = () => MODES[mode].names ?? COLOR_NAMES.slice(0, arenaSize());
const session = () => sessions[MODES[mode].model];
const KEYS = { ArrowUp: 0, KeyW: 0, ArrowDown: 1, KeyS: 1, ArrowLeft: 2, KeyA: 2, ArrowRight: 3, KeyD: 3 };
const OPPOSITE = [1, 0, 3, 2];

const $ = (id) => document.getElementById(id);
const canvas = $("board");
const ctx = canvas.getContext("2d");
const overlay = $("overlay");

let mode = "watch";
let game = null;
let prev = null; // state before the latest turn, for animating between turns
let lastTick = 0; // time the latest turn started animating (fixed cadence, see advance)
let tickDelay = 180;
let paused = false;
let pausedAt = 0;
let restartTimer = null;
let waitingForHuman = false;
let humanQueue = [];
let brain = { probs: null, value: null, chosen: -1 };
let aiPlan = null; // AI moves for the current state, decided while the last move animates
let planToken = 0;
const tally = { watch: [0, 0, 0], play: [0, 0, 0], bot: [0, 0, 0], arena: Array(9).fill(0) };

if (new URLSearchParams(location.search).has("embed")) document.body.classList.add("embed");

// ---------- game flow ----------
// The game is turn-based, exactly like training. The animation loop (frame) owns the
// clock: it renders every display frame and advances one turn every tickDelay ms, so
// motion stays continuous. AI moves are computed ahead of time so turns never stall.

function startGame() {
  clearTimeout(restartTimer);
  const n = controllers().length;
  game = newGame(MODES[mode].board, n, Math.random, 100, mode === "arena" ? n : 1);
  prev = structuredClone(game);
  brain = { probs: null, value: null, chosen: -1 };
  humanQueue = [];
  waitingForHuman = controllers().includes("human");
  $("dpad").hidden = mode !== "play";
  if (waitingForHuman) {
    showOverlay("You're blue. Press an arrow key or click to start.");
  } else {
    hideOverlay();
  }
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
  let decided = null;
  const actions = Array(state.snakes.length).fill(0);
  const net = session();
  const { window } = MODELS[MODES[mode].model];
  const aiSnakes = controllers().map((c, i) => (c === "ai" && state.snakes[i].alive ? i : -1)).filter((i) => i >= 0);
  if (aiSnakes.length && net) {
    const [ch, size] = window ? [NUM_CHANNELS + 1, window] : [NUM_CHANNELS, state.boardSize];
    const plane = ch * size * size;
    const data = new Float32Array(aiSnakes.length * plane);
    aiSnakes.forEach((i, j) => data.set(window ? windowObservation(state, i, window) : observation(state, i), j * plane));
    const out = await net.run({ obs: new ort.Tensor("float32", data, [aiSnakes.length, ch, size, size]) });
    aiSnakes.forEach((i, j) => {
      const probs = maskedSoftmax(Array.from(out.logits.data.slice(j * 4, j * 4 + 4)), actionMask(state, i));
      // Copies of one deterministic network mirror each other from symmetric starts, so
      // all-AI modes sample from the policy (as in training); other modes play greedily.
      actions[i] = mode === "watch" || mode === "arena" ? sample(probs) : probs.indexOf(Math.max(...probs));
      if (i === 0) decided = { probs, value: out.value.data[j], chosen: actions[i] };
    });
  }
  return { actions, brain: decided };
}

function advance(now) {
  if (!session() || paused || waitingForHuman || game.over || !aiPlan) return;
  if (now - lastTick < tickDelay) return;
  const actions = aiPlan.actions.slice();
  if (aiPlan.brain) brain = aiPlan.brain;
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
  const dr = h[0] - n[0], dc = h[1] - n[1];
  const idx = DIRECTIONS.findIndex(([r, c]) => r === dr && c === dc);
  return idx;
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
  const drawSlot = mode === "arena" ? 8 : 2;
  if (w === -1) t[drawSlot]++; else t[w]++;
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

let waitForReplay = false;

function onDirection(dir) {
  if (mode !== "play") return;
  if (waitForReplay) {
    waitForReplay = false;
    startGame();
    humanQueue = [dir];
    waitingForHuman = false;
    lastTick = performance.now() - tickDelay;
    hideOverlay();
    return;
  }
  if (humanQueue.length < 2) humanQueue.push(dir);
  if (waitingForHuman) {
    waitingForHuman = false;
    lastTick = performance.now() - tickDelay;
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
  if (!game) return;
  const B = game.boardSize, W = canvas.width, cell = W / B;
  ctx.clearRect(0, 0, W, W);
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
    const r = Math.floor(i / B), c = i % B;
    ctx.save();
    ctx.fillStyle = theme.food;
    ctx.beginPath();
    ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, cell * 0.24 * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  game.snakes.forEach((snake, i) => drawSnake(prev.snakes[i], snake, COLORS[i % COLORS.length], cell, t));
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

function drawSnake(before, snake, color, cell, t) {
  const died = before.alive && !snake.alive;
  if (!snake.alive && !died) {
    ctx.globalAlpha = 0.25;
  } else {
    ctx.globalAlpha = died ? 1 - 0.7 * t : 1;
  }
  const points = bodyPoints(before, snake, t).map(([r, c]) => [(c + 0.5) * cell, (r + 0.5) * cell]);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = cell * 0.64;
  // Body: one continuous rounded stroke, a slightly darker shade toward the tail.
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

// Blend hex colour a toward b by t (0..1), so body shading never stacks like alpha does.
function mix(a, b, t) {
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
}

function frame(now) {
  if (game) {
    advance(now);
    const clock = paused ? pausedAt : now;
    render(Math.max(0, Math.min(1, (clock - lastTick) / tickDelay)));
  }
  requestAnimationFrame(frame);
}

function showOverlay(title, sub = "") {
  overlay.innerHTML = "";
  const div = document.createElement("div");
  div.textContent = title;
  if (sub) {
    const small = document.createElement("small");
    small.textContent = sub;
    div.appendChild(small);
  }
  overlay.appendChild(div);
  overlay.hidden = false;
}

function hideOverlay() { overlay.hidden = true; }

// ---------- side panel ----------

function updatePanel() {
  const label = names();
  // In the arena the cards are a live leaderboard: alive first, then by length.
  const order = game.snakes.map((_, i) => i);
  if (mode === "arena") {
    order.sort((a, b) => (game.snakes[b].alive - game.snakes[a].alive)
      || (game.snakes[b].body.length - game.snakes[a].body.length));
  }
  $("snakes").innerHTML = order.map((i) => {
    const s = game.snakes[i];
    return `<li class="${s.alive ? "" : "dead"}"><span class="dot" style="background:${COLORS[i % COLORS.length]}"></span>
      <span>${label[i]}</span><span class="len">${s.body.length}</span></li>`;
  }).join("");

  $("probs").innerHTML = ARROWS.map((arrow, a) => {
    const p = brain.probs ? brain.probs[a] : 0;
    return `<li class="${a === brain.chosen ? "chosen" : ""}">
      <span>${arrow}</span><div class="track"><div style="width:${(100 * p).toFixed(1)}%"></div></div>
      <span class="pct">${brain.probs ? Math.round(100 * p) + "%" : "–"}</span></li>`;
  }).join("");

  const bar = $("value-bar");
  if (brain.value === null) {
    bar.style.width = "0"; $("value-text").textContent = "–";
  } else {
    const v = Math.max(-1, Math.min(1, brain.value));
    bar.style.left = v >= 0 ? "50%" : `${50 + 50 * v}%`;
    bar.style.width = `${50 * Math.abs(v)}%`;
    $("value-text").textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
  }
  document.querySelector(".ai-name").textContent =
    mode === "watch" || mode === "arena" ? "Orange" : "AI";

  const t = tally[mode];
  if (mode === "arena") {
    const wins = label.map((n, i) => `${n} ${t[i]}`).join(" · ");
    $("tally").textContent = `Wins: ${wins}`;
  } else {
    $("tally").textContent = `Wins: ${label[0]} ${t[0]} · ${label[1]} ${t[1]}`;
  }
}

// ---------- training stats ----------

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
  if (!points || !points.length) return;
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
  const ticks = 5;
  for (let k = 0; k <= ticks; k++) {
    const v = (maxX * k) / ticks;
    grid += `<text x="${x(v)}" y="${H - 10}" text-anchor="middle">${(v / 1e6).toFixed(0)}M</text>`;
  }
  $("chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Win rate over training">
      ${grid}
      <path d="${line("win_vs_greedy")}" fill="none" stroke="var(--fg-faint)" stroke-width="2"/>
      <path d="${line("win_vs_flood_fill")}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    </svg>
    <div class="legend"><span>Duel win rate during training</span>
      <span style="--c:var(--accent)">vs best scripted bot</span>
      <span style="--c:var(--fg-faint)">vs greedy bot</span></div>`;
}

// ---------- wiring ----------

document.querySelectorAll(".tabs button").forEach((btn) => btn.addEventListener("click", async () => {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b === btn));
  mode = btn.dataset.mode;
  waitForReplay = false;
  $("arena-controls").hidden = mode !== "arena";
  btn.blur();
  if (!(await loadModel(MODES[mode].model))) return;
  startGame();
  resizeCanvas();
}));
function setArenaSize(n) {
  arenaN = Math.max(2, Math.min(8, n));
  $("arena-count").textContent = `${arenaN} snakes`;
  if (mode === "arena") startGame();
}
$("fewer").addEventListener("click", () => setArenaSize(arenaN - 1));
$("more").addEventListener("click", () => setArenaSize(arenaN + 1));
$("speed").addEventListener("click", () => {
  speedIdx = (speedIdx + 1) % SPEEDS.length;
  const [label, delay] = SPEEDS[speedIdx];
  tickDelay = delay;
  $("speed").textContent = `Speed ${label}×`;
});
$("theme").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch {}
  readTheme();
});
$("restart").addEventListener("click", () => { waitForReplay = false; startGame(); });
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  if (paused) pausedAt = performance.now();
  else lastTick += performance.now() - pausedAt;
});
document.querySelectorAll("#dpad button").forEach((b) => b.addEventListener("click", () => onDirection(Number(b.dataset.dir))));
window.addEventListener("keydown", (e) => {
  if (e.code in KEYS && mode === "play") {
    e.preventDefault();
    onDirection(KEYS[e.code]);
  }
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
window.addEventListener("resize", resizeCanvas);

// Networks load on first use, so the arena model doesn't slow down the first page load.
async function loadModel(name) {
  if (sessions[name]) return true;
  showOverlay("Loading…");
  try {
    sessions[name] = await ort.InferenceSession.create(MODELS[name].url);
    return true;
  } catch (err) {
    showOverlay("Couldn't load the network", String(err));
    return false;
  }
}

async function init() {
  readTheme();
  game = newGame(MODES[mode].board, 2);
  prev = structuredClone(game);
  resizeCanvas();
  requestAnimationFrame(frame);
  updatePanel();
  loadStats();
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  if (await loadModel("duel")) startGame();
}

init();
