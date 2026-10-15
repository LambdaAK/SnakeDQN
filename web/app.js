import { actionMask, DIRECTIONS, greedyAction, newGame, NUM_CHANNELS, observation, step } from "./rules.js";

const BOARD = 11;
const COLORS = ["#f59e0b", "#38bdf8"];
const ARROWS = ["↑", "↓", "←", "→"];
const MODES = {
  watch: { names: ["Orange AI", "Blue AI"], controllers: ["ai", "ai"] },
  play: { names: ["AI", "You"], controllers: ["ai", "human"] },
  bot: { names: ["AI", "Greedy bot"], controllers: ["ai", "greedy"] },
};
const KEYS = { ArrowUp: 0, KeyW: 0, ArrowDown: 1, KeyS: 1, ArrowLeft: 2, KeyA: 2, ArrowRight: 3, KeyD: 3 };
const OPPOSITE = [1, 0, 3, 2];

const $ = (id) => document.getElementById(id);
const canvas = $("board");
const ctx = canvas.getContext("2d");
const overlay = $("overlay");

let session = null;
let mode = "watch";
let game = null;
let prev = null; // state before the latest turn, for animating between turns
let lastTick = 0;
let tickDelay = 300;
let paused = false;
let busy = false;
let timer = null;
let restartTimer = null;
let waitingForHuman = false;
let humanQueue = [];
let brain = { probs: null, value: null, chosen: -1 };
const tally = { watch: [0, 0, 0], play: [0, 0, 0], bot: [0, 0, 0] };

if (new URLSearchParams(location.search).has("embed")) document.body.classList.add("embed");

// ---------- game flow ----------

function startGame() {
  clearTimeout(restartTimer);
  game = newGame(BOARD, 2);
  prev = structuredClone(game);
  lastTick = performance.now();
  brain = { probs: null, value: null, chosen: -1 };
  humanQueue = [];
  waitingForHuman = MODES[mode].controllers.includes("human");
  $("dpad").hidden = mode !== "play";
  if (waitingForHuman) {
    showOverlay("Press an arrow key, or click where you want to go", "Your snake (blue) keeps moving; steer with arrows, WASD, clicks or swipes");
  } else {
    hideOverlay();
  }
  updatePanel();
  schedule();
}

function schedule() {
  clearTimeout(timer);
  tickDelay = 440 - Number($("speed").value);
  timer = setTimeout(async () => {
    await tick();
    if (!game.over) schedule();
  }, tickDelay);
}

async function tick() {
  if (!session || paused || busy || game.over || waitingForHuman) return;
  busy = true;
  try {
    const actions = await chooseActions();
    prev = structuredClone(game);
    step(game, actions);
    lastTick = performance.now();
  } finally {
    busy = false;
  }
  updatePanel();
  // Let the final move finish animating before announcing the result.
  if (game.over) restartTimer = setTimeout(onGameOver, tickDelay);
}

async function chooseActions() {
  const controllers = MODES[mode].controllers;
  const actions = [0, 0];
  const aiSnakes = controllers.map((c, i) => (c === "ai" && game.snakes[i].alive ? i : -1)).filter((i) => i >= 0);
  if (aiSnakes.length) {
    const plane = NUM_CHANNELS * BOARD * BOARD;
    const data = new Float32Array(aiSnakes.length * plane);
    aiSnakes.forEach((i, j) => data.set(observation(game, i), j * plane));
    const out = await session.run({ obs: new ort.Tensor("float32", data, [aiSnakes.length, NUM_CHANNELS, BOARD, BOARD]) });
    aiSnakes.forEach((i, j) => {
      const probs = maskedSoftmax(Array.from(out.logits.data.slice(j * 4, j * 4 + 4)), actionMask(game, i));
      // Two copies of one deterministic network mirror each other from symmetric starts,
      // so AI-vs-AI samples from the policy (as in training); other modes play greedily.
      actions[i] = mode === "watch" ? sample(probs) : probs.indexOf(Math.max(...probs));
      if (i === 0) brain = { probs, value: out.value.data[j], chosen: actions[i] };
    });
  }
  controllers.forEach((c, i) => {
    if (!game.snakes[i].alive) return;
    if (c === "greedy") actions[i] = greedyAction(game, i);
    if (c === "human") actions[i] = humanAction(i);
  });
  return actions;
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
  const names = MODES[mode].names;
  const t = tally[mode];
  if (w === -1) t[2]++; else t[w]++;
  let title;
  if (w === -1) title = "Draw: both snakes went down";
  else if (mode === "play") title = w === 1 ? "You win!" : "The AI wins";
  else title = `${names[w]} wins`;
  if (mode === "play") {
    showOverlay(title, `Turn ${game.turn} · press a direction to play again`);
    waitForReplay = true;
  } else {
    showOverlay(title, `Turn ${game.turn} · next game starting`);
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
    hideOverlay();
    return;
  }
  if (humanQueue.length < 2) humanQueue.push(dir);
  if (waitingForHuman) {
    waitingForHuman = false;
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
  const W = canvas.width, cell = W / BOARD;
  ctx.clearRect(0, 0, W, W);
  for (let r = 0; r < BOARD; r++) {
    for (let c = 0; c < BOARD; c++) {
      ctx.fillStyle = (r + c) % 2 ? "#10172b" : "#0e1426";
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
  }
  for (let i = 0; i < BOARD * BOARD; i++) {
    const now = game.food[i], before = prev.food[i];
    if (!now && !before) continue;
    let scale = 1;
    if (now && !before) scale = easeOutBack(Math.min(1, t * 1.4)); // new food pops in
    if (!now && before) scale = t < 0.55 ? 1 : Math.max(0, 1 - (t - 0.55) / 0.3); // eaten
    if (scale <= 0) continue;
    const r = Math.floor(i / BOARD), c = i % BOARD;
    ctx.save();
    ctx.shadowColor = "#f43f5e";
    ctx.shadowBlur = cell * 0.5;
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath();
    ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, cell * 0.24 * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  game.snakes.forEach((snake, i) => drawSnake(prev.snakes[i], snake, COLORS[i], cell, t));
}

const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const easeInOut = (t) => t * t * (3 - 2 * t);
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
  for (let j = 1; j < b.length - 1; j++) points.push(b[j]);
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
  const s = easeInOut(t);
  const points = bodyPoints(before, snake, s).map(([r, c]) => [(c + 0.5) * cell, (r + 0.5) * cell]);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = cell * 0.64;
  // Body: one continuous rounded stroke, a slightly darker shade toward the tail.
  for (let j = points.length - 1; j > 0; j--) {
    ctx.strokeStyle = mix(color, "#0e1426", 0.4 * (j / points.length));
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
  const t = paused ? 1 : Math.min(1, (now - lastTick) / tickDelay);
  render(t);
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
  const names = MODES[mode].names;
  $("snakes").innerHTML = game.snakes.map((s, i) => `
    <div class="snake-card ${s.alive ? "" : "dead"}">
      <span class="dot" style="background:${COLORS[i]}"></span>
      <div>
        <div class="name">${names[i]}</div>
        <div class="health"><div style="width:${s.alive ? (100 * s.health) / game.maxHealth : 0}%;background:${COLORS[i]}"></div></div>
      </div>
      <span class="meta">length ${s.body.length}${s.alive ? "" : " · out"}</span>
    </div>`).join("");

  $("probs").innerHTML = ARROWS.map((arrow, a) => {
    const p = brain.probs ? brain.probs[a] : 0;
    return `<div class="prob ${a === brain.chosen ? "chosen" : ""}">
      <span>${arrow}</span><div class="bar"><div style="width:${(100 * p).toFixed(1)}%"></div></div>
      <span class="pct">${brain.probs ? Math.round(100 * p) + "%" : "–"}</span></div>`;
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
  document.querySelector(".ai-name").textContent = mode === "watch" ? "orange" : "AI's";

  const t = tally[mode];
  $("tally").textContent = `This session: ${names[0]} ${t[0]} · ${names[1]} ${t[1]} · draws ${t[2]}`;
}

// ---------- training stats ----------

async function loadStats() {
  let stats;
  try {
    stats = await (await fetch("stats.json")).json();
  } catch {
    document.querySelector(".explain h2:nth-of-type(2)").hidden = true;
    return;
  }
  drawChart(stats.curve);
  const tiles = [];
  const fe = stats.final_eval;
  if (fe) {
    $("eval-games").textContent = "200";
    tiles.push([`${Math.round(100 * fe.win_vs_flood_fill)}%`, `wins vs flood-fill bot (${fe.games} games)`]);
    tiles.push([`${Math.round(100 * fe.win_vs_greedy)}%`, `wins vs greedy bot (${fe.games} games)`]);
  }
  tiles.push([`${(stats.checkpoint.env_steps / 1e6).toFixed(0)}M`, "self-play turns of training"]);
  tiles.push([`${(stats.checkpoint.parameters / 1e6).toFixed(1)}M`, "network parameters"]);
  $("results").innerHTML = tiles.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
}

function drawChart(points) {
  if (!points || !points.length) return;
  const W = 720, H = 260, L = 44, R = 12, T = 12, B = 34;
  const maxX = points[points.length - 1].env_steps;
  const x = (v) => L + ((W - L - R) * v) / maxX;
  const y = (v) => T + (H - T - B) * (1 - v);
  const line = (key) => points.map((p, i) => `${i ? "L" : "M"}${x(p.env_steps).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");
  let grid = "";
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    grid += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#26304f"/>`;
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
      <path d="${line("win_vs_greedy")}" fill="none" stroke="#38bdf8" stroke-width="2.5"/>
      <path d="${line("win_vs_flood_fill")}" fill="none" stroke="#f59e0b" stroke-width="2.5"/>
    </svg>
    <div class="legend"><span style="--c:#f59e0b">vs flood-fill bot (strongest scripted)</span>
      <span style="--c:#38bdf8">vs greedy bot</span><span>x-axis: self-play turns</span></div>`;
}

// ---------- wiring ----------

document.querySelectorAll(".modes button").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".modes button").forEach((b) => b.classList.toggle("active", b === btn));
  mode = btn.dataset.mode;
  waitForReplay = false;
  startGame();
}));
$("restart").addEventListener("click", () => { waitForReplay = false; startGame(); });
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  if (!paused) schedule();
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
  const cell = rect.width / BOARD;
  const [hr, hc] = game.snakes[1].body[0];
  const dx = (e.clientX - rect.left) / cell - (hc + 0.5);
  const dy = (e.clientY - rect.top) / cell - (hr + 0.5);
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 0.5) return;
  onDirection(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 3 : 2) : (dy > 0 ? 1 : 0));
});
window.addEventListener("resize", resizeCanvas);

async function init() {
  game = newGame(BOARD, 2);
  prev = structuredClone(game);
  resizeCanvas();
  requestAnimationFrame(frame);
  updatePanel();
  showOverlay("Loading neural network…");
  loadStats();
  try {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    session = await ort.InferenceSession.create("model.onnx");
  } catch (err) {
    showOverlay("Couldn't load the neural network", String(err));
    return;
  }
  startGame();
}

init();
