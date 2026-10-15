// Canvas rendering of a slither arena. A camera follows one snake; snakes are round-capped
// strokes through their trail points (tapering at the tail), with the head interpolated
// between physics ticks so motion stays smooth at 60 fps on a 30 Hz simulation.
import { bodyRadius } from "./rules.js";

export const COLORS = ["#f59e0b", "#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#facc15",
  "#fb7185", "#22d3ee", "#fb923c", "#4ade80", "#c084fc", "#60a5fa"];
export const NAMES = ["Orange", "Blue", "Violet", "Green", "Pink", "Yellow", "Coral", "Cyan",
  "Amber", "Lime", "Purple", "Sky"];

/** Head of snake k at fraction `alpha` of the way from the previous tick to the current one. */
export function headAt(game, prevHeads, k, alpha) {
  const h = game.snakes[k].body[0], p = prevHeads[k];
  if (!p) return h;
  return [p[0] + (h[0] - p[0]) * alpha, p[1] + (h[1] - p[1]) * alpha];
}

/**
 * Draw the arena and return the camera ({x, y, scale}), eased toward the followed snake.
 * view: { alpha, prevHeads, follow, human, theme, dpr }
 * theme: { board, outside, grid, wall, fg }
 */
export function drawArena(ctx, game, cfg, view, camera) {
  const { width: W, height: H } = ctx.canvas;
  const { theme, dpr } = view;
  const target = game.snakes[view.follow];
  let cam = camera;
  if (target && target.alive) {
    const [fx, fy] = headAt(game, view.prevHeads, view.follow, view.alpha);
    const want = Math.min(W, H) / (2 * (70 + 6 * bodyRadius(target.mass, cfg)));
    cam = camera
      ? { x: camera.x + (fx - camera.x) * 0.2, y: camera.y + (fy - camera.y) * 0.2, scale: camera.scale + (want - camera.scale) * 0.05 }
      : { x: fx, y: fy, scale: want };
  } else if (!cam) {
    cam = { x: 0, y: 0, scale: Math.min(W, H) / (2.1 * cfg.arena_radius) };
  }
  const sx = (x) => W / 2 + (x - cam.x) * cam.scale;
  const sy = (y) => H / 2 + (y - cam.y) * cam.scale;
  const R = cfg.arena_radius;

  ctx.fillStyle = theme.outside;
  ctx.fillRect(0, 0, W, H);
  ctx.beginPath();
  ctx.arc(sx(0), sy(0), R * cam.scale, 0, 2 * Math.PI);
  ctx.fillStyle = theme.board;
  ctx.fill();

  // background grid, clipped to the arena
  ctx.save();
  ctx.clip();
  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = dpr;
  ctx.beginPath();
  const step = 30, halfW = W / 2 / cam.scale, halfH = H / 2 / cam.scale;
  for (let x = Math.floor((cam.x - halfW) / step) * step; x <= cam.x + halfW; x += step) { ctx.moveTo(sx(x), 0); ctx.lineTo(sx(x), H); }
  for (let y = Math.floor((cam.y - halfH) / step) * step; y <= cam.y + halfH; y += step) { ctx.moveTo(0, sy(y)); ctx.lineTo(W, sy(y)); }
  ctx.stroke();
  ctx.restore();

  ctx.beginPath();
  ctx.arc(sx(0), sy(0), R * cam.scale, 0, 2 * Math.PI);
  ctx.strokeStyle = theme.wall;
  ctx.lineWidth = 3 * dpr;
  ctx.stroke();

  // food: a soft halo and a bright core, coloured by slot so the field looks lively
  game.food.forEach((item, f) => {
    if (!item) return;
    const x = sx(item[0]), y = sy(item[1]);
    if (x < -20 || y < -20 || x > W + 20 || y > H + 20) return;
    const r = (0.9 + Math.sqrt(item[2])) * cam.scale;
    const color = COLORS[f % COLORS.length];
    ctx.globalAlpha = 0.22;
    dot(ctx, x, y, r * 2.2, color);
    ctx.globalAlpha = 1;
    dot(ctx, x, y, r, color);
  });

  // snakes, smallest first so big snakes are drawn on top
  game.snakes.map((s, k) => k).filter((k) => game.snakes[k].alive)
    .sort((a, b) => game.snakes[a].mass - game.snakes[b].mass)
    .forEach((k) => drawSnake(ctx, game, cfg, k, view, sx, sy, cam.scale));

  drawMinimap(ctx, game, cfg, view, W, H);
  return cam;
}

function dot(ctx, x, y, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawSnake(ctx, game, cfg, k, view, sx, sy, scale) {
  const s = game.snakes[k];
  const color = COLORS[k % COLORS.length];
  const r = bodyRadius(s.mass, cfg) * scale;
  const head = headAt(game, view.prevHeads, k, view.alpha);
  const pts = [head, ...s.body.slice(1)];
  const tail = Math.min(12, Math.floor(pts.length / 3));
  const main = pts.length - tail;
  const path = (from, to) => {
    ctx.beginPath();
    ctx.moveTo(sx(pts[from][0]), sy(pts[from][1]));
    for (let i = from + 1; i < to; i++) ctx.lineTo(sx(pts[i][0]), sy(pts[i][1]));
  };
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (s.boosting) {
    ctx.globalAlpha = 0.35;
    path(0, pts.length);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * r + 8 * view.dpr;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  // an outline pass, then the body colour; the last `tail` segments taper
  for (const [width, stroke] of [[2 * r + 2 * view.dpr, darker(color)], [2 * r, color]]) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    path(0, main + 1);
    ctx.stroke();
    for (let i = main; i < pts.length - 1; i++) {
      const t = 1 - (i - main + 1) / (tail + 1);
      ctx.lineWidth = Math.max(view.dpr, width * (0.35 + 0.65 * t));
      ctx.beginPath();
      ctx.moveTo(sx(pts[i][0]), sy(pts[i][1]));
      ctx.lineTo(sx(pts[i + 1][0]), sy(pts[i + 1][1]));
      ctx.stroke();
    }
  }
  // eyes, looking along the heading
  const hx = sx(head[0]), hy = sy(head[1]);
  const c = Math.cos(s.heading), sn = Math.sin(s.heading);
  for (const side of [-1, 1]) {
    const ex = hx + (c * 0.35 - sn * side * 0.5) * r, ey = hy + (sn * 0.35 + c * side * 0.5) * r;
    dot(ctx, ex, ey, 0.32 * r, "#ffffff");
    dot(ctx, ex + c * 0.12 * r, ey + sn * 0.12 * r, 0.16 * r, "#1f1f1d");
  }
  if (k === view.human) {
    ctx.fillStyle = view.theme.fg;
    ctx.font = `600 ${12 * view.dpr}px -apple-system, BlinkMacSystemFont, Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("You", hx, hy - r - 8 * view.dpr);
  }
}

function drawMinimap(ctx, game, cfg, view, W, H) {
  const { dpr, theme } = view;
  const m = 34 * dpr, cx = W - m - 10 * dpr, cy = H - m - 10 * dpr, k = m / cfg.arena_radius;
  ctx.globalAlpha = 0.85;
  dot(ctx, cx, cy, m, theme.board);
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, m, 0, 2 * Math.PI);
  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = dpr;
  ctx.stroke();
  game.snakes.forEach((s, i) => {
    if (!s.alive) return;
    const [x, y] = s.body[0];
    dot(ctx, cx + x * k, cy + y * k, (i === view.follow ? 2.6 : 1.6) * dpr, COLORS[i % COLORS.length]);
  });
}

// #rrggbb -> a darker rgb() for outlines
function darker(hex) {
  const v = [1, 3, 5].map((i) => Math.round(parseInt(hex.slice(i, i + 2), 16) * 0.7));
  return `rgb(${v.join(",")})`;
}
