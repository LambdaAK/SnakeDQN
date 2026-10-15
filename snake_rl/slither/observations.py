"""Head-centred sector observation (spec §3): 32 sectors x 8 channels + 7 scalars.

Everything is rotated into the snake's heading frame. A point's sector is
floor((atan2(y', x') + pi + w/2) / w) mod K, w = 2 pi / K; sector K/2 is centred straight ahead,
sector 0 straight behind, and
positive angles are to the left (counter-clockwise). tests/slither_reference.observe is
the loop version; web/slither/observe.js ports it.
"""

from __future__ import annotations

import numpy as np

from snake_rl.slither.rules import body_radius, head_points, live_points, near_pairs

OBS_CHANNELS = 8
NUM_SCALARS = 7
SIZE_SCALE = 15.0  # radius normaliser (radius is about 13 at max mass)


def obs_size(cfg) -> int:
    return cfg.sectors * OBS_CHANNELS + NUM_SCALARS


def _close(d, vision):
    return np.clip(1.0 - d / vision, 0.0, 1.0)


def _sector(rx, ry, c, sn, width, k):
    x = rx * c + ry * sn
    y = ry * c - rx * sn
    return np.floor((np.arctan2(y, x) + np.pi + width / 2) / width).astype(np.int64) % k


def build_observations(state) -> np.ndarray:
    """float32 (N, S, obs_size); all zeros for dead or disabled slots."""
    cfg = state.cfg
    n, s = state.num_envs, state.num_snakes
    k, vision = cfg.sectors, cfg.vision
    width = 2.0 * np.pi / k
    alive = state.alive
    heads = head_points(state)
    hx, hy = heads[..., 0], heads[..., 1]
    c, sn = np.cos(state.heading), np.sin(state.heading)
    radius = body_radius(state.mass, cfg)
    sec = np.zeros((n, s, k, OBS_CHANNELS))
    flat = sec.reshape(-1)
    slots = np.arange(s)

    # 0: wall, along each sector's centre ray
    th = state.heading[..., None] + (-np.pi + np.arange(k) * width)
    dx, dy = np.cos(th), np.sin(th)
    b = hx[..., None] * dx + hy[..., None] * dy
    cc = (hx * hx + hy * hy - cfg.arena_radius * cfg.arena_radius)[..., None]
    sec[..., 0] = _close(-b + np.sqrt(np.maximum(b * b - cc, 0.0)), vision)

    # 1-3: bodies (every live point, including heads)
    pe, pj, pp, pl = live_points(state, alive)
    px, py = state.body[pe, pj, pp, 0], state.body[pe, pj, pp, 1]
    q, i = near_pairs(pe, px, py, heads, alive, vision + float(radius.max(initial=0.0)) + 1e-6)
    e, j, l = pe[q], pj[q], pl[q]
    rx, ry = px[q] - hx[e, i], py[q] - hy[e, i]
    v = _close(np.sqrt(rx * rx + ry * ry) - radius[e, j], vision)
    base = ((e * s + i) * k + _sector(rx, ry, c[e, i], sn[e, i], width, k)) * OBS_CHANNELS
    enemy = (j != i) & (v > 0)
    mine = (j == i) & (l >= cfg.neck) & (v > 0)
    np.maximum.at(flat, base[enemy] + 1, v[enemy])
    np.maximum.at(flat, base[enemy] + 2, radius[e, j][enemy] / SIZE_SCALE)
    np.maximum.at(flat, base[mine] + 3, v[mine])

    # 4: food value per sector
    fe, ff = np.nonzero(state.food_active)
    fx, fy = state.food_pos[fe, ff, 0], state.food_pos[fe, ff, 1]
    q, i = near_pairs(fe, fx, fy, heads, alive, vision)
    e = fe[q]
    rx, ry = fx[q] - hx[e, i], fy[q] - hy[e, i]
    seen = np.sqrt(rx * rx + ry * ry) < vision
    food = np.zeros((n, s, k))
    idx = ((e * s + i) * k + _sector(rx, ry, c[e, i], sn[e, i], width, k))[seen]
    np.add.at(food.reshape(-1), idx, state.food_value[e, ff[q]][seen])
    sec[..., 4] = np.minimum(food / 10.0, 1.0)

    # 5-7: nearest enemy head per sector (ties: lower slot) and its relative heading
    for j in range(s):
        rx, ry = hx[:, j:j + 1] - hx, hy[:, j:j + 1] - hy
        v = _close(np.sqrt(rx * rx + ry * ry) - radius[:, j:j + 1], vision)
        ok = alive & alive[:, j:j + 1] & (slots != j) & (v > 0)
        ee, ii = np.nonzero(ok)
        qq = _sector(rx[ee, ii], ry[ee, ii], c[ee, ii], sn[ee, ii], width, k)
        better = v[ee, ii] > sec[ee, ii, qq, 5]
        ee, ii, qq = ee[better], ii[better], qq[better]
        sec[ee, ii, qq, 5] = v[ee, ii]
        dh = state.heading[ee, j] - state.heading[ee, ii]
        sec[ee, ii, qq, 6] = np.cos(dh)
        sec[ee, ii, qq, 7] = np.sin(dh)

    # scalars
    sc = np.zeros((n, s, NUM_SCALARS))
    sc[..., 0] = np.log(np.where(alive, state.mass, 1.0)) / np.log(cfg.max_mass)
    sc[..., 1] = radius / SIZE_SCALE
    sc[..., 2] = state.boosting
    sc[..., 3] = state.mass >= cfg.boost_min_mass
    dist = np.sqrt(hx * hx + hy * hy)
    sc[..., 4] = dist / cfg.arena_radius
    away = dist > 1e-9
    safe = np.where(away, dist, 1.0)
    bx, by = -hx / safe, -hy / safe
    sc[..., 5] = np.where(away, bx * c + by * sn, 1.0)
    sc[..., 6] = np.where(away, by * c - bx * sn, 0.0)

    obs = np.concatenate([sec.reshape(n, s, -1), sc], axis=-1)
    obs[~alive] = 0.0
    return obs.astype(np.float32)
