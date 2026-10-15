"""Deliberately simple slither rules: plain Python, one arena, one snake at a time.

Works on snapshot dicts (snake_rl.slither.rules.snapshot). The batched engine must
reproduce it tick for tick, and web/slither/rules.js is a line-by-line port of it, so keep
every arithmetic expression in the same form and order as the other two.
"""

import math

from snake_rl.slither.config import u_layout


def target_length(mass, cfg):
    return math.floor(cfg.base_length + cfg.length_per_mass * mass)


def body_radius(mass, cfg):
    return cfg.base_radius + cfg.radius_per_sqrt_mass * math.sqrt(mass)


def step_tick(snap, turns, boosts, u, cfg):
    """Advance one arena by one physics tick, in place (spec §2 steps 1-7)."""
    snakes, food = snap["snakes"], snap["food"]
    count = len(snakes)
    lay = u_layout(count, cfg)
    alive = [s["alive"] for s in snakes]

    # 1. steer
    for k, s in enumerate(snakes):
        if alive[k]:
            s["heading"] = s["heading"] + min(max(turns[k], -cfg.max_turn), cfg.max_turn)

    # 2. boost, move, trim
    boost_drops = []
    for k, s in enumerate(snakes):
        b = alive[k] and boosts[k] and s["mass"] >= cfg.boost_min_mass
        if b:
            s["mass"] = s["mass"] - cfg.boost_cost
        s["boost_ticks"] = s["boost_ticks"] + 1 if b else 0
        s["boosting"] = b
        if not alive[k]:
            continue
        for _ in range(2 if b else 1):
            hx, hy = s["body"][0]
            s["body"].insert(0, [hx + math.cos(s["heading"]), hy + math.sin(s["heading"])])
        del s["body"][min(len(s["body"]), target_length(s["mass"], cfg)):]
        if b and s["boost_ticks"] % cfg.boost_drop_every == 0:
            tx, ty = s["body"][-1]
            boost_drops.append([tx, ty, cfg.boost_drop_value, False])

    # 3. deaths
    radius = [body_radius(s["mass"], cfg) for s in snakes]
    died, killer, kills = [False] * count, [-1] * count, [0] * count
    for i in range(count):
        if not alive[i]:
            continue
        hx, hy = snakes[i]["body"][0]
        wall = math.sqrt(hx * hx + hy * hy) + radius[i] > cfg.arena_radius
        head_on, first = False, -1
        for j in range(count):
            if j == i or not alive[j]:
                continue
            t = radius[i] + radius[j]
            ox, oy = snakes[j]["body"][0]
            dx, dy = hx - ox, hy - oy
            if dx * dx + dy * dy < t * t:
                head_on = True
            if first < 0:
                for px, py in snakes[j]["body"][1:]:
                    dx, dy = px - hx, py - hy
                    if dx * dx + dy * dy < t * t:
                        first = j
                        break
        died[i] = wall or head_on or first >= 0
        killer[i] = first if first >= 0 and not head_on else -1
    for j in killer:
        if j >= 0:
            kills[j] += 1

    # 4. eat
    eaten = [0.0] * count
    for f, item in enumerate(food):
        if item is None:
            continue
        best, best_d2 = -1, math.inf
        for k in range(count):
            if not alive[k] or died[k]:
                continue
            hx, hy = snakes[k]["body"][0]
            dx, dy = item[0] - hx, item[1] - hy
            d2 = dx * dx + dy * dy
            reach = radius[k] + cfg.eat_margin
            if d2 < reach * reach and d2 < best_d2:
                best, best_d2 = k, d2
        if best >= 0:
            eaten[best] += item[2]
            food[f] = None
    for k, s in enumerate(snakes):
        s["mass"] = min(s["mass"] + eaten[k], cfg.max_mass)

    # 5. death drops
    death_drops = []
    most = cfg.death_drop_max_pellets
    jitter = 2.0 * cfg.death_drop_jitter
    for k, s in enumerate(snakes):
        if not died[k]:
            continue
        length, m = len(s["body"]), s["mass"]
        n = max(min(math.ceil(m / 2), most), 1)
        value = cfg.death_drop_fraction * m / n
        for q in range(n):
            px, py = s["body"][(q * length) // n]
            base = lay["drops"] + (k * most + q) * 2
            death_drops.append([px + (u[base] - 0.5) * jitter, py + (u[base + 1] - 0.5) * jitter,
                                value, False])
        s.update(alive=False, body=[], respawn=cfg.respawn_delay, boosting=False, boost_ticks=0)

    # 6. respawn slots that were dead when the tick started
    for k, s in enumerate(snakes):
        if s["enabled"] and not alive[k]:
            s["respawn"] -= 1
            if s["respawn"] <= 0:
                respawn(snap, k, u, cfg)

    # 7. new pellets
    natural = sum(1 for item in food if item is not None and item[3])
    need = max(0, min(cfg.natural_food_target - natural, cfg.natural_food_per_tick))
    fresh = boost_drops + death_drops
    for q in range(need):
        rr = (cfg.arena_radius - cfg.food_spawn_margin) * math.sqrt(u[lay["food"] + 2 * q])
        th = 2.0 * math.pi * u[lay["food"] + 2 * q + 1]
        fresh.append([rr * math.cos(th), rr * math.sin(th), cfg.natural_food_value, True])
    free = [f for f, item in enumerate(food) if item is None]
    for f, item in zip(free, fresh):
        food[f] = item
    snap["tick"] += 1
    return {"died": died, "killer": killer, "kills": kills, "eaten": eaten}


def respawn(snap, k, u, cfg):
    """Place slot k at the first candidate clear of every live head, else the farthest one."""
    snakes = snap["snakes"]
    c = cfg.respawn_candidates
    base = u_layout(len(snakes), cfg)["respawn"] + k * (2 * c + 1)
    clearance2 = cfg.respawn_clearance * cfg.respawn_clearance
    candidates = []
    for q in range(c):
        rr = (cfg.arena_radius - cfg.respawn_margin) * math.sqrt(u[base + 2 * q])
        th = 2.0 * math.pi * u[base + 2 * q + 1]
        x, y = rr * math.cos(th), rr * math.sin(th)
        nearest = math.inf
        for other in snakes:
            if other["alive"]:
                ox, oy = other["body"][0]
                dx, dy = x - ox, y - oy
                nearest = min(nearest, dx * dx + dy * dy)
        candidates.append((x, y, nearest))
    pick = next((q for q in range(c) if candidates[q][2] >= clearance2), None)
    if pick is None:
        pick = max(range(c), key=lambda q: candidates[q][2])  # first of equal maxima
    x, y, _ = candidates[pick]
    h = 2.0 * math.pi * u[base + 2 * c]
    cos_h, sin_h = math.cos(h), math.sin(h)
    body = [[x - i * cos_h, y - i * sin_h] for i in range(target_length(cfg.start_mass, cfg))]
    snakes[k].update(body=body, heading=h, mass=cfg.start_mass, alive=True, boosting=False,
                     boost_ticks=0, respawn=0)


def observe(snap, k, cfg):
    """The spec §3 observation of snake k, as a list of floats (zeros if it is dead)."""
    K, V = cfg.sectors, cfg.vision
    width = 2.0 * math.pi / K
    out = [0.0] * (K * 8 + 7)
    snakes = snap["snakes"]
    me = snakes[k]
    if not me["alive"]:
        return out
    hx, hy = me["body"][0]
    h = me["heading"]
    c, sn = math.cos(h), math.sin(h)
    radius = [body_radius(s["mass"], cfg) for s in snakes]

    def close(d):
        return min(max(1.0 - d / V, 0.0), 1.0)

    def sector(rx, ry):
        x = rx * c + ry * sn
        y = ry * c - rx * sn
        return math.floor((math.atan2(y, x) + math.pi + width / 2) / width) % K

    for q in range(K):
        th = h + (-math.pi + q * width)
        dx, dy = math.cos(th), math.sin(th)
        b = hx * dx + hy * dy
        cc = hx * hx + hy * hy - cfg.arena_radius * cfg.arena_radius
        out[q * 8] = close(-b + math.sqrt(max(b * b - cc, 0.0)))

    for j, other in enumerate(snakes):
        if not other["alive"]:
            continue
        for idx, (px, py) in enumerate(other["body"]):
            if j == k and idx < cfg.neck:
                continue
            rx, ry = px - hx, py - hy
            v = close(math.sqrt(rx * rx + ry * ry) - radius[j])
            if v <= 0:
                continue
            q = sector(rx, ry)
            if j == k:
                out[q * 8 + 3] = max(out[q * 8 + 3], v)
            else:
                out[q * 8 + 1] = max(out[q * 8 + 1], v)
                out[q * 8 + 2] = max(out[q * 8 + 2], radius[j] / 15.0)

    food = [0.0] * K
    for item in snap["food"]:
        if item is None:
            continue
        rx, ry = item[0] - hx, item[1] - hy
        if math.sqrt(rx * rx + ry * ry) < V:
            food[sector(rx, ry)] += item[2]
    for q in range(K):
        out[q * 8 + 4] = min(food[q] / 10.0, 1.0)

    for j, other in enumerate(snakes):
        if j == k or not other["alive"]:
            continue
        ox, oy = other["body"][0]
        rx, ry = ox - hx, oy - hy
        v = close(math.sqrt(rx * rx + ry * ry) - radius[j])
        if v <= 0:
            continue
        q = sector(rx, ry)
        if v > out[q * 8 + 5]:
            out[q * 8 + 5] = v
            out[q * 8 + 6] = math.cos(other["heading"] - h)
            out[q * 8 + 7] = math.sin(other["heading"] - h)

    base = K * 8
    out[base] = math.log(me["mass"]) / math.log(cfg.max_mass)
    out[base + 1] = radius[k] / 15.0
    out[base + 2] = 1.0 if me["boosting"] else 0.0
    out[base + 3] = 1.0 if me["mass"] >= cfg.boost_min_mass else 0.0
    dist = math.sqrt(hx * hx + hy * hy)
    out[base + 4] = dist / cfg.arena_radius
    if dist > 1e-9:
        bx, by = -hx / dist, -hy / dist
        out[base + 5] = bx * c + by * sn
        out[base + 6] = by * c - bx * sn
    else:
        out[base + 5] = 1.0
    return out
