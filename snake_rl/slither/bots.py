"""Scripted slither bots (spec §4). They read the same observation as the network, so they
run batched over any leading shape: bot(obs[..., 263], mask[..., 18]) -> actions[...].

Each bot scores the 32 sectors (food is good, danger is bad, turning costs a little, plus
noise) and steers toward the best one. The hunter also boosts to cut across the path of a
smaller snake whose head is ahead of it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snake_rl.slither.actions import steer_bin
from snake_rl.slither.config import SlitherConfig
from snake_rl.slither.observations import OBS_CHANNELS, SIZE_SCALE

BOT_NAMES = ("wander", "forager", "hunter")
HUNT_MAX_DANGER = 0.5  # skip the chase when the aimed sector's wall/body danger is above this
HUNT_MIN_CLOSE = 0.4  # only chase heads at least this close (0 = edge of vision, 1 = touching)
HUNT_CONE = np.pi / 2  # only chase heads within this angle of straight ahead
HUNT_LEAD = 50.0  # aim this far ahead of the prey's head, so we cross its path


@dataclass(frozen=True)
class BotStyle:
    food: float    # weight on a sector's food channel
    danger: float  # weight on a sector's danger (wall, enemy body, enemy head)
    turn: float    # cost per radian of turning
    noise: float   # random jitter on sector scores
    hunt: bool     # boost across smaller snakes' paths


STYLES = {
    "wander": BotStyle(food=0.0, danger=4.0, turn=0.3, noise=1.0, hunt=False),
    "forager": BotStyle(food=3.0, danger=4.0, turn=0.2, noise=0.05, hunt=False),
    "hunter": BotStyle(food=3.0, danger=4.0, turn=0.2, noise=0.05, hunt=True),
}


class SlitherBot:
    def __init__(self, name: str, seed: int | None = None, cfg: SlitherConfig | None = None):
        if name not in STYLES:
            raise ValueError(f"unknown bot {name!r}; choose from {BOT_NAMES}")
        self.name = name
        self.style = STYLES[name]
        self.cfg = cfg or SlitherConfig()
        self.rng = np.random.default_rng(seed)
        k = self.cfg.sectors
        self.phi = -np.pi + np.arange(k) * 2.0 * np.pi / k  # sector centre angles (16 = ahead)

    def __call__(self, obs, mask) -> np.ndarray:
        obs = np.asarray(obs, np.float64)
        mask = np.asarray(mask, bool)
        k = self.cfg.sectors
        lead = obs.shape[:-1]
        sec = obs[..., :k * OBS_CHANNELS].reshape(*lead, k, OBS_CHANNELS)
        danger = np.maximum(np.maximum(sec[..., 0], sec[..., 1]), sec[..., 5])
        # Bodies are wide, so a sector is nearly as dangerous as its neighbours.
        danger = np.maximum(danger, 0.8 * np.maximum(np.roll(danger, 1, -1), np.roll(danger, -1, -1)))
        st = self.style
        score = (st.food * sec[..., 4] - st.danger * danger - st.turn * np.abs(self.phi)
                 + st.noise * self.rng.random(danger.shape))
        target = self.phi[np.argmax(score, axis=-1)]
        boost = np.zeros(lead, bool)
        if st.hunt:
            target, boost = self._hunt(obs, sec, target, mask)
        return (steer_bin(target, self.cfg) * 2 + boost).astype(np.int64)

    def _hunt(self, obs, sec, target, mask):
        k = self.cfg.sectors
        my_size = obs[..., k * OBS_CHANNELS + 1]
        prey = ((np.abs(self.phi) < HUNT_CONE) & (sec[..., 5] > HUNT_MIN_CLOSE)
                & (sec[..., 2] < my_size[..., None]) & (sec[..., 0] < 0.5))
        has = prey.any(axis=-1)
        pick = np.argmax(np.where(prey, sec[..., 5], -1.0), axis=-1)[..., None]
        near = np.take_along_axis(sec[..., 5], pick, -1)[..., 0]
        size = np.take_along_axis(sec[..., 2], pick, -1)[..., 0]
        cos_h = np.take_along_axis(sec[..., 6], pick, -1)[..., 0]
        sin_h = np.take_along_axis(sec[..., 7], pick, -1)[..., 0]
        ang = self.phi[pick[..., 0]]
        dist = (1.0 - near) * self.cfg.vision + size * SIZE_SCALE
        aim = np.arctan2(dist * np.sin(ang) + HUNT_LEAD * sin_h, dist * np.cos(ang) + HUNT_LEAD * cos_h)
        # Don't cut across a path that runs through a wall or enemy body.
        aim_sector = np.round((aim + np.pi) / (2.0 * np.pi / k)).astype(np.int64) % k
        blocked = np.take_along_axis(np.maximum(sec[..., 0], sec[..., 1]), aim_sector[..., None], -1)[..., 0]
        has = has & (blocked < HUNT_MAX_DANGER)
        return np.where(has, aim, target), has & mask[..., 1]
