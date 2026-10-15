"""Every constant of the slither game in one place. Units: world units and physics ticks."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SlitherConfig:
    arena_radius: float = 300.0
    tick_hz: int = 30
    ticks_per_decision: int = 3
    max_turn: float = 0.15            # radians per tick
    start_mass: float = 10.0
    max_mass: float = 500.0
    base_length: float = 20.0         # body points = floor(base_length + length_per_mass * mass)
    length_per_mass: float = 2.0
    base_radius: float = 2.0          # radius = base_radius + radius_per_sqrt_mass * sqrt(mass)
    radius_per_sqrt_mass: float = 0.5
    body_capacity: int = 1024         # ring buffer size; must exceed the longest body + 2
    boost_min_mass: float = 15.0
    boost_cost: float = 0.2           # mass per boosting tick
    boost_drop_every: int = 5         # ticks of continuous boost per dropped pellet
    boost_drop_value: float = 0.8
    eat_margin: float = 3.0
    food_capacity: int = 1024
    natural_food_target: int = 200
    natural_food_per_tick: int = 2
    natural_food_value: float = 1.0
    food_spawn_margin: float = 10.0
    death_drop_fraction: float = 0.8
    death_drop_max_pellets: int = 64
    death_drop_jitter: float = 2.0
    respawn_delay: int = 30
    respawn_candidates: int = 16
    respawn_margin: float = 60.0
    respawn_clearance: float = 40.0
    vision: float = 60.0
    sectors: int = 32
    neck: int = 10                    # own-body points closer to the head than this are not observed

    def to_dict(self) -> dict:
        return asdict(self)


def u_layout(num_slots: int, cfg: SlitherConfig) -> dict[str, int]:
    """Offsets into one arena's row of the per-tick uniform array `u`:
    [respawn: per slot, `respawn_candidates` (radius, angle) pairs then a heading]
    [drops:   per slot, `death_drop_max_pellets` (x, y) jitter pairs]
    [food:    `natural_food_per_tick` (radius, angle) pairs]"""
    drops = num_slots * (2 * cfg.respawn_candidates + 1)
    food = drops + num_slots * cfg.death_drop_max_pellets * 2
    return {"respawn": 0, "drops": drops, "food": food,
            "size": food + 2 * cfg.natural_food_per_tick}
