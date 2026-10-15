"""Slither mode: a continuous, slither.io-style arena (spec 2026-10-15-slither-mode-design)."""

from snake_rl.slither.config import SlitherConfig, u_layout
from snake_rl.slither.env import SlitherEnv
from snake_rl.slither.rules import (SlitherState, TickEvents, from_snapshots, reset_arenas,
                                    snapshot, step_tick)

__all__ = ["SlitherConfig", "SlitherEnv", "SlitherState", "TickEvents", "from_snapshots",
           "reset_arenas", "snapshot", "step_tick", "u_layout"]
