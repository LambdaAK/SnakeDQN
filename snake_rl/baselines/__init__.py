"""Scripted baseline bots and a factory to build them by name."""

from snake_rl.baselines.hamiltonian import HamiltonianBot, hamiltonian_cycle
from snake_rl.baselines.simple import GreedyBot, RandomBot, SafeRandomBot

_BOTS = {
    "random": RandomBot,
    "safe_random": SafeRandomBot,
    "greedy": GreedyBot,
}
BOT_NAMES = tuple(_BOTS)


def make_bot(name: str, seed: int | None = None, board_size: int = 11):
    if name == "hamiltonian":
        return HamiltonianBot(board_size)
    if name not in _BOTS:
        raise ValueError(f"unknown bot {name!r}; choose from {sorted(_BOTS) + ['hamiltonian']}")
    return _BOTS[name](seed)
