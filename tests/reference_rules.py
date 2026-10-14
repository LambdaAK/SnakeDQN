"""Slow, readable Battlesnake rules for a single game, used to check the vectorized rules."""

from collections import Counter
from dataclasses import dataclass

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIN_FOOD = 1
SPAWN_CHANCE = 0.15


@dataclass
class RefSnake:
    body: list
    health: int
    alive: bool


@dataclass
class RefGame:
    board_size: int
    snakes: list
    food: set
    max_health: int
    turn: int
    over: bool


def from_state(state, env):
    snakes = []
    for k in range(state.num_snakes):
        length = int(state.length[env, k])
        segments = [tuple(int(v) for v in p) for p in state.body[env, k, :length]]
        snakes.append(RefSnake(segments, int(state.health[env, k]), bool(state.alive[env, k])))
    food = {(int(r), int(c)) for r, c in zip(*state.food[env].nonzero())}
    return RefGame(state.board_size, snakes, food, state.max_health,
                   int(state.turn[env]), bool(state.over[env]))


def ref_step(game, actions, spawn_u):
    if game.over:
        return
    b = game.board_size
    movers = [i for i, snake in enumerate(game.snakes) if snake.alive]

    for i in movers:
        snake = game.snakes[i]
        dr, dc = MOVES[int(actions[i])]
        head = (snake.body[0][0] + dr, snake.body[0][1] + dc)
        snake.body = [head] + snake.body[:-1]
        snake.health -= 1

    eaten = set()
    for i in movers:
        snake = game.snakes[i]
        if snake.body[0] in game.food:
            snake.health = game.max_health
            snake.body.append(snake.body[-1])
            eaten.add(snake.body[0])
    game.food -= eaten

    if len(game.food) < MIN_FOOD or spawn_u[0] < SPAWN_CHANCE:
        occupied = {seg for i in movers for seg in game.snakes[i].body}
        free = [(r, c) for r in range(b) for c in range(b)
                if (r, c) not in game.food and (r, c) not in occupied]
        if free:
            game.food.add(free[int(spawn_u[1] * len(free))])

    def on_board(p):
        return 0 <= p[0] < b and 0 <= p[1] < b

    survivors = [i for i in movers
                 if game.snakes[i].health > 0 and on_board(game.snakes[i].body[0])]
    eliminated = {i for i in movers if i not in survivors}
    body_cells = Counter(seg for i in survivors for seg in game.snakes[i].body[1:])
    for i in survivors:
        head = game.snakes[i].body[0]
        if head in body_cells:
            eliminated.add(i)
        for k in survivors:
            other = game.snakes[k]
            if k != i and other.body[0] == head and len(other.body) >= len(game.snakes[i].body):
                eliminated.add(i)
    for i in eliminated:
        game.snakes[i].alive = False

    game.turn += 1
    n_alive = sum(snake.alive for snake in game.snakes)
    game.over = n_alive == 0 if len(game.snakes) == 1 else n_alive <= 1
