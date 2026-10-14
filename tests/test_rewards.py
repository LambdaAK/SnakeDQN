import numpy as np
import pytest

from snake_rl.env.rewards import food, get_reward, winloss
from snake_rl.env.rules import StepEvents


def events(ate, died, game_over, winner):
    died = np.array(died)
    return StepEvents(
        ate=np.array(ate), died=died, alive_before=np.ones_like(died),
        game_over=np.array(game_over), winner=np.array(winner),
    )


def test_winloss_win_and_loss():
    e = events(ate=[[False, False]], died=[[True, False]], game_over=[True], winner=[1])
    assert winloss(e).tolist() == [[-1.0, 1.0]]


def test_winloss_draw_gives_zero():
    e = events(ate=[[False, False]], died=[[True, True]], game_over=[True], winner=[-1])
    assert winloss(e).tolist() == [[0.0, 0.0]]


def test_winloss_mid_game_death():
    e = events(ate=[[False, False, False]], died=[[False, True, False]],
               game_over=[False], winner=[-1])
    assert winloss(e).tolist() == [[0.0, -1.0, 0.0]]


def test_food_reward():
    e = events(ate=[[True], [False], [True]], died=[[False], [True], [True]],
               game_over=[False, True, True], winner=[-1, -1, -1])
    assert food(e).tolist() == [[1.0], [-1.0], [0.0]]


def test_get_reward():
    assert get_reward("food") is food
    with pytest.raises(ValueError):
        get_reward("nope")
