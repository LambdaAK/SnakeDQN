from snake_rl.env.rules import GameState, action_mask, blocked_cells


def test_walls_and_neck_are_masked():
    state = GameState.from_layout(7, [[(0, 0), (1, 0), (2, 0)]])
    assert action_mask(state)[0, 0].tolist() == [False, False, False, True]


def test_vacating_tail_is_allowed():
    state = GameState.from_layout(7, [[(2, 2), (2, 3), (3, 3), (3, 2)]])
    assert action_mask(state)[0, 0, 1]  # down into the tail


def test_duplicated_tail_is_blocked():
    state = GameState.from_layout(7, [[(2, 2), (2, 3), (3, 3), (3, 2), (3, 2)]])
    assert not action_mask(state)[0, 0, 1]
    assert blocked_cells(state)[0, 3, 2]


def test_other_snakes_body_is_blocked_but_its_tail_is_not():
    state = GameState.from_layout(
        7, [[(2, 3), (1, 3), (0, 3)], [(3, 4), (3, 3), (3, 2)]]
    )
    mask = action_mask(state)
    assert not mask[0, 0, 1]  # (3, 3) is snake 1's body
    blocked = blocked_cells(state)
    assert blocked[0, 3, 4] and blocked[0, 3, 3] and not blocked[0, 3, 2]


def test_dead_snakes_and_finished_games_have_empty_masks():
    state = GameState.from_layout(
        7, [[(3, 3), (4, 3), (5, 3)], [(1, 1), (1, 2), (1, 3)]]
    )
    state.alive[0, 1] = False
    mask = action_mask(state)
    assert not mask[0, 1].any() and mask[0, 0].any()
    state.over[0] = True
    assert not action_mask(state).any()
