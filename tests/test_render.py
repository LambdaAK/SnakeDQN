from snake_rl.env.render import render
from snake_rl.env.rules import GameState


def layout():
    return GameState.from_layout(
        7, [[(3, 3), (4, 3), (5, 3)], [(1, 1), (1, 2), (1, 3)]], food=[(0, 6)]
    )


def test_plain_render_layout():
    text = render(layout(), color=False)
    lines = text.splitlines()
    assert len(lines) == 1 + 2 + 7 + 2
    assert lines[0] == "Turn 0"
    assert "\033" not in text
    assert text.count("██") == 2 and text.count("▓▓") == 4 and text.count("◆◆") == 1
    assert "Snake 1: length 3, health 100" in text


def test_dead_snake_not_drawn():
    state = layout()
    state.alive[0, 1] = False
    text = render(state, color=False)
    assert text.count("██") == 1
    assert "(dead)" in text


def test_color_render_has_escape_codes():
    assert "\033[" in render(layout())
