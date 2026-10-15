from snake_rl.slither.bots import SlitherBot
from snake_rl.slither.eval import evaluate_vs_bot


def run(policy_bot, opponent, arenas=8, decisions=100, seed=3):
    return evaluate_vs_bot(SlitherBot(policy_bot, seed=seed + 10), opponent, arenas=arenas,
                           decisions=decisions, warmup=10, seed=seed, name=policy_bot)


def test_result_is_well_formed_and_deterministic():
    a, b = run("wander", "wander"), run("wander", "wander")
    assert a == b
    assert 0 <= a.mass_share <= 1 and a.mass_share_ci[0] <= a.mass_share <= a.mass_share_ci[1]
    assert a.mean_mass > 0 and a.mean_mass_ci[0] <= a.mean_mass <= a.mean_mass_ci[1]
    assert a.kills_per_death_ci[0] <= a.kills_per_death <= a.kills_per_death_ci[1]
    assert set(a.to_dict()) >= {"policy", "bot", "mass_share", "kills_per_death", "mean_life"}


def test_bot_strength_ordering():
    """Spec §4: hunter beats forager beats wander, measured as mass share above the fair 1/8."""
    assert run("forager", "wander", arenas=16, decisions=400).mass_share > 0.125
    assert run("hunter", "forager", arenas=16, decisions=400).mass_share > 0.125
