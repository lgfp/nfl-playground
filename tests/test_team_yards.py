import numpy as np
import pandas as pd
import pytest

from nflfp import team_yards as TY
from nflfp import yards as Y

TEAMS = [f"T{i}" for i in range(10)]


def synth(seed=0, seasons=(2025, 2026), weeks=10, beta_rush=0.0, beta_pass=0.0, wind_pass=0.0, sigma=0.2):
    """Games + environment for several seasons; rush/pass yards follow team strengths plus planted spread and wind effects."""
    rng = np.random.default_rng(seed)
    off = {s: dict(zip(TEAMS, np.exp(rng.normal(0, sigma, len(TEAMS))))) for s in Y.SIDES}
    dfn = {s: dict(zip(TEAMS, np.exp(rng.normal(0, sigma, len(TEAMS))))) for s in Y.SIDES}
    rows = []
    for season in seasons:
        for w in range(1, weeks + 1):
            perm = rng.permutation(len(TEAMS))
            for a, b in zip(perm[::2], perm[1::2]):
                spread, wind = float(rng.integers(-10, 11)), float(rng.integers(0, 5))
                for o, d, fav in ((TEAMS[a], TEAMS[b], spread), (TEAMS[b], TEAMS[a], -spread)):
                    rows.append(dict(season=season, week=w, off_team=o, def_team=d, fav=fav, wind5=wind,
                                     rush_yds=110 * off["rush"][o] * dfn["rush"][d] + beta_rush * fav,
                                     pass_yds=230 * off["pass"][o] * dfn["pass"][d] + beta_pass * fav + wind_pass * wind))
    t = pd.DataFrame(rows)
    return t[TY.KEYS + ["rush_yds", "pass_yds"]], t[TY.KEYS + ["fav", "wind5"]]


def test_defaults_match_the_chosen_settings():
    c = TY.TeamYardsConfig()
    assert (c.rush_strength, c.rush_prior_weight, c.pass_strength, c.pass_prior_weight) == (12.0, 0.15, 8.0, 0.3)


def test_every_fixture_gets_finite_non_negative_projections_and_stages_add_up():
    games, env = synth()
    out = TY.walk_forward(games, env, [2025, 2026])
    assert len(out) == len(env[env.season.isin([2025, 2026])]) - len(TEAMS)          # 2025 week 1 has no history to go on
    assert out[["rush_exp", "pass_exp"]].notna().all().all() and (out[["rush_exp", "pass_exp"]] >= 0).all().all()
    np.testing.assert_allclose(out.rush_exp, np.maximum(out.rush_ridge + out.slope_rush_vegas * out.fav, 0.0))
    np.testing.assert_allclose(out.pass_exp, np.maximum(out.pass_wind + out.slope_pass_vegas * out.fav, 0.0))


def test_rush_ridge_average_formula():
    games, env = synth(seed=1)
    out = TY.walk_forward(games, env, [2025, 2026])
    cfg = TY.TeamYardsConfig()
    row = out[(out.season == 2026) & (out.week == 5)].iloc[0]
    cur = games[(games.season == 2026) & (games.week < 5)]
    prev = games[games.season == 2025]
    n = (cur.off_team == row.off_team).sum()
    own = cur[cur.off_team == row.off_team].rush_yds.mean()
    last = prev[prev.off_team == row.off_team].rush_yds.mean()
    mu = cur.rush_yds.mean()
    target = cfg.rush_prior_weight * last + (1 - cfg.rush_prior_weight) * mu
    assert row.rush_ridge == pytest.approx((n * own + cfg.rush_strength * target) / (n + cfg.rush_strength))


def test_week_one_is_the_target():
    games, env = synth(seed=2)
    out = TY.walk_forward(games, env, [2025, 2026])
    cfg = TY.TeamYardsConfig()
    r = out[(out.season == 2026) & (out.week == 1)].iloc[0]
    prev = games[games.season == 2025]
    want = cfg.rush_prior_weight * prev[prev.off_team == r.off_team].rush_yds.mean() + (1 - cfg.rush_prior_weight) * prev.rush_yds.mean()
    assert r.rush_ridge == pytest.approx(want)


def test_planted_spread_and_wind_effects_are_recovered():
    games, env = synth(seed=3, weeks=16, beta_rush=1.5, beta_pass=1.0, wind_pass=-8.0, sigma=0.06)   # quiet teams, many games: the planted effects stand out
    out = TY.walk_forward(games, env, [2025, 2026])
    last = out[(out.season == 2026)].iloc[-1]
    assert last.slope_rush_vegas == pytest.approx(1.5, rel=0.35)
    assert last.slope_pass_vegas == pytest.approx(1.0, rel=0.4)
    assert last.slope_pass_wind == pytest.approx(-8.0, rel=0.25)


def test_no_look_ahead_and_future_weeks_can_be_projected():
    games, env = synth(seed=4)
    ref = TY.project_week(games, env, 2026, 6)
    bad = games.copy()
    bad.loc[(bad.season == 2026) & (bad.week >= 6), ["rush_yds", "pass_yds"]] *= 1000      # corrupt the target week and later
    pd.testing.assert_frame_equal(ref[["rush_exp", "pass_exp"]], TY.project_week(bad, env, 2026, 6)[["rush_exp", "pass_exp"]])
    future = TY.project_week(games[~((games.season == 2026) & (games.week >= 6))], env, 2026, 6)   # week 6 not played yet
    pd.testing.assert_frame_equal(ref[["rush_exp", "pass_exp"]], future[["rush_exp", "pass_exp"]])
    assert future.rush_yds.isna().all() and len(future) == len(TEAMS)


def test_unknown_wind_is_treated_as_average():
    games, env = synth(seed=5, wind_pass=-6.0)
    base = TY.project_week(games, env, 2026, 7)
    unknown = env.copy()
    unknown.loc[(unknown.season == 2026) & (unknown.week == 7), "wind5"] = np.nan
    out = TY.project_week(games, unknown, 2026, 7)
    np.testing.assert_allclose(out.pass_wind, out.pass_ridge_opp)       # no wind information: no wind adjustment
    assert not np.allclose(base.pass_wind, base.pass_ridge_opp)
