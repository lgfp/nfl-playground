import numpy as np
import pandas as pd
import pytest

from nflfp import yards as Y

TEAMS = [f"T{i}" for i in range(10)]


def synth(seed=0, weeks=12, off=None, dfn=None, season=2026, mu=(110.0, 230.0), beta=0.0):
    """Noiseless multiplicative games (+ an optional favourite effect: +beta*fav rush, -beta*fav pass yards)."""
    rng = np.random.default_rng(seed)
    off = off if off is not None else {s: dict(zip(TEAMS, np.exp(rng.normal(0, .2, len(TEAMS))))) for s in Y.SIDES}
    dfn = dfn if dfn is not None else {s: dict(zip(TEAMS, np.exp(rng.normal(0, .2, len(TEAMS))))) for s in Y.SIDES}
    rows = []
    for w in range(1, weeks + 1):
        perm = rng.permutation(len(TEAMS))
        for a, b in zip(perm[::2], perm[1::2]):
            spread = float(rng.integers(-10, 11))
            for o, d, fav in ((TEAMS[a], TEAMS[b], spread), (TEAMS[b], TEAMS[a], -spread)):
                rows.append(dict(season=season, week=w, off_team=o, def_team=d, fav=fav,
                                 rush_yds=mu[0] * off["rush"][o] * dfn["rush"][d] + beta * fav,
                                 pass_yds=mu[1] * off["pass"][o] * dfn["pass"][d] - beta * fav))
    g = pd.DataFrame(rows)
    return g.drop(columns="fav"), g[["season", "week", "off_team", "def_team", "fav"]], off, dfn


def test_plain_fit_recovers_known_factors():
    g, _, off, dfn = synth()
    r = Y.fit_ratings(g, TEAMS, Y.YardsConfig())
    for s in Y.SIDES:
        true_o = pd.Series(off[s]) / pd.Series(off[s]).mean()
        true_d = pd.Series(dfn[s]) / pd.Series(dfn[s]).mean()
        np.testing.assert_allclose(r.off[s].reindex(TEAMS), true_o.reindex(TEAMS), rtol=0.03)
        np.testing.assert_allclose(r.dfn[s].reindex(TEAMS), true_d.reindex(TEAMS), rtol=0.03)
    assert r.off.mean().round(6).eq(1.0).all() and r.dfn.mean().round(6).eq(1.0).all()


def test_opponent_adjustment_credits_a_hard_schedule():
    # A and B both gain 100 yards, but A only faced the leakiest defense and B only the stingiest: B is the better offense
    rows = [("A", "LEAKY", 100, 200), ("B", "TIGHT", 100, 200), ("X", "LEAKY", 150, 300), ("X", "TIGHT", 50, 100),
            ("LEAKY", "X", 100, 200), ("TIGHT", "X", 100, 200)]
    g = pd.DataFrame([dict(season=2026, week=i, off_team=o, def_team=d, rush_yds=ry, pass_yds=py) for i, (o, d, ry, py) in enumerate(rows)])
    r = Y.fit_ratings(g, ["A", "B", "X", "LEAKY", "TIGHT"], Y.YardsConfig())
    assert r.off.loc["B", "rush"] > r.off.loc["A", "rush"]


def test_ridge_limits_and_target_composition():
    g, _, _, _ = synth()
    prev, _, _, _ = synth(seed=7, season=2025)
    cfg = Y.YardsConfig(prior_weight=0.3)
    prior = Y.build_prior(prev, TEAMS, cfg)
    t_off, t_dfn = prior.targets(0.3)
    pd.testing.assert_frame_equal(t_off, 0.3 * prior.off_factor + 0.7)           # 30% last season, 70% league average (1.0)
    plain = Y.fit_ratings(g, TEAMS, cfg)
    zero = Y.fit_ratings(g, TEAMS, Y.YardsConfig(ridge_strength=0.0), prior, ridge=True)
    pd.testing.assert_frame_equal(plain.off, zero.off, atol=1e-8)                 # strength 0 == plain
    huge = Y.fit_ratings(g, TEAMS, Y.YardsConfig(ridge_strength=1e9), prior, ridge=True)
    np.testing.assert_allclose(huge.off.to_numpy(), (t_off / t_off.mean()).to_numpy(), rtol=1e-4)   # huge strength == the target
    mid = Y.fit_ratings(g, TEAMS, Y.YardsConfig(ridge_strength=8), prior, ridge=True)
    spread = lambda r: r.off["rush"].std()
    assert spread(huge) < spread(mid) < spread(plain) * 1.001                     # more ridge, less spread


def test_raw_prior_basis_is_plain_team_average_over_league_average():
    prev, _, _, _ = synth(seed=3, season=2025)
    p = Y.build_prior(prev, TEAMS, Y.YardsConfig(prior_basis="raw"))
    exp = prev.groupby("off_team").rush_yds.mean() / prev.rush_yds.mean()
    np.testing.assert_allclose(p.off_factor["rush"].reindex(exp.index), exp, rtol=1e-12)
    assert p.mu["rush"] == pytest.approx(prev.rush_yds.mean())


def test_week_one_uses_last_season_only():
    prev, prev_fx, _, _ = synth(seed=1, season=2025)
    _, cur_fx, _, _ = synth(seed=2, season=2026)
    games = prev
    fixtures = pd.concat([prev_fx, cur_fx], ignore_index=True)
    out = Y.project_week(games, fixtures, 2026, 1, Y.YardsConfig())
    prior = Y.build_prior(prev, TEAMS, Y.YardsConfig())
    assert out[Y.OUT_COLS].notna().all().all() and len(out) == len(TEAMS)
    # plain at week 1: league mean of last season, every team equal
    np.testing.assert_allclose(out["rush_plain"], prior.mu["rush"])
    # ridge at week 1: last season's mean times the target factors (normalised)
    t_off, t_dfn = prior.targets(0.3)
    r = out.iloc[0]
    assert r["rush_ridge"] == pytest.approx(prior.mu["rush"] * t_off.loc[r.off_team, "rush"] * t_dfn.loc[r.def_team, "rush"], rel=1e-9)


def test_lines_transfer_recovers_planted_effect_and_conserves_yards():
    g, fx, _, _ = synth(seed=5, beta=3.0)
    resid = Y._residuals(g, fx, Y.fit_ratings(g, TEAMS, Y.YardsConfig()))
    adj = Y.fit_lines(resid, "transfer")
    assert adj.beta == pytest.approx(3.0, rel=0.25)                               # in-sample fit absorbs a bit of the effect
    r, p = adj.apply(100.0, 230.0, 7.0)
    assert (r + p) == pytest.approx(330.0) and r > 100.0 and p < 230.0            # yards move from the pass to the run
    free = Y.fit_lines(resid, "free")
    assert free.slopes["rush"] > 0 > free.slopes["pass"]
    assert Y.fit_lines(pd.DataFrame({"fav": [], "rush_resid": [], "pass_resid": []}), "free").slopes == {"rush": 0.0, "pass": 0.0}


def test_eight_numbers_and_no_look_ahead():
    prev, prev_fx, _, _ = synth(seed=1, season=2025)
    cur, cur_fx, _, _ = synth(seed=2, season=2026, beta=2.0)
    games, fixtures = pd.concat([prev, cur], ignore_index=True), pd.concat([prev_fx, cur_fx], ignore_index=True)
    out = Y.project_week(games, fixtures, 2026, 6, Y.YardsConfig())
    assert len(Y.OUT_COLS) == 8 and set(Y.OUT_COLS) <= set(out.columns) and out[Y.OUT_COLS].notna().all().all()
    assert (out["rush_plain_vegas"] != out["rush_plain"]).any() and np.allclose(
        out["rush_plain_vegas"] + out["pass_plain_vegas"], out["rush_plain"] + out["pass_plain"])   # transfer: total unchanged
    bad = games.copy()
    bad.loc[(bad.season == 2026) & (bad.week >= 6), ["rush_yds", "pass_yds"]] *= 1000      # corrupt this week and later
    out2 = Y.project_week(bad, fixtures, 2026, 6, Y.YardsConfig())
    pd.testing.assert_frame_equal(out[Y.OUT_COLS], out2[Y.OUT_COLS])


def test_relative_lines_mode_recovers_a_planted_multiplicative_effect():
    rng = np.random.default_rng(0)
    f = rng.integers(-10, 11, 400).astype(float)
    er, ep = rng.uniform(80, 150, 400), rng.uniform(180, 280, 400)
    resid = pd.DataFrame({"fav": f, "rush_exp": er, "pass_exp": ep, "rush_resid": er * 0.012 * f, "pass_resid": ep * -0.004 * f})
    adj = Y.fit_lines(resid, "relative")
    assert adj.slopes["rush"] == pytest.approx(0.012) and adj.slopes["pass"] == pytest.approx(-0.004)
    r, p = adj.apply(100.0, 200.0, 5.0)
    assert r == pytest.approx(106.0) and p == pytest.approx(196.0)
    with pytest.raises(ValueError):
        Y.fit_lines(resid, "bogus")


def test_unbalanced_mode_recovers_beta_and_leaves_favourite_pass_alone():
    rng = np.random.default_rng(1)
    f = rng.integers(-10, 11, 600).astype(float)
    b = 1.5
    rr = b * f                                   # rush: +b*fav for everyone
    pr = np.where(f < 0, 2 * b * -f, 0.0)        # pass: +2b*|fav| for underdogs only
    resid = pd.DataFrame({"fav": f, "rush_resid": rr, "pass_resid": pr, "rush_exp": 100.0, "pass_exp": 230.0})
    adj = Y.fit_lines(resid, "unbalanced")
    assert adj.beta == pytest.approx(b)
    assert adj.apply(100.0, 230.0, 6.0) == pytest.approx((109.0, 230.0))      # favourite: +beta*fav rush, pass unchanged
    assert adj.apply(100.0, 230.0, -6.0) == pytest.approx((91.0, 248.0))      # underdog: -beta*|fav| rush, +2*beta*|fav| pass
    flat = Y.fit_lines(resid.assign(rush_resid=b * np.sign(f), pass_resid=np.where(f < 0, 2 * b, 0.0)), "unbalanced_flat")
    assert flat.beta == pytest.approx(b) and flat.apply(100.0, 230.0, 6.0) == pytest.approx((101.5, 230.0)) and flat.apply(100.0, 230.0, -1.0) == pytest.approx((98.5, 233.0))
    assert adj.as_slopes() == (b, 2 * b)
