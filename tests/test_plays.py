import numpy as np
import pandas as pd
import pytest

from nflfp import plays as P
from nflfp import yards as Y

TEAMS = [f"T{i}" for i in range(10)]
MU = dict(rush_att=25.0, pass_att=36.0, rush_ypc=4.5, pass_ypa=6.4)


def synth(seed=0, weeks=12, season=2026, gamma=0.0, factors=None):
    """Noiseless games: plays and yards-per-play follow MU*OFF*DEF; yards = plays * yards per play."""
    rng = np.random.default_rng(seed)
    F = factors or {k: {q: dict(zip(TEAMS, np.exp(rng.normal(0, .15, len(TEAMS))))) for q in P.QUANTS} for k in ("off", "dfn")}
    rows = []
    for w in range(1, weeks + 1):
        perm = rng.permutation(len(TEAMS))
        for a, b in zip(perm[::2], perm[1::2]):
            spread = float(rng.integers(-10, 11))
            for o, d, fav in ((TEAMS[a], TEAMS[b], spread), (TEAMS[b], TEAMS[a], -spread)):
                v = {q: MU[q] * F["off"][q][o] * F["dfn"][q][d] for q in P.QUANTS}
                ra, pa = v["rush_att"] + gamma * fav, v["pass_att"] - gamma * fav
                rows.append(dict(season=season, week=w, off_team=o, def_team=d, fav=fav, rush_att=ra, pass_att=pa,
                                 rush_yds=ra * v["rush_ypc"], pass_yds=pa * v["pass_ypa"]))
    g = pd.DataFrame(rows)
    return g.drop(columns="fav"), g[["season", "week", "off_team", "def_team", "fav"]], F


def test_team_game_plays_counts_plays_and_yards():
    pbp = pd.DataFrame([
        dict(season=2026, week=1, posteam="A", defteam="B", play_type="run", rushing_yards=5.0, passing_yards=np.nan),
        dict(season=2026, week=1, posteam="A", defteam="B", play_type="run", rushing_yards=-1.0, passing_yards=np.nan),
        dict(season=2026, week=1, posteam="A", defteam="B", play_type="pass", rushing_yards=np.nan, passing_yards=20.0),
        dict(season=2026, week=1, posteam="A", defteam="B", play_type="pass", rushing_yards=np.nan, passing_yards=0.0),   # incompletion / sack: a play, no yards
        dict(season=2026, week=1, posteam="B", defteam="A", play_type="run", rushing_yards=10.0, passing_yards=np.nan)])
    g = P.team_game_plays(pbp).set_index("off_team")
    assert g.loc["A", ["rush_att", "rush_yds", "pass_att", "pass_yds"]].tolist() == [2, 4.0, 2, 20.0]
    assert g.loc["B", ["rush_att", "rush_yds", "pass_att", "pass_yds"]].tolist() == [1, 10.0, 0, 0.0]


def test_efficiency_is_weighted_by_plays_and_pooled():
    g = pd.DataFrame([dict(rush_att=30, rush_yds=120.0), dict(rush_att=1, rush_yds=40.0)])
    ok, y, w, mu = P._series(g, "rush_ypc")
    assert mu == pytest.approx(160.0 / 31.0)                       # pooled: total yards / total plays, not the mean of 4.0 and 40.0
    assert w[0] / w[1] == pytest.approx(30.0) and ok.all()
    ok, y, w, mu = P._series(pd.DataFrame([dict(rush_att=0, rush_yds=0.0), dict(rush_att=10, rush_yds=45.0)]), "rush_ypc")
    assert ok.tolist() == [False, True] and mu == pytest.approx(4.5)   # a game with no plays has no efficiency


def test_plain_fit_recovers_factors_and_expected_yards_are_plays_times_efficiency():
    g, _, F = synth()
    r = P.fit_ratings(g, TEAMS, P.PlaysConfig())
    for q in P.QUANTS:
        true_o = pd.Series(F["off"][q]) / pd.Series(F["off"][q]).mean()
        np.testing.assert_allclose(r.off[q].reindex(TEAMS), true_o.reindex(TEAMS), rtol=0.04)
    e = r.expected("T0", "T1")
    assert e["rush_yds"] == pytest.approx(e["rush_att"] * e["rush_ypc"]) and e["pass_yds"] == pytest.approx(e["pass_att"] * e["pass_ypa"])


def test_ridge_limits_and_target():
    g, _, _ = synth()
    prev, _, _ = synth(seed=7, season=2025)
    cfg = P.PlaysConfig()
    prior = P.build_prior(prev, TEAMS)
    t_off, _ = prior.targets(0.3)
    pd.testing.assert_frame_equal(t_off, 0.3 * prior.off_factor + 0.7)
    plain = P.fit_ratings(g, TEAMS, cfg)
    zero = P.fit_ratings(g, TEAMS, P.PlaysConfig(strength_volume=0.0, strength_efficiency=0.0), prior, ridge=True)
    pd.testing.assert_frame_equal(plain.off, zero.off, atol=1e-8)
    huge = P.fit_ratings(g, TEAMS, P.PlaysConfig(strength_volume=1e9, strength_efficiency=1e9), prior, ridge=True)
    np.testing.assert_allclose(huge.off.to_numpy(), (t_off / t_off.mean()).to_numpy(), rtol=1e-4)
    # volume and efficiency use their own strengths
    only_vol = P.fit_ratings(g, TEAMS, P.PlaysConfig(strength_volume=1e9, strength_efficiency=0.0), prior, ridge=True)
    np.testing.assert_allclose(only_vol.off["rush_ypc"].to_numpy(), plain.off["rush_ypc"].to_numpy(), atol=1e-8)
    np.testing.assert_allclose(only_vol.off["rush_att"].to_numpy(), (t_off["rush_att"] / t_off["rush_att"].mean()).to_numpy(), rtol=1e-4)


def test_week_one_is_last_season_only():
    prev, prev_fx, _ = synth(seed=1, season=2025)
    _, cur_fx, _ = synth(seed=2, season=2026)
    out = P.project_week(prev, pd.concat([prev_fx, cur_fx], ignore_index=True), 2026, 1)
    prior = P.build_prior(prev, TEAMS)
    assert out[Y.OUT_COLS].notna().all().all() and len(out) == len(TEAMS)
    np.testing.assert_allclose(out["rush_plain"], prior.mu["rush_att"] * prior.mu["rush_ypc"])
    t_off, t_dfn = prior.targets(0.3)
    r = out.iloc[0]
    want = prior.mu["rush_att"] * t_off.loc[r.off_team, "rush_att"] * t_dfn.loc[r.def_team, "rush_att"] * prior.mu["rush_ypc"] * t_off.loc[r.off_team, "rush_ypc"] * t_dfn.loc[r.def_team, "rush_ypc"]
    assert r["rush_ridge"] == pytest.approx(want, rel=1e-9)


def test_volume_lines_transfer_moves_plays_and_relative_mode_fits():
    g, fx, _ = synth(seed=5, gamma=0.4)
    resid = P._volume_residuals(g, fx, P.fit_ratings(g, TEAMS, P.PlaysConfig()))
    adj = P.fit_volume_lines(resid, "transfer")
    assert adj.gamma == pytest.approx(0.4, rel=0.3)
    r, p = adj.apply(25.0, 36.0, 7.0)
    assert r + p == pytest.approx(61.0) and r > 25.0 and p < 36.0        # plays move from passing to running
    rel = P.fit_volume_lines(resid, "relative")
    assert rel.slopes["rush"] > 0 > rel.slopes["pass"]
    with pytest.raises(ValueError):
        P.fit_volume_lines(resid, "free")


def test_eight_numbers_and_no_look_ahead():
    prev, prev_fx, _ = synth(seed=1, season=2025)
    cur, cur_fx, _ = synth(seed=2, season=2026, gamma=0.3)
    games, fixtures = pd.concat([prev, cur], ignore_index=True), pd.concat([prev_fx, cur_fx], ignore_index=True)
    out = P.project_week(games, fixtures, 2026, 6)
    assert set(Y.OUT_COLS) <= set(out.columns) and out[Y.OUT_COLS].notna().all().all() and {"rush_att_ridge", "pass_ypa_ridge"} <= set(out.columns)
    assert (out["rush_ridge_vegas"] != out["rush_ridge"]).any()
    bad = games.copy()
    bad.loc[(bad.season == 2026) & (bad.week >= 6), ["rush_att", "pass_att", "rush_yds", "pass_yds"]] *= 1000
    pd.testing.assert_frame_equal(out[Y.OUT_COLS], P.project_week(bad, fixtures, 2026, 6)[Y.OUT_COLS])
