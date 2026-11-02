import numpy as np
import pandas as pd
import pytest

from nflfp import model


def synthetic_games(seed=0, n_teams=12, weeks=14):
    """Games generated from a known multiplicative model (no noise) on a round-robin-ish schedule."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    off = dict(zip(teams, np.exp(rng.normal(0, .25, n_teams))))
    dfn = dict(zip(teams, np.exp(rng.normal(0, .25, n_teams))))
    mu = 100.0
    rows = []
    for w in range(1, weeks + 1):
        perm = rng.permutation(n_teams)
        for a, b in zip(perm[::2], perm[1::2]):
            for o, d in ((teams[a], teams[b]), (teams[b], teams[a])):
                rows.append(dict(off_team=o, def_team=d, week=w, rush_yds=mu * off[o] * dfn[d]))
    tg = pd.DataFrame(rows)
    for c in model.COMPONENTS:
        tg[c] = tg["rush_yds"]
    return tg, off, dfn


def test_recovers_known_strengths_when_opponents_differ():
    tg, off, dfn = synthetic_games()
    r = model.fit_ratings(tg, shrink={c: 0.0 for c in model.COMPONENTS})
    # factors are normalised to mean 1; compare up to that normalisation
    true_d = pd.Series(dfn); true_d /= true_d.mean()
    true_o = pd.Series(off); true_o /= true_o.mean()
    pd.testing.assert_series_equal(r.dfn["rush_yds"], true_d.reindex(r.dfn.index), check_names=False, atol=1e-6, rtol=1e-6)
    pd.testing.assert_series_equal(r.off["rush_yds"], true_o.reindex(r.off.index), check_names=False, atol=1e-6, rtol=1e-6)


def test_schedule_strength_is_removed():
    # Both defenses allow 150 raw, but EASY faced only the weak offense (normally 100)
    # while HARD faced only the strong offense (normally 200), so HARD is the better defense.
    games = [("STRONG", "N1", 200), ("STRONG", "N2", 200), ("WEAK", "N1", 100), ("WEAK", "N2", 100),
             ("WEAK", "EASY", 150), ("WEAK", "EASY", 150), ("STRONG", "HARD", 150), ("STRONG", "HARD", 150)]
    tg = pd.DataFrame([dict(off_team=o, def_team=d, week=i, rush_yds=y) for i, (o, d, y) in enumerate(games)])
    for c in model.COMPONENTS:
        tg[c] = tg["rush_yds"]
    r = model.fit_ratings(tg, shrink={c: 0.0 for c in model.COMPONENTS})
    assert r.raw_allowed.loc["EASY", "rush_yds"] == r.raw_allowed.loc["HARD", "rush_yds"]
    assert r.dfn.loc["HARD", "rush_yds"] < r.dfn.loc["EASY", "rush_yds"]


def test_shrinkage_pulls_toward_average():
    tg, *_ = synthetic_games()
    lo = model.fit_ratings(tg, shrink={c: 0.0 for c in model.COMPONENTS})
    hi = model.fit_ratings(tg, shrink={c: 20.0 for c in model.COMPONENTS})
    assert (hi.dfn - 1).abs().mean().mean() < (lo.dfn - 1).abs().mean().mean()


def test_player_points_sum_to_team_points():
    tg, *_ = synthetic_games()
    r = model.fit_ratings(tg)
    s = model.Scoring()
    pw = pd.DataFrame([
        dict(player_id="q", player_display_name="Q", position="QB", team="T0", week=1, carries=0, targets=0, attempts=30),
        dict(player_id="a", player_display_name="A", position="RB", team="T0", week=1, carries=15, targets=2, attempts=0),
        dict(player_id="b", player_display_name="B", position="RB", team="T0", week=1, carries=5, targets=8, attempts=0),
        dict(player_id="c", player_display_name="C", position="WR", team="T0", week=1, carries=0, targets=10, attempts=0),
    ])
    shares = model.usage_shares(pw)
    assert shares.carry_share.sum() == pytest.approx(1) and shares.target_share.sum() == pytest.approx(1)
    proj = model.project_players(r, shares, [("T0", "T1")], s)
    rush, rec = model.points(model.team_expectation(r, "T0", "T1"), s)
    assert proj.rush_pts.sum() == pytest.approx(rush)
    assert proj.rec_pts.sum() == pytest.approx(rec)
    assert proj.pass_pts.sum() == pytest.approx(model.pass_points(model.team_expectation(r, "T0", "T1"), s))
    assert proj.set_index("player").loc["Q", "pass_pts"] == pytest.approx(model.pass_points(model.team_expectation(r, "T0", "T1"), s))
    assert proj.set_index("player").loc["A", "rush_pts"] == pytest.approx(0.75 * rush)


def test_usage_uses_current_team_only():
    pw = pd.DataFrame([
        dict(player_id="a", player_display_name="A", position="RB", team="OLD", week=1, carries=20, targets=0, attempts=0),
        dict(player_id="a", player_display_name="A", position="RB", team="NEW", week=2, carries=10, targets=0, attempts=0),
        dict(player_id="b", player_display_name="B", position="RB", team="NEW", week=2, carries=10, targets=0, attempts=0),
    ])
    sh = model.usage_shares(pw).set_index("player")
    assert sh.loc["A", "team"] == "NEW" and sh.loc["A", "carry_share"] == pytest.approx(0.5)


def test_starting_qb_gets_all_passing_share():
    pw = pd.DataFrame([
        dict(player_id="s", player_display_name="S", position="QB", team="T", week=1, carries=0, targets=0, attempts=35),
        dict(player_id="b", player_display_name="B", position="QB", team="T", week=2, carries=0, targets=0, attempts=5),
        dict(player_id="s", player_display_name="S", position="QB", team="T", week=2, carries=0, targets=0, attempts=30),
    ])
    sh = model.usage_shares(pw).set_index("player")
    assert sh.loc["S", "attempt_share"] == 1.0 and sh.loc["B", "attempt_share"] == 0.0
