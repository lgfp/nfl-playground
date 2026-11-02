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


def test_defense_ranking_worst_first_and_separate_for_run_and_pass():
    tg, *_ = synthetic_games()
    for c in ("rec_yds", "rec_tds", "receptions"):  # decouple pass from run
        tg[c] = tg["rush_yds"][::-1].to_numpy()
    r = model.fit_ratings(tg)
    s = model.Scoring()
    run, pas = (model.defense_ranking(r, s, k) for k in ("run", "pass"))
    for t in (run, pas):
        assert list(t["rank"]) == list(range(1, len(t) + 1))
        assert t["adj_pts"].is_monotonic_decreasing
    assert list(run.index) != list(pas.index)


def test_team_projection_matches_components():
    tg, *_ = synthetic_games()
    r = model.fit_ratings(tg)
    s = model.Scoring()
    t = model.project_teams(r, [("T0", "T1")], s).iloc[0]
    rush, rec = model.points(model.team_expectation(r, "T0", "T1"), s)
    assert t.total_pts == pytest.approx(rush + rec)


def test_team_projection_sort_order():
    tg, *_ = synthetic_games()
    r = model.fit_ratings(tg)
    m = [("T0", "T1"), ("T2", "T3"), ("T4", "T5")]
    assert model.project_teams(r, m, model.Scoring()).rush_pts.is_monotonic_decreasing
    assert model.project_teams(r, m, model.Scoring(), sort_by="total_pts").total_pts.is_monotonic_decreasing


def test_backtest_has_no_lookahead():
    from nflfp import backtest
    rows = []
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(8)]
    for w in range(1, 9):
        perm = rng.permutation(8)
        for a, b in zip(perm[::2], perm[1::2]):
            for o, d in ((teams[a], teams[b]), (teams[b], teams[a])):
                rows.append(dict(team=o, opponent_team=d, week=w, rushing_yards=rng.uniform(50, 150), rushing_tds=1,
                                 receiving_yards=rng.uniform(150, 300), receiving_tds=2, receptions=20))
    pw = pd.DataFrame(rows)
    base = backtest.walk_forward(pw, model.Scoring(), min_week=5)
    corrupted = pw.copy()
    corrupted.loc[corrupted.week >= 7, ["rushing_yards", "receiving_yards"]] *= 1000  # change the future
    after = backtest.walk_forward(corrupted, model.Scoring(), min_week=5)
    pd.testing.assert_frame_equal(
        base[base.week < 7][["week", "team", "mod_rush", "mod_rec", "lg_rush", "own_rush"]].reset_index(drop=True),
        after[after.week < 7][["week", "team", "mod_rush", "mod_rec", "lg_rush", "own_rush"]].reset_index(drop=True))


def test_leave_one_out_holds_out_only_the_tested_week():
    from nflfp import backtest
    rng = np.random.default_rng(2)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for w in range(1, 7):
        perm = rng.permutation(8)
        for a, b in zip(perm[::2], perm[1::2]):
            for o, d in ((teams[a], teams[b]), (teams[b], teams[a])):
                rows.append(dict(team=o, opponent_team=d, week=w, rushing_yards=rng.uniform(50, 150), rushing_tds=1,
                                 receiving_yards=rng.uniform(150, 300), receiving_tds=2, receptions=20))
    pw = pd.DataFrame(rows)
    s = model.Scoring()
    base = backtest.walk_forward(pw, s, min_week=1, leave_one_out=True)
    cols = ["week", "team", "mod_rush", "mod_rec"]

    same_week = pw.copy()  # corrupting the tested week itself must not change that week's prediction
    same_week.loc[same_week.week == 3, ["rushing_yards", "receiving_yards"]] *= 1000
    after = backtest.walk_forward(same_week, s, min_week=1, leave_one_out=True)
    pd.testing.assert_frame_equal(base[base.week == 3][cols].reset_index(drop=True),
                                  after[after.week == 3][cols].reset_index(drop=True))

    future = pw.copy()  # but a *future* week must matter (that is the point of this mode)
    future.loc[future.week == 6, ["rushing_yards", "receiving_yards"]] *= 1000
    after = backtest.walk_forward(future, s, min_week=1, leave_one_out=True)
    assert not np.allclose(base[base.week == 3].mod_rush.to_numpy(), after[after.week == 3].mod_rush.to_numpy())


def test_neutral_component_is_league_average_and_others_unchanged():
    tg, *_ = synthetic_games()
    base = model.fit_ratings(tg)
    neu = model.fit_ratings(tg, neutral=("rush_tds",))
    assert (neu.off["rush_tds"] == 1).all() and (neu.dfn["rush_tds"] == 1).all()
    pd.testing.assert_series_equal(neu.dfn["rush_yds"], base.dfn["rush_yds"])
    exp = model.team_expectation(neu, "T0", "T1")
    assert exp["rush_tds"] == pytest.approx(neu.mu["rush_tds"])


def test_weights_act_like_repeated_games_and_default_is_unweighted():
    tg, *_ = synthetic_games()
    noshr = {c: 0.0 for c in model.COMPONENTS}
    pd.testing.assert_frame_equal(model.fit_ratings(tg, noshr).dfn, model.fit_ratings(tg.assign(weight=1.0), noshr).dfn)
    # weight 2 on a game == listing it twice
    doubled = pd.concat([tg, tg.iloc[:5]], ignore_index=True)
    weighted = tg.assign(weight=1.0)
    weighted.loc[weighted.index[:5], "weight"] = 2.0
    pd.testing.assert_frame_equal(model.fit_ratings(doubled, noshr).dfn, model.fit_ratings(weighted, noshr).dfn)
    pd.testing.assert_frame_equal(model.fit_ratings(doubled, noshr).raw_allowed, model.fit_ratings(weighted, noshr).raw_allowed)


def test_def_weight_zero_excludes_games_from_defense_factors_only():
    tg, *_ = synthetic_games()
    noshr = {c: 0.0 for c in model.COMPONENTS}
    extra = tg.copy(); extra["rush_yds"] *= 3  # a distorted "other season"
    extra["weight"], extra["def_weight"] = 1.0, 0.0
    both = pd.concat([tg.assign(weight=1.0, def_weight=1.0), extra], ignore_index=True)
    r_both, r_base = model.fit_ratings(both, noshr), model.fit_ratings(tg, noshr)
    # defense factors only see the current games, so the distortion (a uniform x3) cannot move them
    pd.testing.assert_series_equal(r_both.dfn["rush_yds"], r_base.dfn["rush_yds"], atol=1e-6, rtol=1e-6)
    assert r_both.mu["rush_yds"] > r_base.mu["rush_yds"]


def _lines_fixture():
    from nflfp import lines as linemod
    tg, *_ = synthetic_games()
    r = model.fit_ratings(tg, shrink={c: 0.0 for c in model.COMPONENTS}, neutral=())
    teams = sorted(set(tg.off_team))
    sched = pd.DataFrame([dict(week=w, home_team=a, away_team=b, spread_line=(i % 7) - 3.0, total_line=40.0 + (i % 5))
                          for i, (w, a, b) in enumerate((w, teams[(w + k) % 12], teams[(w + k + 5) % 12]) for w in range(1, 15) for k in range(3))])
    return linemod, tg, r, linemod.team_lines(sched)


def test_team_lines_sign_convention():
    from nflfp import lines as linemod
    sched = pd.DataFrame([dict(week=1, home_team="H", away_team="A", spread_line=3.5, total_line=44.0)])
    t = linemod.team_lines(sched).set_index("team")
    assert t.loc["H", "fav"] == 3.5 and t.loc["A", "fav"] == -3.5 and (t.total_line == 44.0).all()


def test_line_adjustment_recovers_a_planted_effect_and_floors_at_zero():
    linemod, tg, r, lines = _lines_fixture()
    s = model.Scoring()
    d = tg.merge(lines, left_on=["week", "off_team"], right_on=["week", "team"])
    planted = d[tg.columns].copy()
    planted["rush_yds"] = d["rush_yds"] + 2.0 * d["fav"]  # +2 yards per point favoured = +0.2 rush points
    base = linemod.fit_line_adjustment(d[tg.columns], r, s, lines, ("fav",))
    adj = linemod.fit_line_adjustment(planted, r, s, lines, ("fav",))
    # least squares is linear in the target, so the planted effect shifts the rush slope by exactly +0.2
    assert adj.coef["rush"][1] - base.coef["rush"][1] == pytest.approx(0.2, abs=1e-9)
    assert adj.coef["rec"][1] == pytest.approx(base.coef["rec"][1], abs=1e-9)
    team = lines.team.iloc[0]
    assert adj.adjust(int(lines.week.iloc[0]), team, 0.0, 0.0)[0] >= 0.0
    with pytest.raises(SystemExit):
        adj.adjust(99, team, 1.0, 1.0)


def test_line_adjust_hook_changes_team_projection_only_when_given():
    linemod, tg, r, lines = _lines_fixture()
    s = model.Scoring()
    adj = linemod.fit_line_adjustment(tg, r, s, lines)
    wk = int(lines.week.iloc[0]); team = lines[lines.week == wk].team.iloc[0]
    base = model.project_teams(r, [(team, "T1")], s)
    hooked = model.project_teams(r, [(team, "T1")], s, adjust=lambda o, ru, re: adj.adjust(wk, o, ru, re))
    plain = model.project_teams(r, [(team, "T1")], s, adjust=None)
    pd.testing.assert_frame_equal(base, plain)
    assert hooked.rush_pts.iloc[0] != base.rush_pts.iloc[0]


def test_kickoff_is_converted_from_eastern_to_utc():
    from nflfp import weather
    assert weather.kickoff_utc("2025-09-07", "13:00") == pd.Timestamp("2025-09-07 17:00", tz="UTC")  # EDT
    assert weather.kickoff_utc("2025-12-07", "13:00") == pd.Timestamp("2025-12-07 18:00", tz="UTC")  # EST
    assert set(weather.STADIUMS) >= {"LA", "LAC", "LV", "NYG", "NYJ"}
