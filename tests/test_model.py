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
    assert adj.adjust(99, team, 1.0, 1.0) == (1.0, 1.0) and (99, team) in adj.missing  # no line: unadjusted, recorded


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


def test_ranking_metrics_perfect_reversed_and_noise():
    from nflfp import backtest
    rng = np.random.default_rng(3)
    bt = pd.DataFrame({"week": np.repeat(np.arange(1, 9), 20), "act_rush": rng.normal(size=160)})
    bt["act_rec"] = bt.act_rush
    bt["mod_rush"], bt["own_rush"] = bt.act_rush * 5 + 1, -bt.act_rush            # same order / reversed
    bt["mod_rec"], bt["own_rec"] = bt.act_rush, rng.normal(size=160)               # same order / pure noise
    r = backtest.ranking(bt, n_boot=200)
    assert r.loc[("rush", "model"), "pairwise"] == 1.0 and r.loc[("rush", "model"), "rank_corr"] == pytest.approx(1.0)
    assert r.loc[("rush", "team avg"), "pairwise"] == 0.0 and r.loc[("rush", "team avg"), "rank_corr"] == pytest.approx(-1.0)
    assert 0.4 < r.loc[("rec", "team avg"), "pairwise"] < 0.6                      # noise ~ coin flip
    assert r.loc[("rec", "team avg"), "pair_lo"] < 0.5 < r.loc[("rec", "team avg"), "pair_hi"]
    assert r.loc[("rush", "model"), "top_half"] == 1.0


def _wind_fixture(effect=-2.0):
    """Lines fixture + a wind column, with a planted wind effect on receiving points."""
    linemod, tg, r, lines = _lines_fixture()
    rng = np.random.default_rng(5)
    lines = lines.assign(wind5=rng.uniform(0, 3, len(lines)))
    d = tg.merge(lines, left_on=["week", "off_team"], right_on=["week", "team"])
    return linemod, tg, r, lines, d, effect


def test_wind_adjustment_recovers_planted_receiving_effect_and_not_rushing():
    linemod, tg, r, lines, d, effect = _wind_fixture()
    s = model.Scoring()
    base = linemod.fit_line_adjustment(d[tg.columns], r, s, lines, ("fav",), wind=True)
    planted = d[tg.columns].copy()
    planted["rec_yds"] = d["rec_yds"] + (effect / s.yard) * d["wind5"]     # effect points per 5 mph
    adj = linemod.fit_line_adjustment(planted, r, s, lines, ("fav",), wind=True)
    i = adj.use["rec"].index("wind") + 1
    assert adj.coef["rec"][i] - base.coef["rec"][i] == pytest.approx(effect, abs=1e-9)
    assert "wind" not in adj.use["rush"]                                    # wind is for passing only


def test_wind_unknown_is_neutral_and_indoor_wind_is_zero():
    linemod, tg, r, lines, d, _ = _wind_fixture()
    s = model.Scoring()
    adj = linemod.fit_line_adjustment(d[tg.columns], r, s, lines, ("fav", "total"), wind=True)
    wk, team = int(lines.week.iloc[0]), lines.team.iloc[0]
    with_wind = adj.adjust(wk, team, 10.0, 40.0)
    unknown = linemod.LineAdjustment(**{**adj.__dict__, "lines": adj.lines.assign(wind5=np.nan), "missing": set(), "missing_wind": set()})
    neutral = unknown.adjust(wk, team, 10.0, 40.0)
    avg_wind = linemod.LineAdjustment(**{**adj.__dict__, "lines": adj.lines.assign(wind5=adj.wind_mean), "missing": set(), "missing_wind": set()})
    assert neutral == pytest.approx(avg_wind.adjust(wk, team, 10.0, 40.0))  # unknown wind == average wind
    assert (wk, team) in unknown.missing_wind and with_wind[1] != neutral[1]
    sched = pd.DataFrame([dict(game_id="g1", week=1, home_team="H", away_team="A", spread_line=1.0, total_line=44.0, roof="dome"),
                          dict(game_id="g2", week=2, home_team="H", away_team="A", spread_line=1.0, total_line=44.0, roof=np.nan),
                          dict(game_id="g3", week=3, home_team="X", away_team="Y", spread_line=1.0, total_line=44.0, roof="outdoors")])
    wx = pd.DataFrame({"wind_mph": [20.0, 20.0, 20.0]}, index=pd.Index(["g1", "g2", "g3"], name="game_id"))
    t = linemod.team_lines(sched, wx)
    assert (t[t.week == 1].wind5 == 0).all()       # dome: no wind
    assert (t[t.week == 2].wind5 == 0).all()       # unknown roof -> the home team's usual roof (dome)
    assert (t[t.week == 3].wind5 == 4.0).all()     # outdoors: 20 mph = 4 units


def _usage_fixture():
    rows = []
    for w in (1, 2, 3):
        rows += [dict(player_id="rb1", player_display_name="RB One", position="RB", team="T", week=w, carries=15, targets=3, attempts=0),
                 dict(player_id="rb2", player_display_name="RB Two", position="RB", team="T", week=w, carries=5, targets=1, attempts=0),
                 dict(player_id="rb3", player_display_name="RB Three", position="RB", team="T", week=w, carries=10, targets=1, attempts=0),
                 dict(player_id="wr1", player_display_name="WR One", position="WR", team="T", week=w, carries=0, targets=6, attempts=0),
                 dict(player_id="qb1", player_display_name="QB One", position="QB", team="T", week=w, carries=0, targets=0, attempts=30)]
    rows.append(dict(player_id="qb2", player_display_name="QB Two", position="QB", team="T", week=1, carries=0, targets=0, attempts=8))
    return pd.DataFrame(rows)


def test_excluded_players_are_dropped_and_shares_renormalise():
    pw = _usage_fixture()
    base = model.usage_shares(pw).set_index("player_id")
    out = model.usage_shares(pw, exclude={"rb1"}).set_index("player_id")
    assert "rb1" not in out.index
    assert out.carry_share.sum() == pytest.approx(1) and out.target_share.sum() == pytest.approx(1)
    # remaining RBs keep their relative usage: RB3 had 2x RB2's carries and still does
    assert out.loc["rb3", "carry_share"] == pytest.approx(2 * out.loc["rb2", "carry_share"])
    assert out.loc["rb2", "carry_share"] > base.loc["rb2", "carry_share"]
    assert base.loc["rb1", "carry_share"] == pytest.approx(0.5)


def test_passing_points_move_to_the_backup_when_the_starter_is_excluded():
    pw = _usage_fixture()
    assert model.usage_shares(pw).set_index("player_id").loc["qb1", "attempt_share"] == 1.0
    out = model.usage_shares(pw, exclude={"qb1"}).set_index("player_id")
    assert out.loc["qb2", "attempt_share"] == 1.0   # QB Two did not play in the latest week: fallback to most attempts


def test_sleeper_mapping_by_id_name_and_team_code_and_status_filter():
    from nflfp import sleeper
    known = pd.DataFrame([dict(player_id="00-1", player_display_name="James Cook", team="BUF", position="RB"),
                          dict(player_id="00-2", player_display_name="Puka Nacua", team="LA", position="WR"),
                          dict(player_id="00-3", player_display_name="Justin Jefferson", team="MIN", position="WR"),
                          dict(player_id="00-4", player_display_name="Justin Jefferson", team="CLE", position="LB")])
    players = {"a": dict(first_name="James", last_name="Cook III", team="BUF", position="RB", gsis_id=None, injury_status="Out"),        # suffix
               "b": dict(first_name="Puka", last_name="Nacua", team="LAR", position="WR", gsis_id=None, injury_status="IR"),             # LAR -> LA
               "c": dict(first_name="Justin", last_name="Jefferson", team="MIN", position="WR", gsis_id="00-3", injury_status="Out"),    # by id
               "d": dict(first_name="Other", last_name="Guy", team="MIN", position="WR", gsis_id="00-3", injury_status="Questionable"),  # not out
               "e": dict(first_name="Nobody", last_name="Known", team="MIN", position="WR", gsis_id=None, injury_status="Out")}          # unmatched
    out = sleeper.unavailable(players, known)
    assert set(out.player_id) == {"00-1", "00-2", "00-3"}


def test_league_scoring_translation_and_lineup_and_upgrades():
    from nflfp import league
    sc, notes = league.scoring_from_league({"rec": 0.5, "rec_yd": 0.1, "rush_yd": 0.1, "rec_td": 6.0, "pass_yd": 0.04, "pass_td": 4.0})
    assert (sc.ppr, sc.yard, sc.td, sc.pass_yard, sc.pass_td) == (0.5, 0.1, 6.0, 0.04, 4.0) and notes == []
    assert league.scoring_from_league({"rush_yd": 0.2, "rec_yd": 0.1})[1]          # differing yard rates are flagged

    roster = pd.DataFrame([("Q1", "QB", 20.0), ("Q2", "QB", 18.0), ("R1", "RB", 15.0), ("R2", "RB", 12.0), ("R3", "RB", 11.0),
                           ("W1", "WR", 14.0), ("W2", "WR", 13.0), ("W3", "WR", 9.0), ("T1", "TE", 8.0), ("X", "RB", np.nan)],
                          columns=["player", "position", "pts"])
    lu = league.best_lineup(roster, ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF", "BN"])
    assert list(lu.player) == ["Q1", "R1", "R2", "W1", "W2", "T1", "R3", "W3"]       # fixed slots first, flex from the rest
    sf = league.best_lineup(roster, ["QB", "SUPER_FLEX"])
    assert list(sf.player) == ["Q1", "Q2"]                                           # a QB fits the super flex

    avail = pd.DataFrame([("A", "RB", 14.5), ("B", "RB", 13.1), ("C", "WR", 11.0), ("D", "TE", 9.0)], columns=["player", "position", "pts"])
    up = league.find_upgrades(avail, roster, 0.20)
    # RB A (14.5) is >=20% above R3 (11.0 x 1.2 = 13.2) but not above R2 (12.0 x 1.2 = 14.4 -> yes, 14.5 clears it too)
    assert list(up.player) == ["A", "C"]
    a = up.set_index("player")
    assert a.loc["A", "replaces"] == "R3" and a.loc["A", "beats"] == 2 and a.loc["A", "gain"] == pytest.approx(3.5)
    assert a.loc["C", "replaces"] == "W3" and a.loc["C", "gain_pct"] == pytest.approx(11 / 9 - 1)
    # TE D: 9.0 < 1.2 x 8.0 = 9.6 -> not an upgrade;  RB B: 13.1 < 1.2 x 11.0 = 13.2 -> not an upgrade
    assert "D" not in up.player.values and "B" not in up.player.values


def test_decay_weights_recent_games_more_and_default_is_unchanged():
    pw = pd.DataFrame([dict(player_id=pid, player_display_name=pid, position="RB", team="T", week=w, carries=c, targets=0, attempts=0)
                       for w, (a, b) in enumerate([(20, 5), (20, 5), (5, 20), (5, 20)], start=1) for pid, c in (("A", a), ("B", b))])
    eq = model.usage_shares(pw).set_index("player_id")
    assert eq.loc["A", "carry_share"] == pytest.approx(0.5)                              # equal weights: 50 / 100
    pd.testing.assert_frame_equal(model.usage_shares(pw, decay=1.0), model.usage_shares(pw))
    rec = model.usage_shares(pw, decay=0.5).set_index("player_id")
    assert rec.loc["B", "carry_share"] > 0.5 > rec.loc["A", "carry_share"]               # B carried more recently
    assert rec.carry_share.sum() == pytest.approx(1)
    last = model.usage_shares(pw, decay=0.0).set_index("player_id")                      # latest game only: 5 vs 20
    assert last.loc["A", "carry_share"] == pytest.approx(0.2) and last.loc["B", "carry_share"] == pytest.approx(0.8)
    assert eq.loc["A", "carries"] == 50                                                  # plain totals are unchanged


def test_decay_default_is_wired_through_pipeline_cli_and_league():
    import inspect
    import re
    from nflfp import league, pipeline
    assert model.DEFAULT_DECAY == 0.5
    assert inspect.signature(pipeline.project).parameters["decay"].default == model.DEFAULT_DECAY
    for mod in ("__main__", "league"):
        src = inspect.getsource(__import__(f"nflfp.{mod}", fromlist=["x"]))
        assert re.search(r'"--decay".*\n?.*default=model\.DEFAULT_DECAY', src), mod


def test_cache_expiry_refresh_and_failure_handling(tmp_path, monkeypatch, capsys):
    import os
    import time
    from nflfp import data
    monkeypatch.setattr(data, "CACHE", tmp_path)
    calls = []

    def fake_retrieve(url, dest):
        calls.append(str(dest))
        if getattr(fake_retrieve, "fail", False):
            raise OSError("network down")
        open(dest, "w").write("fresh")

    monkeypatch.setattr(data.urllib.request, "urlretrieve", fake_retrieve)
    f = tmp_path / "x.csv"

    def age(hours):
        t = time.time() - hours * 3600
        os.utime(f, (t, t))

    assert data._fetch("u", "x.csv", False, 24) == f and f.read_text() == "fresh" and len(calls) == 1   # no file: download
    data._fetch("u", "x.csv", False, 24); assert len(calls) == 1                                         # fresh: reuse
    age(30); f.write_text("old"); age(30); data._fetch("u", "x.csv", False, 24)
    assert len(calls) == 2 and f.read_text() == "fresh"                                                  # older than ttl: refetch
    age(500); data._fetch("u", "x.csv", False, None); assert len(calls) == 2                             # ttl None: never expires
    data._fetch("u", "x.csv", True, None); assert len(calls) == 3                                        # --refresh always refetches
    fake_retrieve.fail = True
    f.write_text("stale"); age(30)
    assert data._fetch("u", "x.csv", False, 24) == f and f.read_text() == "stale"                        # failed refresh: keep the old copy
    assert "could not refresh" in capsys.readouterr().err and not (tmp_path / "x.csv.part").exists()
    with pytest.raises(OSError):
        data._fetch("u", "nothing.csv", False, 24)                                                       # nothing to fall back on
    assert not (tmp_path / "nothing.csv.part").exists()


def test_current_season_rolls_over_in_march():
    from nflfp import data
    ts = pd.Timestamp
    assert data.current_season(ts("2026-11-03")) == 2026 and data.current_season(ts("2027-01-15")) == 2026
    assert data.current_season(ts("2027-02-28")) == 2026 and data.current_season(ts("2027-03-01")) == 2027
