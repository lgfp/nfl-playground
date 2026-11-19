import numpy as np
import pytest

from nflfp import league_odds as lo
from nflfp import season_odds as so

SLOTS = so.lineup_slots(["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF", "BN", "BN"])


# ------------------------------------------------------------------------------------------------------------ lineups
def test_lineup_slots():
    assert SLOTS == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DEF": 1}
    with pytest.raises(ValueError):
        so.lineup_slots(["QB", "DL"])


def test_points_from_stats_ignores_unscored_and_gp():
    assert so.points_from_stats({"rec": 5, "rec_yd": 60, "gp": 1, "rec_tgt": 8}, {"rec": 0.5, "rec_yd": 0.1, "gp": 99}) == pytest.approx(8.5)


def test_best_lineup_fills_flex_from_leftovers():
    players = [("q1", "QB", 20), ("q2", "QB", 18),                       # second QB cannot start (no superflex)
               ("r1", "RB", 15), ("r2", "RB", 14), ("r3", "RB", 13), ("r4", "RB", 12),
               ("w1", "WR", 16), ("w2", "WR", 11), ("w3", "WR", 10),
               ("t1", "TE", 9), ("t2", "TE", 8),
               ("k1", "K", 7), ("d1", "DEF", 6)]
    total, starters = so.best_lineup(players, SLOTS)
    assert set(starters) == {"q1", "r1", "r2", "w1", "w2", "t1", "k1", "d1", "r3", "r4"}   # flex: the best two of what is left (13, 12 beat 10 and 8)
    assert total == pytest.approx(20 + 15 + 14 + 16 + 11 + 9 + 7 + 6 + 13 + 12)


def test_best_lineup_handles_thin_rosters():
    total, starters = so.best_lineup([("q1", "QB", 20), ("r1", "RB", 10)], SLOTS)
    assert total == 30 and set(starters) == {"q1", "r1"}
    assert so.best_lineup([], SLOTS) == (0.0, [])


def test_best_lineup_superflex_prefers_second_qb():
    slots = so.lineup_slots(["QB", "RB", "SUPER_FLEX"])
    total, starters = so.best_lineup([("q1", "QB", 20), ("q2", "QB", 18), ("r1", "RB", 10), ("r2", "RB", 9)], slots)
    assert set(starters) == {"q1", "r1", "q2"} and total == 48


# ------------------------------------------------------------------------------------------------------------ bracket
SLEEPER_4 = [{"m": 1, "r": 1, "l": None, "w": None, "t1": 7, "t2": 8}, {"m": 2, "r": 1, "l": None, "w": None, "t1": 5, "t2": 2},
             {"m": 3, "p": 1, "r": 2, "t1": None, "t2": None, "t2_from": {"w": 2}, "t1_from": {"w": 1}},
             {"m": 4, "p": 3, "r": 2, "t1": None, "t2": None, "t2_from": {"l": 2}, "t1_from": {"l": 1}}]


def test_parse_bracket_maps_rosters_to_seeds():
    br = so.parse_bracket(SLEEPER_4, {7: 1, 5: 2, 2: 3, 8: 4})
    assert [(m["id"], m["a"], m["b"]) for m in br] == [(1, ("seed", 1), ("seed", 4)), (2, ("seed", 2), ("seed", 3)), (3, ("w", 1), ("w", 2)), (4, ("l", 1), ("l", 2))]
    assert [m["place"] for m in br] == [None, None, 1, 3]
    assert br == [dict(m, a=m["a"], b=m["b"]) for m in so.standard_bracket(4)]


def test_round_weeks():
    assert so.round_weeks(2, 15, 0) == {1: [15], 2: [16]}
    assert so.round_weeks(3, 15, 1) == {1: [15], 2: [16], 3: [17, 18]}
    assert so.round_weeks(2, 15, 2) == {1: [15, 16], 2: [17, 18]}


# ------------------------------------------------------------------------------------------------------------ simulation
def season(means, n_playoff=4, bracket=None, rounds=None, wins=None, pf=None, schedule=None, weeks=(11, 12)):
    T = len(means)
    sched = schedule if schedule is not None else {w: [(i, i + 1) for i in range(0, T, 2)] for w in weeks}
    mean = {w: np.array(means, float) for w in list(weeks) + [15, 16]}
    return so.Season(ids=list("ABCDEFGH"[:T]), wins=np.zeros(T) if wins is None else np.array(wins, float), pf=np.zeros(T) if pf is None else np.array(pf, float),
                     schedule=sched, mean=mean, n_playoff=n_playoff, bracket=bracket if bracket is not None else so.standard_bracket(n_playoff),
                     rounds=rounds or {1: [15], 2: [16]})


def test_probabilities_are_consistent():
    s = season([180, 170, 160, 150, 140, 130, 120, 110])
    r = so.simulate(s, sd=25, n=20000, seed=1)
    assert r.p_playoffs.sum() == pytest.approx(4.0)
    assert r.p_champion.sum() == pytest.approx(1.0)
    assert r.p_final.sum() == pytest.approx(2.0)
    assert r.p_third.sum() == pytest.approx(1.0)
    assert np.allclose(r[[f"p_seed{j}" for j in range(1, 5)]].sum(axis=1), r.p_playoffs)
    assert np.allclose(r[[f"p_seed{j}" for j in range(1, 5)]].sum(axis=0)[:], 1.0)
    assert (np.diff(r.p_champion.to_numpy()) <= 0.02).all()       # monotone in strength, up to noise


def test_equal_teams_are_equal():
    r = so.simulate(season([150] * 8), sd=25, n=40000, seed=2)
    assert np.allclose(r.p_playoffs, 0.5, atol=0.02)
    assert np.allclose(r.p_champion, 0.125, atol=0.015)


def test_dominant_team_wins_everything():
    r = so.simulate(season([400, 100, 100, 100]), sd=1.0, n=2000)
    assert r.p_champion[0] == 1.0 and r.p_seed1[0] == 1.0 and r.exp_wins[0] == 2.0


def test_current_record_and_points_break_ties():
    # two teams level on wins with equal strength; the one with more points scored so far takes the better seed
    s = season([150, 150], n_playoff=1, bracket=[], wins=[5, 5], pf=[1000, 900], weeks=(11,))
    s.schedule = {}
    r = so.simulate(s, sd=1.0, n=1000)
    assert r.p_seed1[0] > 0.99 and r.p_playoffs[1] < 0.01


def test_wins_beat_points():
    s = season([150, 150], n_playoff=1, bracket=[], wins=[6, 5], pf=[900, 1500], weeks=(11,))
    s.schedule = {}
    r = so.simulate(s, sd=1.0, n=1000)
    assert r.p_playoffs[0] > 0.99


def test_two_week_championship_sums_the_weeks():
    # B is stronger in week 15, A in week 16 by more: over the two weeks A wins the one-match final
    s = season([150, 150], n_playoff=2, rounds={1: [15, 16]}, weeks=(11,))
    s.schedule = {}
    s.wins, s.pf = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    s.mean[15], s.mean[16] = np.array([100.0, 130.0]), np.array([200.0, 120.0])
    r = so.simulate(s, sd=1.0, n=500)
    assert r.p_champion[0] == 1.0 and r.p_final.tolist() == [1.0, 1.0]


def test_six_team_bracket_gives_byes_to_top_two():
    s = season([300, 250, 150, 140, 130, 120, 100, 90], n_playoff=6, bracket=so.standard_bracket(6), rounds={1: [15], 2: [16], 3: [17]}, weeks=(11, 12, 13))
    s.mean[17] = s.mean[15]
    r = so.simulate(s, sd=1.0, n=500)
    assert r.p_playoffs.sum() == pytest.approx(6.0)
    assert r.p_champion.sum() == pytest.approx(1.0)
    assert r.p_champion[0] == 1.0


# ------------------------------------------------------------------------------------------------------------ feed conversion
SC = {"rec": 0.5, "rec_yd": 0.1, "rec_fd": 0.5, "rush_yd": 0.1, "rush_fd": 0.5, "fgm_50_59": 4.0, "fgmiss_50_59": -0.5, "def_pr_yd": 0.1, "sack": 1.0}


def test_feed_first_downs_are_replaced_by_rates():
    wr = {"rec": 5.0, "rec_yd": 60.0, "rec_fd": 6.0, "adp_dd_ppr": 50.0}            # the feed says rec_yd / 10 first downs: more than catches
    assert lo.points(wr, SC, "WR") == pytest.approx(0.5 * 5 + 0.1 * 60 + 0.5 * lo.FD_PER_REC["WR"] * 5)


def test_feed_bye_week_has_no_projection():
    assert lo.points({"adp_dd_ppr": 120.0, "pos_adp_dd_ppr": 30.0}, SC, "RB") is None


def test_feed_defense_return_yards_are_dropped_and_kicker_fifty_plus_is_aliased():
    assert lo.points({"sack": 3.0, "def_pr_yd": 170.0}, SC, "DEF") == pytest.approx(3.0)
    assert lo.points({"fgm_50p": 0.5, "fgmiss_50p": 0.2}, SC, "K") == pytest.approx(0.5 * 4 - 0.2 * 0.5)
