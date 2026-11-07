import numpy as np
import pandas as pd

from nflfp import td_attribution as TA
from nflfp.model import Scoring


def plays():
    """Team X's games: weeks 1-3 of 2026 (and week 1 of 2025). Runners R1 (heavy, from the 5) and R2; receivers W1 / W2; the QB passes."""
    rows = []
    def add(season, week, kind, pid, n, xtd):
        for _ in range(n):
            rows.append(dict(season=season, week=week, posteam="X", play_type=kind, xtd=xtd,
                             rusher_player_id=pid if kind == "run" else None, rusher_player_name=pid if kind == "run" else None,
                             receiver_player_id=pid if kind == "pass" else None, receiver_player_name=pid if kind == "pass" else None))
    for season, week in ((2025, 1), (2026, 1), (2026, 2), (2026, 3)):
        add(season, week, "run", "R1", 10, 0.10)      # R1: 1.0 xTD a game
        add(season, week, "run", "R2", 10, 0.05)      # R2: 0.5
        add(season, week, "pass", "W1", 6, 0.1)
        add(season, week, "pass", "W2", 3, 0.1)
    return pd.DataFrame(rows)


def test_shares_are_xtd_for_rushing_and_targets_for_receiving_and_sum_to_one():
    sh = TA.shares(plays(), 2026, 4).set_index("pid")
    assert np.isclose(sh.rush_share.sum(), 1) and np.isclose(sh.rec_share.sum(), 1)
    assert np.isclose(sh.loc["R1", "rush_share"], 2 / 3) and np.isclose(sh.loc["W1", "rec_share"], 2 / 3)       # xTD 1.0 vs 0.5; 6 vs 3 targets


def test_excluding_a_player_hands_his_share_to_teammates_pro_rata():
    sh = TA.shares(plays(), 2026, 4, exclude={"R1"}).set_index("pid")
    assert "R1" not in sh.index and np.isclose(sh.loc["R2", "rush_share"], 1.0)


def test_only_earlier_games_count_and_recent_games_weigh_more():
    p = plays()
    early = TA.shares(p, 2026, 2).set_index("pid")                       # only 2025 w1 and 2026 w1 are before week 2
    assert np.isclose(early.rush_share.sum(), 1)
    p2 = p.copy()
    p2.loc[(p2.season == 2026) & (p2.week == 3) & (p2.rusher_player_id == "R2"), "xtd"] = 0.5     # R2 has a big latest game
    a = TA.shares(p2, 2026, 4, cfg=TA.ShareConfig(decay=1.0)).set_index("pid").loc["R2", "rush_share"]
    b = TA.shares(p2, 2026, 4, cfg=TA.ShareConfig(decay=0.5)).set_index("pid").loc["R2", "rush_share"]
    assert b > a
    assert TA.shares(p2, 2026, 3).set_index("pid").loc["R2", "rush_share"] < a      # week 3's game is not visible to week 3


def test_attribution_and_points():
    sh = TA.shares(plays(), 2026, 4)
    team = pd.DataFrame(dict(off_team=["X"], def_team=["Y"], rush_td_exp=[1.2], pass_td_exp=[1.5]))
    td = TA.attribute(team, sh)
    assert np.isclose(td.rush_tds.sum(), 1.2) and np.isclose(td.rec_tds.sum(), 1.5)
    m2 = pd.DataFrame(dict(sleeper_id=["a", "b", "c", "q"], team=["X"] * 4, exp_pts=[10.0, 5.0, 4.0, 15.0], attempts_share=[0, 0, 0, 1.0]))
    out = TA.add_touchdowns(m2, td, {"a": "R1", "b": "W1", "c": "nobody", "q": "QB"}, Scoring())
    r1, w1, c, q = (out.set_index("sleeper_id").loc[k] for k in ("a", "b", "c", "q"))
    assert np.isclose(r1.rush_tds, 1.2 * 2 / 3) and np.isclose(w1.rec_tds, 1.5 * 2 / 3) and c.td_pts == 0
    assert np.isclose(q.pass_tds, 1.5) and np.isclose(q.td_pts, 4.0 * 1.5)
    assert np.isclose(r1.total_pts, 10.0 + 6.0 * r1.rush_tds)
