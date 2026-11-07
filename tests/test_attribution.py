import numpy as np
import pandas as pd

from nflfp import attribution as A
from nflfp.model import Scoring


def vol():
    rows = [("a", "RB1", "RB", "X", None, 12, 2, 0), ("b", "RB2", "RB", "X", None, 8, 1, 0), ("c", "WR1", "WR", "X", None, 0, 7, 0),
            ("d", "QB", "QB", "X", None, 3, 0, 30), ("e", "RBout", "RB", "X", "Out", 10, 3, 0), ("f", "RBy", "RB", "Y", None, 10, 4, 0),
            ("g", "QBy", "QB", "Y", None, 1, 0, 28)]
    return pd.DataFrame(rows, columns=["sleeper_id", "name", "position", "team", "injury_status", "carries", "targets", "attempts"])


def test_shares_sum_to_one_per_team_and_out_players_are_dropped():
    sh = A.shares(vol())
    assert "e" not in set(sh.sleeper_id)
    for col in ("carries_share", "targets_share", "attempts_share"):
        assert np.allclose(sh.groupby("team")[col].sum(), 1.0)
    assert np.isclose(sh.set_index("sleeper_id").loc["a", "carries_share"], 12 / 23)       # 12 of 12 + 8 + 3, Out RB's 10 removed
    assert "c" not in set(A.shares(vol(), exclude={"c"}).sleeper_id)


def test_attribution_conserves_team_yards_and_scores_them():
    ty = pd.DataFrame(dict(off_team=["X", "Y"], def_team=["Y", "X"], rush_exp=[100.0, 120.0], pass_exp=[250.0, 200.0]))
    rates = A.Rates(rush_td_per_yd=0.01, pass_td_per_yd=0.008, rec_per_pass_yd=0.1)
    out = A.attribute(ty, A.shares(vol()), rates)
    g = out.groupby("team")
    assert np.allclose(g.rush_yds.sum(), [100, 120]) and np.allclose(g.rec_yds.sum(), [250, 200]) and np.allclose(g.pass_yds.sum(), [250, 200])
    sc = Scoring()
    x = out.set_index("sleeper_id").loc["c"]                                              # the only WR: all of X's targets
    assert np.isclose(x.targets_share, 7 / 10) and np.isclose(x.rec_yds, 250 * 0.7)
    qb = out.set_index("sleeper_id").loc["d"]                                             # all of X's attempts
    assert np.isclose(qb.pass_pts, sc.pass_yard * 250 + sc.pass_td * 250 * 0.008)
    assert np.isclose(out.exp_pts.sum(), out.rush_pts.sum() + out.rec_pts.sum() + out.pass_pts.sum())


def test_league_rates():
    pw = pd.DataFrame(dict(rushing_yards=[100, 100], rushing_tds=[1, 0], passing_yards=[250, 250], passing_tds=[2, 2], receptions=[20, 30]))
    r = A.league_rates(pw)
    assert np.isclose(r.rush_td_per_yd, 0.005) and np.isclose(r.pass_td_per_yd, 0.008) and np.isclose(r.rec_per_pass_yd, 0.1)
