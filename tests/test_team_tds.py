import numpy as np
import pandas as pd

from nflfp import team_tds as M3
from nflfp import tds as T


def synth(seed=0, seasons=(2025, 2026), weeks=12, teams=10, b_imp=0.08):
    """Yards, implied totals and TDs for several seasons; rush/pass TDs are Poisson with a planted effect of the implied total."""
    rng = np.random.default_rng(seed)
    ts = [f"T{i}" for i in range(teams)]
    rows = []
    for season in seasons:
        for w in range(1, weeks + 1):
            perm = rng.permutation(teams)
            for a, b in zip(perm[::2], perm[1::2]):
                total = float(rng.integers(38, 56))
                spread = float(rng.integers(-7, 8))
                for o, d, fav in ((ts[a], ts[b], spread), (ts[b], ts[a], -spread)):
                    imp = total / 2 + fav / 2
                    rows.append(dict(season=season, week=w, off_team=o, def_team=d, implied=imp, rush_exp=110 + rng.normal(0, 15), pass_exp=230 + rng.normal(0, 30),
                                     rush_td=rng.poisson(np.exp(0.0 + b_imp * (imp - 22))), pass_td=rng.poisson(1.5 * np.exp(b_imp * (imp - 22)))))
    t = pd.DataFrame(rows)
    return t[M3.KEYS + ["rush_td", "pass_td"]], t[M3.KEYS + ["rush_exp", "pass_exp"]], t[M3.KEYS + ["implied"]]


def test_implied_total_from_schedule():
    sched = pd.DataFrame(dict(season=[2026], week=[9], home_team=["A"], away_team=["B"], spread_line=[3.0], total_line=[44.0]))
    d = M3.implied_totals(sched).set_index("off_team")
    assert d.loc["A", "implied"] == 44 / 2 + 3 / 2 and d.loc["B", "implied"] == 44 / 2 - 3 / 2


def test_projections_are_positive_and_track_the_planted_implied_total_effect():
    tds, yards, imp = synth()
    out = M3.walk_forward(tds, yards, imp, [2026])
    assert len(out) == len(imp[imp.season == 2026])
    assert (out[["rush_td_exp", "pass_td_exp"]] > 0).all().all()
    assert np.allclose(out.total_td_exp, out.rush_td_exp + out.pass_td_exp)
    late = out[out.week >= 4]
    assert np.corrcoef(late.implied, late.rush_td_exp)[0, 1] > 0.9 and np.corrcoef(late.implied, late.pass_td_exp)[0, 1] > 0.9
    assert late.pass_td_exp.mean() > late.rush_td_exp.mean()


def test_no_look_ahead_and_future_weeks_are_projected():
    tds, yards, imp = synth()
    full = M3.walk_forward(tds, yards, imp, [2026])
    changed = tds.copy()
    changed.loc[(changed.season == 2026) & (changed.week >= 7), ["rush_td", "pass_td"]] = 9
    alt = M3.walk_forward(changed, yards, imp, [2026])
    k = M3.KEYS
    a, b = full[full.week <= 7].sort_values(k), alt[alt.week <= 7].sort_values(k)       # week 7's own TDs may differ, its forecast may not
    np.testing.assert_allclose(a.rush_td_exp, b.rush_td_exp)
    np.testing.assert_allclose(a.pass_td_exp, b.pass_td_exp)
    played = tds[~((tds.season == 2026) & (tds.week >= 10))]                              # weeks 10+ not played yet
    wk = M3.project_week(played, yards, imp, 2026, 10)
    assert len(wk) == 10 and wk.rush_td_exp.notna().all() and wk.rush_td.isna().all()


def test_few_games_fall_back_to_the_league_average():
    tds, yards, imp = synth(seasons=(2026,), weeks=3, teams=6)
    out = M3.walk_forward(tds, yards, imp, [2026])
    wk3 = out[out.week == 3]
    past = tds[tds.week < 3]
    np.testing.assert_allclose(wk3.rush_td_exp, past.rush_td.mean())
    np.testing.assert_allclose(wk3.pass_td_exp, past.pass_td.mean())


def test_poisson_fit_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20000, 2))
    y = rng.poisson(np.exp(0.3 + 0.5 * X[:, 0] - 0.2 * X[:, 1])).astype(float)
    b = T.poisson_fit(X, y, ridge=1e-6)
    np.testing.assert_allclose(b, [0.3, 0.5, -0.2], atol=0.03)
    assert np.allclose(T.poisson_predict(b, X[:3]), np.exp(b[0] + X[:3] @ b[1:]))
