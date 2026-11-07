"""Two proposed fixes to the TD modules, alone and together (walk-forward, 2025-26 weeks 4+):

  POS  receiving TDs weighted by the position's TD-per-target rate (RB < WR < TE), team total preserved (module 4)
  TEAM team rushing TDs scaled by the team's own rushing-TD record (ridge toward the league, K pseudo-games, 30% last season) (module 3)

    uv run python -m analysis.td_fixes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import data, td_attribution as TA, team_tds as M3, team_yards as TY, tds as T

K_TEAM = 16.0            # fixed before looking at results
POS = ("RB", "WR", "TE")


def team_factor(tg: pd.DataFrame, s: int, w: int, k: float, side: str = "rush") -> pd.Series:
    """Per team: ridge estimate of its {side} TDs per game divided by the league's (1.0 = league average). Earlier games only."""
    col = f"{side}_td"
    prev, cur = tg[tg.season == s - 1], tg[(tg.season == s) & (tg.week < w)]
    base = cur if len(cur) else prev
    lg = float(base[col].mean())
    pm = prev.groupby("off_team")[col].mean() if len(prev) else pd.Series(dtype=float)
    teams = tg.off_team.unique()
    prior = pd.Series({t: 0.3 * pm.get(t, lg) + 0.7 * lg for t in teams})
    ssum, n = cur.groupby("off_team")[col].sum().reindex(teams).fillna(0.0), cur.groupby("off_team").size().reindex(teams).fillna(0.0)
    return ((ssum + k * prior) / (n + k)) / lg


def pos_multiplier(stats: dict, s: int, w: int) -> dict:
    """Position -> receiving TDs per target relative to the all-RB/WR/TE rate, from season s-1 and season s before week w."""
    parts = [stats[s - 1]] if (s - 1) in stats else []
    parts.append(stats[s][stats[s].week < w])
    d = pd.concat(parts)
    d = d[d.position.isin(POS)].groupby("position")[["targets", "receiving_tds"]].sum()
    rate = d.receiving_tds / d.targets
    return (rate / (d.receiving_tds.sum() / d.targets.sum())).to_dict()


def with_pos_fix(sh: pd.DataFrame, pos_of: dict, mult: dict) -> pd.DataFrame:
    sh = sh.copy()
    m = sh.pid.map(lambda p: mult.get(pos_of.get(p), 1.0))
    adj = sh.rec_share * m
    tot = adj.groupby(sh.team).transform("sum")
    sh["rec_share"] = (adj / tot).where(tot > 0, 0.0)
    return sh


def boot(delta: np.ndarray, groups: np.ndarray, rng, n=2000):
    codes = pd.factorize(groups)[0]
    per = np.bincount(codes, delta, codes.max() + 1)
    b = per[rng.integers(0, len(per), size=(n, len(per)))].sum(1)
    return per.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)


if __name__ == "__main__":
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 220
    plays = T.with_xtd(T.load_plays([2024, 2025, 2026]))
    tg = T.team_game_tds(plays)
    games, _ = load()
    yards = TY.walk_forward(games, load_env(), [2024, 2025, 2026])
    imp = pd.concat([M3.implied_totals(data.load_schedule(y)) for y in (2024, 2025, 2026)], ignore_index=True)
    out = M3.walk_forward(tg, yards, imp, [2024, 2025, 2026])
    stats = {y: data.load_player_weeks(y) for y in (2024, 2025, 2026)}
    allp = pd.concat(stats.values()).sort_values(["season", "week"])
    pos_of = allp.groupby("player_id").position.last().to_dict()
    ev = out[(out.season >= 2025) & (out.week >= 4) & out.rush_td.notna()].copy()
    rng = np.random.default_rng(0)

    # ---- fix 2 at the team level, for a few K
    print(f"TEAM rushing TDs per team-game, n={len(ev)}: MSE skill vs league avg (and vs module 3 as shipped, with 95% CI of the MSE change, negative = better)")
    fac = {k: {} for k in (8.0, 16.0, 32.0, 64.0)}
    lgm = {}
    for (s, w), g in ev.groupby(["season", "week"]):
        for k in fac:
            fac[k][(s, w)] = team_factor(tg, s, w, k)
        past = tg[((tg.season == s) & (tg.week < w)) | (tg.season == s - 1)]
        lgm[(s, w)] = past.rush_td.mean()
    ev["lg"] = [lgm[(s, w)] for s, w in zip(ev.season, ev.week)]
    gk = (ev.season.astype(str) + "-" + ev.week.astype(str) + "-" + np.minimum(ev.off_team, ev.def_team) + np.maximum(ev.off_team, ev.def_team)).to_numpy()
    y = ev.rush_td.to_numpy(float)
    sse = lambda p: ((y - p) ** 2)
    print(f"  league average: skill 0.0000 | module 3 as shipped: skill {1 - sse(ev.rush_td_exp).sum() / sse(ev.lg).sum():.4f}")
    for k in fac:
        f = np.array([fac[k][(s, w)][t] for s, w, t in zip(ev.season, ev.week, ev.off_team)])
        p = ev.rush_td_exp.to_numpy() * f
        d, lo, hi = boot(sse(p) - sse(ev.rush_td_exp.to_numpy()), gk, rng)
        print(f"  x team factor, K={int(k):>2}: skill {1 - sse(p).sum() / sse(ev.lg).sum():.4f} | change in total squared error {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | CIN mean exp {p[(ev.off_team == 'CIN').to_numpy()].mean():.2f} vs actual {y[(ev.off_team == 'CIN').to_numpy()].mean():.2f}")

    # ---- player level: four variants
    rows = []
    for (s, w), g in ev.groupby(["season", "week"]):
        sh = TA.shares(plays, s, w)
        sh_pos = with_pos_fix(sh, pos_of, pos_multiplier(stats, s, w))
        g2 = g.copy()
        g2["rush_td_exp"] = g.rush_td_exp.to_numpy() * np.array([fac[16.0][(s, w)][t] for t in g.off_team])
        v = {"module 4 as shipped": TA.attribute(g, sh), "POS fix (receiving)": TA.attribute(g, sh_pos), "TEAM fix (rushing, K=16)": TA.attribute(g2, sh),
             "POS + TEAM": TA.attribute(g2, sh_pos)}
        for kind, play_type, pid, min_recent in (("rush", "run", "rusher_player_id", 8), ("rec", "pass", "receiver_player_id", 12)):
            pl = plays[(plays.play_type == play_type) & plays[pid].notna()]
            hist = pl[(pl.season >= s - 1) & ((pl.season < s) | (pl.week < w))]
            cand = []
            for team, gg in hist.groupby("posteam"):
                last6 = gg[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(6)
                n = gg.merge(last6, on=["season", "week"]).groupby(pid).size()
                cand.append(pd.DataFrame({"team": team, "pid": n[n >= min_recent].index}))
            act = pl[(pl.season == s) & (pl.week == w)].groupby(["posteam", pid]).td.sum().rename("td").reset_index().rename(columns={"posteam": "team", pid: "pid"})
            D = pd.concat(cand).merge(act, on=["team", "pid"], how="left").fillna({"td": 0.0})
            for name, td in v.items():
                col = "rush_tds" if kind == "rush" else "rec_tds"
                D[name] = D.merge(td[["team", "pid", col]], on=["team", "pid"], how="left")[col].fillna(0.0).to_numpy()
            D["kind"], D["season"], D["week"] = kind, s, w
            D["pos"] = D.pid.map(pos_of)
            rows.append(D)
    R = pd.concat(rows, ignore_index=True)
    names = ["module 4 as shipped", "POS fix (receiving)", "TEAM fix (rushing, K=16)", "POS + TEAM"]
    for kind in ("rush", "rec"):
        d = R[R.kind == kind]
        base = d.td.mean()
        gkp = (d.season.astype(str) + d.week.astype(str) + d.team).to_numpy()
        print(f"\nPLAYER {kind.upper()} TDs per candidate-game, n={len(d)}, mean actual {base:.3f}: skill = MSE reduction vs the mean candidate; change = total squared error vs module 4 as shipped [95% CI] (negative = better)")
        b = ((d.td - base) ** 2).sum()
        for nm in names:
            e = (d.td - d[nm]) ** 2
            dd, lo, hi = boot(e.to_numpy() - ((d.td - d[names[0]]) ** 2).to_numpy(), gkp, rng)
            print(f"  {nm:26s} skill {1 - e.sum() / b:.4f} | mean pred {d[nm].mean():.3f} (bias {d[nm].mean() - base:+.3f}) | change {dd:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    d = R[(R.kind == "rec") & R.pos.isin(POS)]
    print("\nRECEIVING TDs per candidate-game by position: actual vs mean projected")
    print(d.groupby("pos")[["td"] + names].mean().assign(n=d.groupby("pos").size()).to_string())
    d = R[(R.kind == "rush") & R.pos.isin(["QB", "RB", "WR", "TE"])]
    print("\nRUSHING TDs per candidate-game by position:")
    print(d.groupby("pos")[["td"] + names].mean().assign(n=d.groupby("pos").size()).to_string())
    for tm in ("CIN", "MIN", "BAL"):
        d = R[(R.team == tm)]
        print(f"\n{tm} candidates, rush + rec TDs per team-game: actual {d.groupby(['season', 'week']).td.sum().mean():.2f}; " +
              "; ".join(f"{nm} {d.groupby(['season', 'week'])[nm].sum().mean():.2f}" for nm in names))
