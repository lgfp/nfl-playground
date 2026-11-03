"""How should a team's receiving points be split among its players: by targets, yards, receiving points, or a mix?

Walk-forward from week 4 (shares built from earlier weeks only, recency decay as in the model). To isolate the *split* from
team-level error, each method divides the team's ACTUAL receiving points in the game among its players using its
predicted shares; the result is compared with each RB/WR/TE's actual receiving points (0 if he didn't play).

    uv run python analysis/receiving_split.py [seasons...] [--decay 0.5]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from nflfp import data, model

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SEASONS = [int(a) for a in args] or [2025, 2026]
DECAY = float(sys.argv[sys.argv.index("--decay") + 1]) if "--decay" in sys.argv else model.DEFAULT_DECAY
S = model.Scoring()
POS = ("RB", "WR", "TE")
COLS = ["player_id", "player_display_name", "position", "team", "week", "targets", "receptions", "receiving_yards", "receiving_tds"]


def with_points(pw: pd.DataFrame) -> pd.DataFrame:
    p = pw[COLS].copy()
    p["rec_pts"] = p.receiving_yards * S.yard + p.receiving_tds * S.td + p.receptions * S.ppr
    return p


def past_shares(past: pd.DataFrame, decay: float) -> pd.DataFrame:
    """Decay-weighted share of each current team's targets / receiving yards / receiving points, per player."""
    p = past.sort_values("week")
    p = p[p.team == p.player_id.map(p.groupby("player_id").team.last())]
    w = np.power(float(decay), p.groupby("team").week.rank(method="dense", ascending=False) - 1)
    for c in ("targets", "receiving_yards", "rec_pts"):
        p[c + "_w"] = p[c] * w
    g = p.groupby(["team", "player_id"])[["targets_w", "receiving_yards_w", "rec_pts_w"]].sum()
    out = g / g.groupby("team").transform("sum")
    return out.rename(columns={"targets_w": "s_targets", "receiving_yards_w": "s_yards", "rec_pts_w": "s_points"}).fillna(0.0)


def main() -> None:
    pd.options.display.float_format = "{:.3f}".format
    methods = {"targets (current)": ("s_targets", 1.0), "yards": ("s_yards", 1.0), "receiving points": ("s_points", 1.0),
               "75% targets + 25% yards": ("mix", 0.75), "50% targets + 50% yards": ("mix", 0.5), "25% targets + 75% yards": ("mix", 0.25)}
    rows = []
    for y in SEASONS:
        pw = with_points(data.load_player_weeks(y))
        pw = pw[pw.position.isin(POS)]
        for w in sorted(pw.week.unique()):
            if w < 4:
                continue
            sh = past_shares(pw[pw.week < w], DECAY)
            for team, g in pw[pw.week == w].groupby("team"):
                total = g.rec_pts.sum()
                if total <= 0:
                    continue
                known = sh.loc[team] if team in sh.index.get_level_values(0) else sh.iloc[0:0]
                ids = known.index.union(pd.Index(g.player_id))
                d = pd.DataFrame(index=ids).join(known, how="left").fillna(0.0)
                d["actual"] = g.set_index("player_id").rec_pts.reindex(ids).fillna(0.0)
                d["position"] = g.set_index("player_id").position.reindex(ids)
                d["position"] = d.position.fillna(pw.groupby("player_id").position.last().reindex(ids))
                for name, (col, lam) in methods.items():
                    sh_i = lam * d.s_targets + (1 - lam) * d.s_yards if col == "mix" else d[col]
                    d["p_" + name] = total * sh_i / max(sh_i.sum(), 1e-9)
                d["season"], d["week"], d["team"] = y, w, team
                rows.append(d.reset_index().rename(columns={"index": "player_id"}))
    df = pd.concat(rows, ignore_index=True)
    df = df[df.position.isin(POS)]
    gid = df.groupby(["season", "week", "team"]).ngroup().to_numpy()
    ng = gid.max() + 1
    rng = np.random.default_rng(0)
    pick = rng.integers(0, ng, (1500, ng))

    def stats_by_group(x, a):
        return np.array([np.bincount(gid, v, ng) for v in (np.ones(len(x)), x, a, x * x, a * a, x * a)])

    def corr_from(st):
        n, sx, sa, sxx, saa, sxa = st
        return (sxa - sx * sa / n) / np.sqrt((sxx - sx * sx / n) * (saa - sa * sa / n))

    a = df.actual.to_numpy()
    base_st = stats_by_group(df["p_targets (current)"].to_numpy(), a)
    print(f"{len(df)} RB/WR/TE player-games over {ng} team-games (week 4+, seasons {SEASONS}, decay {DECAY}); "
          "team total = actual, so this measures the split only\n")
    print(f"{'split by':26s} {'corr':>6s} {'vs targets [95% CI]':>22s} {'MAE':>6s} | mean predicted vs actual points: RB, WR, TE")
    for name in methods:
        x = df["p_" + name].to_numpy()
        st = stats_by_group(x, a)
        boots = np.array([corr_from(st[:, i].sum(1)) - corr_from(base_st[:, i].sum(1)) for i in pick])
        c = corr_from(st.sum(1))
        by = " ".join(f"{df[df.position == p]['p_' + name].mean():5.2f}/{df[df.position == p].actual.mean():4.2f}" for p in POS)
        print(f"{name:26s} {c:6.3f} {c - corr_from(base_st.sum(1)):+8.3f} [{np.percentile(boots, 2.5):+.3f},{np.percentile(boots, 97.5):+.3f}] {np.abs(x - a).mean():6.3f} | {by}")


if __name__ == "__main__":
    main()
