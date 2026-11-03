"""Does weighting recent games more improve the predicted carry / target split? (team level, walk-forward)

For every team-game from week 4 on, the shares are built from *earlier* weeks only (usage_shares with a decay),
then compared with how that game's carries / targets were actually split among the team's players. The error is
the total variation distance, 0.5 * sum_i |predicted share_i - actual share_i| over all of the team's players:
the fraction of the team's carries (targets) that the predicted split puts on the wrong player. 0 = perfect, 1 = completely wrong.
It is a whole-team measure - it does not ask whether we picked the right individual player.

    uv run python analysis/share_decay.py [seasons...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from nflfp import data, model

SEASONS = [int(a) for a in sys.argv[1:]] or [2025, 2026]
DECAYS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.0]


def tv_by_team_game(pw: pd.DataFrame, decay: float, min_week: int = 4) -> pd.DataFrame:
    rows = []
    for w in sorted(pw["week"].unique()):
        if w < min_week:
            continue
        sh = model.usage_shares(pw[pw["week"] < w], decay=decay)
        cur = pw[pw["week"] == w]
        for team, g in cur.groupby("team"):
            pred = sh[sh["team"] == team].set_index("player_id")
            row = dict(week=w, team=team)
            for kind, col, share in (("carries", "carries", "carry_share"), ("targets", "targets", "target_share")):
                tot = g[col].sum()
                if tot == 0:
                    row[kind] = np.nan
                    continue
                actual = g.set_index("player_id")[col] / tot
                union = actual.index.union(pred.index)
                row[kind] = 0.5 * (actual.reindex(union, fill_value=0.0) - pred[share].reindex(union, fill_value=0.0)).abs().sum()
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    pd.options.display.float_format = "{:.3f}".format
    res = {}
    for d in DECAYS:
        res[d] = pd.concat([tv_by_team_game(data.load_player_weeks(y), d).assign(season=y) for y in SEASONS], ignore_index=True)
    base = res[1.0]
    n = len(base)
    idx = np.random.default_rng(0).integers(0, n, (2000, n))
    print(f"{n} team-games (week 4+, seasons {SEASONS}). Error = share of the team's carries / targets on the wrong player; "
          "change vs equal weighting with 95% CI (negative = better)\n")
    print(f"{'decay':>6s} {'weight of 1 game back':>22s} | {'carries error':>13s} {'change [95% CI]':>22s} | {'targets error':>13s} {'change [95% CI]':>22s}")
    for d in DECAYS:
        cells = []
        for k in ("carries", "targets"):
            x, b = res[d][k].to_numpy(), base[k].to_numpy()
            ok = ~np.isnan(x) & ~np.isnan(b)
            diff = np.where(ok, x - b, 0.0)
            boots = diff[idx].sum(1) / ok[idx].sum(1)
            cells.append(f"{np.nanmean(x):13.3f} {diff.sum() / ok.sum():+8.3f} [{np.percentile(boots, 2.5):+.3f},{np.percentile(boots, 97.5):+.3f}]")
        label = "equal (current)" if d == 1.0 else ("latest game only" if d == 0 else f"{d:.2f}")
        print(f"{d:6.2f} {label:>22s} | {cells[0]} | {cells[1]}")
    for s in SEASONS:
        m = (base["season"] == s).to_numpy()
        print(f"\n{s}: equal weighting carries {base[m].carries.mean():.3f} / targets {base[m].targets.mean():.3f};  "
              f"decay 0.8: {res[0.8][m].carries.mean():.3f} / {res[0.8][m].targets.mean():.3f}")


if __name__ == "__main__":
    main()
