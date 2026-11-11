"""Does playing a teammate of the opponent's players hedge, and what does it cost? Win probability with correlated scores.

1. Correlations of score surprises (actual minus Sleeper's projection, scaled by how noisy that projection level is), same team and opposing teams, from every team-game
   2021-26: QB-WR, QB-TE, QB-RB, WR-TE, WR-RB, TE-RB, and the same players across the two sides of a game.
2. A Gaussian copula over the connected players (default: the opponent's QB, RB and TE and your WR on that team, plus a receiver on the other team of the game),
   each player's own outcome distribution taken from players with similar projections. Everyone else is independent.
3. P(win) for your current lineup and for versions that add the hedge.

    uv run python -m analysis.ml_hedge <username> <week> "Hedge Player" "Opp QB"
"""
from __future__ import annotations

import itertools
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis import ml_data as MD
from nflfp import sleeper

warnings.filterwarnings("ignore")
N = 80000
THRESH = {"QB": 12.0, "RB": 8.0, "WR": 8.0, "TE": 5.0}


def residuals(H: pd.DataFrame) -> pd.DataFrame:
    H = H[H.sl.notna() & H.pts.notna() & (H.sl >= H.position.map(THRESH))].copy()
    H["bin"] = H.groupby("position").sl.transform(lambda v: pd.qcut(v, 5, labels=False, duplicates="drop"))
    H["z"] = (H.pts - H.sl) / H.groupby(["position", "bin"]).apply(lambda g: (g.pts - g.sl).std()).reindex(pd.MultiIndex.from_frame(H[["position", "bin"]])).to_numpy()
    return H


def pair_corr(H: pd.DataFrame, pa: str, pb: str, opposite: bool) -> tuple[float, int]:
    a = H[H.position == pa][["season", "week", "team", "opp", "player_id", "z"]]
    b = H[H.position == pb][["season", "week", "team", "opp", "player_id", "z"]]
    m = a.merge(b, left_on=["season", "week", "opp" if opposite else "team"], right_on=["season", "week", "team"], suffixes=("_a", "_b"))
    if not opposite:
        m = m[m.player_id_a != m.player_id_b]
    if pa == pb:
        m = m[m.player_id_a < m.player_id_b]
    return float(np.corrcoef(m.z_a, m.z_b)[0, 1]), len(m)


def nearest_psd(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    out = (v * np.clip(w, 1e-3, None)) @ v.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def marginal(H: pd.DataFrame, pos: str, mean: float, u: np.ndarray, k: int = 200) -> np.ndarray:
    """Outcome at quantile u for a player of this position projected at `mean`, from the k most similar projections (outcomes shifted to the mean)."""
    d = H[H.position == pos]
    near = d.iloc[(d.sl - mean).abs().argsort()[:k]]
    x = np.sort(np.maximum(near.pts.to_numpy() + (mean - near.sl.mean()), 0.0))
    return np.quantile(x, u)


if __name__ == "__main__":
    pd.options.display.width = 220
    user, week, hedge, oppqb = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    F = pd.read_pickle(MD.OUT)
    H = residuals(F[F.season >= 2021])
    R = pd.read_pickle("/tmp/ml_week_rows.pkl")
    dump = sleeper.load_players()
    uid = sleeper.user_id(user)
    lg = sleeper.get(f"user/{uid}/leagues/nfl/2026")[0]
    rs = sleeper.get(f"league/{lg['league_id']}/rosters")
    me = next(r for r in rs if r["owner_id"] == uid)
    mu = sleeper.get(f"league/{lg['league_id']}/matchups/{week}")
    mine = next(x for x in mu if x["roster_id"] == me["roster_id"])
    opp = next(x for x in mu if x["matchup_id"] == mine["matchup_id"] and x["roster_id"] != me["roster_id"])
    nm = lambda s: f'{dump.get(s, {}).get("first_name", "")} {dump.get(s, {}).get("last_name", "")}'.strip()
    row = R.set_index("sleeper_id")
    mean_of = lambda sid: float(row.loc[sid, "blend"]) if sid in row.index and pd.notna(row.loc[sid, "blend"]) else (float(row.loc[sid, "ensemble"]) if sid in row.index else np.nan)
    pos_of = lambda sid: dump[sid]["position"]
    team_of = lambda sid: dump[sid]["team"]
    # players
    mine_all = {nm(s): s for s in me["players"] if pos_of(s) in MD.POS if s in row.index}
    opp_start = {nm(s): s for s in opp["starters"] if dump.get(s, {}).get("position") in MD.POS and s in row.index}
    team = team_of(mine_all[hedge])
    cluster_opp = [n for n, s in opp_start.items() if team_of(s) == team]
    other_side = [n for n, s in mine_all.items() if row.loc[s, "opp"] == team]
    print(f"hedge: {hedge} ({team}); the opponent's {team} players: {cluster_opp}; your players against {team}: {other_side}")
    # correlations
    cl = [(oppqb, "opp"), (hedge, "me")] + [(n, "opp") for n in cluster_opp if n != oppqb] + [(n, "me") for n in other_side]
    names = [n for n, _ in cl]
    sid = {**mine_all, **opp_start}
    pos = [pos_of(sid[n]) for n in names]
    tm = [team_of(sid[n]) for n in names]
    n = len(names)
    C = np.eye(n)
    cache = {}
    for i, j in itertools.combinations(range(n), 2):
        key = (pos[i], pos[j], tm[i] == tm[j])
        if key not in cache:
            c, cnt = pair_corr(H, pos[i], pos[j], opposite=tm[i] != tm[j])
            cache[key] = (c, cnt)
        C[i, j] = C[j, i] = cache[key][0]
    print("\nscore-surprise correlations used (same team = same side of the ball's team; n = team-games):")
    for (a, b, same), (c, cnt) in sorted(cache.items(), key=lambda kv: -abs(kv[1][0])):
        print(f"  {a}-{b} {'same team' if same else 'opposing teams'}: {c:+.3f} (n={cnt})")
    C = nearest_psd(C)
    rng = np.random.default_rng(0)
    Z = rng.multivariate_normal(np.zeros(n), C, size=N)
    U = norm.cdf(Z)
    draws = {names[i]: marginal(H, pos[i], mean_of(sid[names[i]]), U[:, i]) for i in range(n)}

    def solo(name, s):
        return marginal(H, pos_of(s), mean_of(s), rng.random(N))

    kdef = lambda: np.maximum(rng.normal(8.0, 4.2, N), 0)
    opp_total = np.zeros(N)
    for nme, s in opp_start.items():
        opp_total += draws[nme] if nme in draws else solo(nme, s)
    opp_total += kdef() + kdef()                                           # their K and DEF
    base = {nme: (draws[nme] if nme in draws else solo(nme, s)) for nme, s in mine_all.items()}
    cur = ["Dak Prescott", "James Cook III", "Chase Brown", "CeeDee Lamb", "Chris Olave", "Travis Kelce", "Blake Corum", "Justin Jefferson"]
    cur_total = lambda players: sum(base[p] for p in players) + kdef() + kdef()
    print(f"\nopponent expected total {opp_total.mean():.1f} (sd {opp_total.std():.1f}); your current lineup {cur_total(cur).mean():.1f}")
    lineups = {"current lineup": cur}
    for out in ("Blake Corum", "Justin Jefferson", "Chase Brown", "James Cook III"):
        lineups[f"{hedge} in for {out}"] = [p if p != out else hedge for p in cur]
    # independent version of the hedge player (cuts the correlation), to show what the correlation is worth
    indep = {"hedge player independent": marginal(H, pos_of(mine_all[hedge]), mean_of(mine_all[hedge]), rng.random(N))}
    print(f"\n{'lineup':44s} {'P(win)':>7s} {'mean':>6s} {'sd margin':>10s} | independent P(win)")
    for lab, pl in lineups.items():
        t = sum(base[p] for p in pl) + 16.0
        marg = t - (opp_total)
        ind = None
        if hedge in pl:
            t2 = sum(base[p] if p != hedge else indep["hedge player independent"] for p in pl) + 16.0
            ind = float((t2 > opp_total).mean())
        print(f"  {lab:42s} {float((t > opp_total).mean()):7.3f} {t.mean():6.1f} {marg.std():10.1f} | " + (f"{ind:.3f}" if ind is not None else "-"))
    # what the hedge does in the scenarios that matter: the opponent's QB booms or busts
    q = draws[oppqb]
    lo, hi = np.quantile(q, [1 / 3, 2 / 3])
    print(f"\n{oppqb}: projected {mean_of(sid[oppqb]):.1f}; thirds of his outcomes: under {lo:.1f}, {lo:.1f}-{hi:.1f}, over {hi:.1f}")
    for lab, mask in (("bust (bottom third)", q <= lo), ("middle third", (q > lo) & (q <= hi)), ("boom (top third)", q > hi)):
        row_ = [f"{hedge} avg {draws[hedge][mask].mean():.1f}", f"opponent avg {opp_total[mask].mean():.1f}"]
        for l2, pl in (("current", cur), (f"{hedge} for Corum", [p if p != "Blake Corum" else hedge for p in cur])):
            t = sum(base[p] for p in pl) + 16.0
            row_.append(f"P(win) {l2}: {float((t > opp_total)[mask].mean()):.3f}")
        print(f"  {lab:22s} " + " | ".join(row_))
    print(f"\nmean outcomes: {hedge} {draws[hedge].mean():.1f} (blend {mean_of(mine_all[hedge]):.1f}), Corum {base['Blake Corum'].mean():.1f}, Jefferson {base['Justin Jefferson'].mean():.1f}")
