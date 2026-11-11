"""Lead backs who lose the goal-line work to a teammate ("vultured"): how often does it happen, does the split persist, and are they over-projected?

A case = an RB game where, going in, he had at least half of his team's carries but his share of carries inside the 5 was 20+ points lower (recency-weighted).
Projections: Sleeper's, and the tree model trained only on earlier seasons (2023 from 2021-22, ..., 2026 from 2021-25).

    uv run python -m analysis.ml_vulture
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_diagnose import trees
from analysis.ml_model import feature_sets

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pd.options.display.width = 230
    p = MD.pbp()
    r = p[(p.play_type == "run") & p.rusher_player_id.notna()].assign(gl=lambda x: (x.yardline_100 <= 5).astype(float))
    pg = r.groupby(["season", "week", "posteam", "rusher_player_id", "rusher_player_name"]).agg(car=("td", "size"), gl=("gl", "sum"), td=("td", "sum")).reset_index()
    tg = pg.groupby(["season", "week", "posteam"]).agg(t_car=("car", "sum"), t_gl=("gl", "sum")).reset_index()
    pg = pg.merge(tg, on=["season", "week", "posteam"]).rename(columns={"rusher_player_id": "player_id", "posteam": "team"})

    print("SEATTLE: Kenneth Walker and Zach Charbonnet")
    for s in (2024, 2025):
        sea = pg[(pg.season == s) & (pg.team == "SEA")]
        both = sea[sea.rusher_player_name.isin(["K.Walker", "Z.Charbonnet"])]
        weeks_w = set(sea[sea.rusher_player_name == "K.Walker"].week)
        t = sea[sea.week.isin(weeks_w)]
        g = t[t.rusher_player_name.isin(["K.Walker", "Z.Charbonnet"])].groupby("rusher_player_name").agg(games=("week", "nunique"), carries=("car", "sum"), goal_line=("gl", "sum"), rush_td=("td", "sum"))
        tot = t.drop_duplicates(["week"])[["t_car", "t_gl"]].sum()
        g["carry_share"], g["goal_line_share"] = g.carries / tot.t_car, g.goal_line / tot.t_gl
        print(f"  {s}, the {len(weeks_w)} games Walker played:"); print(g.round(2).to_string())

    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    feats = feature_sets(F)["avail+qb+cal"]
    parts = []
    for ts in (2023, 2024, 2025, 2026):
        te = F[F.season == ts].copy()
        te["model"] = trees(F[F.season.between(2021, ts - 1)], feats)(te)
        parts.append(te)
    R = pd.concat(parts)
    R = R.merge(pg[["season", "week", "player_id", "car", "gl", "td", "t_car", "t_gl"]], on=["season", "week", "player_id"], how="left")
    R["gap"] = R.px_gl_share_ew - R.carry_share_ew
    w = R[(R.player_display_name == "Kenneth Walker III") & R.season.isin([2024, 2025])]
    print("\nWalker, projected vs actual (model = trees trained on earlier seasons only):")
    print(w.groupby("season").agg(games=("pts", "size"), actual=("pts", "mean"), sleeper=("sl", "mean"), model=("model", "mean"), carry_share_going_in=("carry_share_ew", "mean"),
                                  goal_line_share_going_in=("px_gl_share_ew", "mean"), rush_td=("td", "sum")).round(2).to_string())
    for s in (2024, 2025):
        x = w[w.season == s]
        print(f"  {s}: Sleeper error {np.mean(x.sl - x.pts):+.2f} a game | model error {np.mean(x.model - x.pts):+.2f} | below Sleeper's projection in {(x.pts < x.sl).sum()} of {x.sl.notna().sum()} games")

    rb = R[(R.position == "RB") & (R.sl >= 8) & (R.carry_share_ew >= 0.5)].dropna(subset=["gap"]).copy()
    rb["case"] = rb.gap <= -0.20
    rb["e_sl"], rb["e_m"] = rb.sl - rb.pts, rb.model - rb.pts
    print(f"\nALL LEAD BACKS 2023-26 (Sleeper projects 8+, half or more of the carries going in): {len(rb)} games; vultured cases: {rb.case.sum()} games, "
          f"{rb[rb.case].groupby(['season', 'player_display_name']).ngroups} player-seasons")
    def line(d, lab):
        se = lambda v: v.std() / np.sqrt(len(v))
        print(f"  {lab:34s} n={len(d):4d} | actual {d.pts.mean():5.2f} | rush TDs/game {d.td.mean():.2f} | Sleeper {d.sl.mean():5.2f} (error {d.e_sl.mean():+.2f} +/-{2 * se(d.e_sl):.2f}) | "
              f"model {d.model.mean():5.2f} (error {d.e_m.mean():+.2f} +/-{2 * se(d.e_m):.2f})")
    line(rb[rb.case], "vultured (goal-line gap <= -20 pts)")
    line(rb[~rb.case], "other lead backs")
    line(rb[rb.gap <= -0.35], "heavily vultured (gap <= -35 pts)")
    line(rb[rb.gap >= 0.15], "goal-line share ABOVE carry share")
    for s, d in rb[rb.case].groupby("season"):
        print(f"     {s}: n={len(d):3d} Sleeper error {d.e_sl.mean():+.2f} | model error {d.e_m.mean():+.2f}")

    # does the split persist into the game being projected?
    g = rb[rb.t_gl >= 1].copy()
    g["act_gap"] = g.gl / g.t_gl - g.car / g.t_car
    print(f"\nDOES THE SPLIT PERSIST? lead-back games where the team had a carry inside the 5 (n={len(g)}):")
    print(f"  correlation of the gap going in with the gap in that game: {np.corrcoef(g.gap, g.act_gap)[0, 1]:+.3f}")
    for lab, d in (("vultured going in", g[g.case]), ("others", g[~g.case])):
        print(f"  {lab:18s}: goal-line share going in {d.px_gl_share_ew.mean():.2f} (carry share {d.carry_share_ew.mean():.2f}) -> in the game: goal-line share {(d.gl.sum() / d.t_gl.sum()):.2f}, carry share {(d.car.sum() / d.t_car.sum()):.2f}")
    print(f"  team carries inside the 5 per game: {tg[tg.season >= 2023].t_gl.mean():.2f}; share of team-games with none: {(tg[tg.season >= 2023].t_gl == 0).mean():.2f}")

    # a sharper measure: carries inside the 5 summed over his previous 8 games (not an average of per-game shares, which skips games without any)
    h = pg.sort_values(["player_id", "season", "week"]).copy()
    for c in ("car", "gl", "t_car", "t_gl"):
        h[f"{c}_8"] = h.groupby("player_id")[c].transform(lambda v: v.shift(1).rolling(8, min_periods=4).sum())
    h["gap8"] = h.gl_8 / h.t_gl_8.where(h.t_gl_8 >= 6) - h.car_8 / h.t_car_8
    h["cs8"] = h.car_8 / h.t_car_8
    S = R.merge(h[["season", "week", "player_id", "gap8", "cs8", "t_gl_8"]], on=["season", "week", "player_id"], how="left")
    S = S[(S.position == "RB") & (S.sl >= 8) & (S.cs8 >= 0.45)].dropna(subset=["gap8"]).copy()
    S["e_sl"], S["e_m"] = S.sl - S.pts, S.model - S.pts
    print(f"\nSHARPER MEASURE (sums over his last 8 games, team had 6+ carries inside the 5, he had 45%+ of carries): {len(S)} games")
    for lab, d in (("gap <= -20 pts", S[S.gap8 <= -0.20]), ("gap <= -30 pts", S[S.gap8 <= -0.30]), ("the rest", S[S.gap8 > -0.20])):
        se = lambda v: 2 * v.std() / np.sqrt(len(v))
        print(f"  {lab:15s} n={len(d):4d}, {d.groupby(['season', 'player_display_name']).ngroups:3d} player-seasons | actual {d.pts.mean():5.2f} | rush TDs {d.td.mean():.2f} | "
              f"Sleeper error {d.e_sl.mean():+.2f} +/-{se(d.e_sl):.2f} | model error {d.e_m.mean():+.2f} +/-{se(d.e_m):.2f}")
    g2 = S[S.t_gl >= 1]
    print(f"  persistence: correlation of this gap with the gap in the game {np.corrcoef(g2.gap8, g2.gl / g2.t_gl - g2.car / g2.t_car)[0, 1]:+.3f}; "
          f"for gap <= -20: goal-line share in the game {g2[g2.gap8 <= -0.2].gl.sum() / g2[g2.gap8 <= -0.2].t_gl.sum():.2f} vs carry share {g2[g2.gap8 <= -0.2].car.sum() / g2[g2.gap8 <= -0.2].t_car.sum():.2f}")
    x = S[S.gap8 <= -0.20].groupby(["season", "player_display_name"]).agg(games=("pts", "size"), gap=("gap8", "mean"), actual=("pts", "mean"), sleeper_err=("e_sl", "mean"), model_err=("e_m", "mean"))
    x = x[x.games >= 4].sort_values("games", ascending=False)
    print(f"  player-seasons with 4+ such games (n={len(x)}): over-projected by Sleeper in {(x.sleeper_err > 0).sum()}, under in {(x.sleeper_err < 0).sum()}; by the model in {(x.model_err > 0).sum()} / {(x.model_err < 0).sum()}")
    print(x.round(2).head(14).to_string())

    ps = rb[rb.case].groupby(["season", "player_display_name"]).agg(games=("pts", "size"), gap=("gap", "mean"), actual=("pts", "mean"), sleeper_err=("e_sl", "mean"), model_err=("e_m", "mean"))
    ps = ps[ps.games >= 4].sort_values("games", ascending=False)
    print(f"\nplayer-seasons with 4+ vultured games (n={len(ps)}): over-projected by Sleeper in {(ps.sleeper_err > 0).sum()}, under-projected in {(ps.sleeper_err < 0).sum()}")
    print(ps.round(2).head(16).to_string())
