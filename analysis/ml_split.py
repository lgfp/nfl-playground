"""One model for all positions, or one per position? Trained 2021-25, tested on 2026.

  pooled        the current model: 154 inputs, trees and neural net shared across positions (ridge already per position)
  split         separate ridge + trees + neural net per position, same 154 inputs
  split, all    separate models, every input we have built
  split, chosen separate models; each position's input groups picked by forward selection on 2025 (trained 2021-24), then refitted on 2021-25
  pooled+split  average of pooled and split-chosen

    uv run python -m analysis.ml_split
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_model import ID, POOL_MIN, feature_sets, fit_predict, gbm, metrics, vs
from analysis.ml_ranking import evaluate

warnings.filterwarnings("ignore")
MIN_GAIN = 0.01          # a group must lower validation MAE by this much to be added


def split_predict(tr, te, feats_by_pos) -> pd.DataFrame:
    parts = []
    for pos in MD.POS:
        a, b = tr[tr.position == pos], te[te.position == pos]
        parts.append(fit_predict(a, b, feats_by_pos[pos]))
    return pd.concat(parts).loc[te.index]


def choose_groups(F, base, groups) -> dict:
    """Forward selection per position: train trees on 2021-24, score MAE on 2025 player-games Sleeper projected at 4+."""
    chosen = {}
    for pos in MD.POS:
        a = F[F.season.between(2021, 2024) & (F.position == pos)]
        v = F[(F.season == 2025) & (F.position == pos) & (F.sl >= POOL_MIN)]
        score = lambda cols: float(np.abs(np.mean([gbm("squared_error", s).fit(a[cols], a.pts).predict(v[cols]) for s in range(3)], axis=0) - v.pts).mean())
        cur, best, left, log = list(base), score(base), dict(groups), []
        while left:
            trial = {g: score(cur + cols) for g, cols in left.items()}
            g = min(trial, key=trial.get)
            if best - trial[g] < MIN_GAIN:
                break
            log.append(f"+{g} {best:.3f}->{trial[g]:.3f}")
            cur, best = cur + left.pop(g), trial[g]
        chosen[pos] = cur
        print(f"  {pos}: validation n={len(v)}, base MAE {score(base):.3f}; " + ("; ".join(log) if log else "no group helped") + f" | {len(cur)} inputs", flush=True)
    return chosen


if __name__ == "__main__":
    pd.options.display.width = 230
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    tr, te = F[F.season.between(2021, 2025)], F[F.season == 2026].copy()
    sets = feature_sets(F)
    grp = lambda c: c.split("_")[0]
    groups = {g: [c for c in F.columns if grp(c) == g] for g in MD.GROUPS}
    everything = sets["avail"] + [c for g in groups.values() for c in g]
    print("groups:", {g: len(c) for g, c in groups.items()}, "| rows per position 2021-25:", tr.position.value_counts().to_dict())
    P = te[ID].copy()
    P["pooled"] = fit_predict(tr, te, sets["avail+qb+cal"])["ensemble"]
    print("trained pooled", flush=True)
    P["split"] = split_predict(tr, te, {p: sets["avail+qb+cal"] for p in MD.POS})["ensemble"]
    print("trained split", flush=True)
    P["split_all"] = split_predict(tr, te, {p: everything for p in MD.POS})["ensemble"]
    print("trained split, all inputs", flush=True)
    print("choosing input groups per position on 2025:")
    chosen = choose_groups(F, sets["avail"], groups)
    P["split_chosen"] = split_predict(tr, te, chosen)["ensemble"]
    P["pooled+split"] = (P.pooled + P.split_chosen) / 2
    for c in ("pooled", "split", "split_all", "split_chosen", "pooled+split"):
        P[f"blend_{c}"] = 0.5 * P.sl + 0.5 * P[c]
    P.to_pickle("/tmp/ml_split.pkl")
    P = P.dropna(subset=["sl"])
    pool = P[P.sl >= POOL_MIN]
    y = pool.pts.to_numpy(float)
    labels = {"pooled": f"pooled, {len(sets['avail+qb+cal'])} inputs", "split": "split by position, same inputs", "split_all": f"split, all {len(everything)} inputs",
              "split_chosen": "split, inputs chosen per position on 2025", "pooled+split": "average of pooled and split-chosen"}
    rows = [dict(model="Sleeper (public)", **metrics(y, pool.sl.to_numpy(float)))]
    for c, lab in labels.items():
        d, lo, hi = vs(pool, c)
        p_, pl, ph = vs(pool, c, "pooled") if c != "pooled" else (0.0, 0.0, 0.0)
        rows.append(dict(model=lab, **metrics(y, pool[c].to_numpy(float)), dMAE_vs_Sleeper=d, lo=lo, hi=hi, dMAE_vs_pooled=p_, p_lo=pl, p_hi=ph))
    for c in ("pooled", "split_chosen", "pooled+split"):
        d, lo, hi = vs(pool, f"blend_{c}")
        rows.append(dict(model=f"50-50 with Sleeper: {labels[c]}", **metrics(y, pool[f"blend_{c}"].to_numpy(float)), dMAE_vs_Sleeper=d, lo=lo, hi=hi))
    print(f"\n2026 weeks 1-9, n={len(pool)}")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))
    print("\nby position: MAE (change vs pooled [95% interval])")
    for pos in MD.POS:
        g = pool[pool.position == pos]
        out = [f"{pos} n={len(g)}: Sleeper {np.abs(g.sl - g.pts).mean():.3f} | pooled {np.abs(g.pooled - g.pts).mean():.3f}"]
        for c in ("split", "split_all", "split_chosen"):
            d, lo, hi = vs(g, c, "pooled")
            out.append(f"{c} {np.abs(g[c] - g.pts).mean():.3f} ({d:+.3f} [{lo:+.3f}, {hi:+.3f}])")
        print("  " + " | ".join(out))
    rp = P[(P.sl >= POOL_MIN) | (P.pooled >= POOL_MIN)]
    cols = {"Sleeper (public)": "sl", **{lab: c for c, lab in labels.items()}, "50-50 with Sleeper: pooled": "blend_pooled", "50-50 with Sleeper: split-chosen": "blend_split_chosen"}
    print("\nranking within position and week:")
    print(evaluate(rp, cols, boots=500).to_string(index=False, float_format="{:.3f}".format))
    for pos in MD.POS:
        e = evaluate(rp[rp.position == pos], {k: cols[k] for k in ("Sleeper (public)", labels["pooled"], labels["split_chosen"])}, boots=200)
        print(f"  {pos}: pairs right " + " | ".join(f"{r.model}: {r.pairs:.3f}" for _, r in e.iterrows()))
