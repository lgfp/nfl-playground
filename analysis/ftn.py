"""FTN (Football Outsiders) DVOA / DYAR from the user's FTN exports in `data/ftn/` (git-ignored: licensed data, never commit it).

Season-level files (end-of-season numbers, recomputed with FTN's current DVOA version):
  Premium-All-Years-DVOA-*.xlsx   one row per team-season, 1977-2025 (total / offense / defense / special teams DVOA, pass and rush splits, weighted DVOA)
  Premium-All-Years-DYAR-*.xlsx   one row per player-season, sheets for QB / RB / WR / TE; nflverse (GSIS) ids on every row from 2019
`prior_season(F)` adds the previous season's numbers to each player-game row of the black-box feature table (group prefix `ftn_`), so nothing from
the season being predicted is used.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DIR = Path("data/ftn")
TEAM = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"}           # FTN code -> nflverse code
SHEETS = {"QB": ("All QB Stats", "pass", "rush"), "RB": ("All RB Stats", "rush", "rec"),
          "WR": ("All WR Stats", "rec", "rush"), "TE": ("All TE Stats", "rec", "rush")}
VOLUME = {"pass": "Passes", "rush": "Runs", "rec": "Passes"}


def _file(pattern: str) -> Path:
    files = sorted(DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {DIR}/ - export it from FTN and put it there")
    return files[-1]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def team_seasons(first: int = 2019) -> pd.DataFrame:
    """One row per team-season: season, team (nflverse code) and DVOA columns (fractions: 0.10 = 10%)."""
    x = pd.read_excel(_file("Premium-All-Years-DVOA*.xlsx"), header=None)
    h = [str(v) for v in x.iloc[0]]
    col = lambda name, start=0: h.index(name, start)                   # several headers repeat (PASS, RUSH, RANK...): find each after its block start
    off0, def0, st0 = col("OFF DVOA"), col("DEF DVOA"), col("ST DVOA")
    t = x.iloc[1:]
    t = t[_num(t[col("YEAR")]) >= first]
    out = pd.DataFrame({"season": _num(t[col("YEAR")]).astype(int), "team": t[col("TEAM")].map(lambda c: TEAM.get(c, c)),
                        "tot": t[col("TOTAL DVOA")], "wei": t[col("WEI DVOA")],
                        "off": t[off0], "off_pass": t[col("PASS", off0)], "off_rush": t[col("RUSH", off0)], "off_wei": t[col("WEI OFF")], "off_var": t[col("VAR", off0)],
                        "def": t[def0], "def_pass": t[col("PASS", def0)], "def_rush": t[col("RUSH", def0)], "def_wei": t[col("WEI DEF")],
                        "def_var": t[col("VARIANCE", def0)], "st": t[st0]})
    for c in out.columns[2:]:
        out[c] = _num(out[c])
    return out.reset_index(drop=True)


def player_seasons(first: int = 2019) -> pd.DataFrame:
    """One row per player-season (nflverse id): DYAR (total value), DVOA (value per play) and volume for passing / rushing / receiving,
    RB rushing success rate and catch rate. NaN where a player has no plays of that kind."""
    f = _file("Premium-All-Years-DYAR*.xlsx")
    parts = []
    for pos, (sheet, a, b) in SHEETS.items():
        p = pd.read_excel(f, sheet_name=sheet, header=3)
        p = p[(_num(p.Year) >= first) & p["GSIS ID"].astype(str).str.startswith("00-")]
        d = pd.DataFrame({"player_id": p["GSIS ID"].astype(str), "season": _num(p.Year).astype(int), "ftn_pos": pos})
        for blk, suf in ((a, ""), (b, ".1")):                           # pandas suffixes the second block's repeated headers with ".1"
            d[f"{blk}_dyar"], d[f"{blk}_dvoa"] = _num(p[f"DYAR{suf}"]), _num(p[f"DVOA{suf}"])
            d[f"{blk}_n"] = _num(p[VOLUME[blk]])
        if "Suc Rate" in p:
            d["succ"] = _num(p["Suc Rate"])
        if "Catch %" in p:
            d["catch"] = _num(p["Catch %"])
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    vol = out[[c for c in out.columns if c.endswith("_n")]].fillna(0).sum(axis=1)
    return out.assign(_v=vol).sort_values("_v").drop_duplicates(["player_id", "season"], keep="last").drop(columns="_v").reset_index(drop=True)


TEAM_COLS = ("off", "off_pass", "off_rush", "off_wei", "def", "def_pass", "def_rush", "def_wei", "st")
PLAYER_COLS = ("pass_dyar", "pass_dvoa", "pass_n", "rush_dyar", "rush_dvoa", "rush_n", "rec_dyar", "rec_dvoa", "rec_n", "succ", "catch")


def prior_season(F: pd.DataFrame) -> pd.DataFrame:
    """F (needs player_id, season, team, opp) plus last season's FTN numbers:
    ftn_t_* the player's team (this season's team, last season's DVOA), ftn_o_* the opponent, ftn_p_* the player himself."""
    ts = team_seasons().assign(season=lambda x: x.season + 1)
    own = ts[["season", "team", *TEAM_COLS]].rename(columns={c: f"ftn_t_{c}" for c in TEAM_COLS})
    opp = ts[["season", "team", *TEAM_COLS]].rename(columns={"team": "opp", **{c: f"ftn_o_{c}" for c in TEAM_COLS}})
    ps = player_seasons().assign(season=lambda x: x.season + 1)
    ps = ps[["player_id", "season", *[c for c in PLAYER_COLS if c in ps]]].rename(columns={c: f"ftn_p_{c}" for c in PLAYER_COLS})
    out = F.merge(own, on=["season", "team"], how="left").merge(opp, on=["season", "opp"], how="left").merge(ps, on=["player_id", "season"], how="left")
    assert len(out) == len(F)
    out.index = F.index
    return out


# ---------------------------------------------------------------------------------------------------------- in-season (weekly) files
def _blocks(rows: list[list], labels: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Sheets laid out as stacked blocks: a header row ('LABEL', 1, 2, ..., n) followed by one row per team. Returns label -> team x week frame."""
    out, cur, weeks = {}, None, None
    for r in rows:
        if r and r[0] in labels:
            cur, weeks = r[0], [w for w in r[1:] if isinstance(w, int)]
            out[cur] = {}
        elif cur and r and isinstance(r[0], str) and len(r[0]) <= 3:
            out[cur][TEAM.get(r[0], r[0])] = dict(zip(weeks, r[1:1 + len(weeks)]))
    return {k: pd.DataFrame(v).T for k, v in out.items()}


def weekly_games(season: int = 2026) -> pd.DataFrame:
    """One row per team-game of `season` from the Premium Splits export: that single game's DVOA (total, offense, defense, special teams,
    pass / rush offense and defense) and the opponent. Single-game values use FTN's opponent adjustments as of the export date."""
    import openpyxl
    wb = openpyxl.load_workbook(_file(f"{season}-Premium-Splits*.xlsx"), read_only=True, data_only=True)
    grab = lambda name: [list(r) for r in wb[name].iter_rows(values_only=True)]
    a = _blocks(grab("Week by Week"), ("TOTAL", "OFFENSE", "DEFENSE", "SPECIAL TEAMS", "TEAM"))
    b = _blocks(grab("teamRunPassWeek"), ("PASSING OFFENSE", "PASSING DEFENSE", "RUSHING OFFENSE", "RUSHING DEFENSE"))
    names = {"TOTAL": "tot", "OFFENSE": "off", "DEFENSE": "def", "SPECIAL TEAMS": "st", "PASSING OFFENSE": "off_pass", "PASSING DEFENSE": "def_pass",
             "RUSHING OFFENSE": "off_rush", "RUSHING DEFENSE": "def_rush"}
    long = []
    for lab, col in names.items():
        m = (a | b)[lab]
        s = m.stack(future_stack=True).rename(col)
        long.append(pd.to_numeric(s, errors="coerce"))
    g = pd.concat(long, axis=1)
    g.index.names = ["team", "week"]
    opp = a["TEAM"].stack().rename("opp").map(lambda c: TEAM.get(c, c))
    opp.index.names = ["team", "week"]
    g = g.join(opp, how="left").reset_index().dropna(subset=["tot"])
    g["week"] = g.week.astype(int)
    return g.assign(season=season)[["season", "week", "team", "opp", *names.values()]].sort_values(["team", "week"]).reset_index(drop=True)


def to_date(g: pd.DataFrame) -> pd.DataFrame:
    """For every team and week 1..max+1: mean single-game DVOA over the team's games BEFORE that week (NaN before its first game), and games played."""
    cols = [c for c in g.columns if c not in ("season", "week", "team", "opp")]
    out = []
    for t, x in g.groupby("team"):
        x = x.sort_values("week")
        for w in range(1, int(g.week.max()) + 2):
            past = x[x.week < w]
            out.append({"season": x.season.iloc[0], "week": w, "team": t, "n_games": len(past), **past[cols].mean().to_dict()})
    return pd.DataFrame(out)
