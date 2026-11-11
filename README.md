# nflfp - opponent-adjusted expected fantasy points

```
uv run nflfp                      # next unplayed games of the 2026 season
uv run nflfp --week 9 --window 4  # a specific week, usage shares from last 4 games
uv run nflfp --csv out.csv        # full player table
uv run pytest
uv run python -m nflfp.backtest           # backtest team rush vs receiving predictions (train on all other weeks; --forward = earlier weeks only)
```

Data comes from [nflverse](https://github.com/nflverse/nflverse-data) (weekly player stats + schedule), cached in `.cache/`.
The current season's files are re-downloaded when older than 24h (nflverse updates nightly, on game days a bit more often);
finished seasons never expire; `--refresh` forces a download. If a refresh fails, the older copy is used with a warning.

## Method

1. **Raw**: per team-game, sum rushing yds/TDs and receiving yds/TDs/receptions (passing yds/TDs = receiving).
2. **Opponent adjustment**: for each component, `output = league_avg x OFF[team] x DEF[opp]`. OFF and DEF are
   solved by alternating updates until converged, so a defense that faced strong offenses is credited.
   Shrinkage toward 1.0 (8 pseudo-games, chosen by backtest) tames small samples. Rushing TDs are not
   adjusted at all (league average): backtests showed they're unpredictable from team strength.
   `league_avg x DEF[team]` is what that defense would give up to an average offense.
3. **Matchup**: expected output = `league_avg x DEF[opp] x OFF[team]` (the opposing offense's deviation from average;
   `--no-offense-adj` to skip).
4. **Player split**: rushing points by share of team carries, receiving points by share of targets,
   all QB passing points to the starter (most pass attempts in the team's latest game) (shares use the player's current team; `--window N` for recent usage).

Game environment (on by default): the team rushing/receiving points get a small linear correction fitted on the
season's earlier games (`nflfp/lines.py`): point spread and game total for both, plus game-time wind for receiving
and QB passing (0 in domes). Backtests: R2 ~0.04 -> ~0.06 (rushing) and ~0.13 -> ~0.17 (receiving). Almost all of that is
the lines; wind adds little once the total is in the model (Vegas totals already price in wind). If a line is not posted
yet or there is no wind forecast, that team is left unadjusted and the run says so. `--no-lines` turns it all off,
`--no-wind` drops wind, `--use-lines fav` (or `total`) picks line features. Wind comes from Open-Meteo (see Weather);
the backtest uses observed wind, so it is a little optimistic compared with a forecast.

Injuries: players Sleeper (public API, no account) lists as Out or IR are dropped from the carry/target shares, so
their usage goes to teammates in proportion to the teammates' own usage; if the starting QB is out, the passing points go
to the remaining QB with the most attempts. Status is as of now, so it only suits projecting the upcoming week;
`--no-injuries` turns it off, `--refresh-injuries` re-downloads the player list (cached 24h). Not backtested: there is no
historical injury data. It also doesn't use with/without-player splits, just proportional redistribution.

Sleeper league tool: `uv run python -m nflfp.league <username> [--week N] [--margin 0.20] [--vs lineup|roster]` reads your
league from Sleeper's public API (rosters, matchup, scoring settings, positions - no login) and prints your matchup with our
projections, a suggested best lineup, and the *available* QB/RB/WR/TE (on nobody's roster) projected at least 20% above a
player in that lineup at the same position (`--vs roster` compares with your whole roster, bench included). Projections
are ours, not Sleeper's; K and D/ST are not modeled.

Recency (on by default): the carry/target shares weight recent games more - a game k games back counts 0.5**k as much
(`--decay`, 1.0 = all games equal; used by `nflfp` and `nflfp.league`). `uv run python analysis/share_decay.py` backtests it at
the team level (walk-forward; error = share of the team's carries/targets put on the wrong player): carries 0.300 -> 0.263 and
targets 0.336 -> 0.312 vs. equal weighting, and better than using only the latest game. Team totals are unaffected.

Default scoring: 0.1/yd rushing/receiving, 0.04/yd passing, 6 per rushing/receiving TD, 4 per passing TD, 0.5 PPR (all overridable, see `--help`).

The backtest (`python -m nflfp.backtest`) reports error (MAE, RMSE, R2) and *direction*: within each week, how often a
higher prediction is a higher outcome (pairs ordered right, top half vs bottom half, weekly rank correlation), against a
matchup-free "team average" baseline. Direction is the fairer way to read this model: it ranks team-games better than it
predicts their points.

Limitations: no injuries/game script/home field, usage shares are volume-only (not efficiency), no kickers/DST.

## Weather

`nflfp/weather.py` pulls game-time weather (3h after kickoff, at the home stadium) from
[Open-Meteo](https://open-meteo.com): free, no API key, non-commercial use, historical archive + 16-day forecast.
The sandbox must allow `api.open-meteo.com` and `archive-api.open-meteo.com`; requests retry on connection resets.

`uv run python analysis/weather_regression.py [season] [game_ids to drop]` regresses game-level rushing/receiving
points (and the model's residuals) on roof, temperature, wind, rain and snow. Neutral-site games are excluded.


## Module 1 - expected team yards (`nflfp/yards.py`)

The first piece of the planned refactor: expected team **rushing** and **passing yards** for a week, with an iterative
opponent-adjusted multiplicative model (`MU * OFF[team] * DEF[opp]`). For each game it returns eight numbers: {rush, pass} x
{plain, ridge} x {no Vegas, Vegas} (`rush_plain`, `rush_ridge`, `rush_plain_vegas`, `rush_ridge_vegas`, same for `pass_`).
Ridge pulls every team toward a target of 30% its last-season team average and 70% the current league average (adjustable via
`YardsConfig.prior_weight`, default 0.3), with `ridge_strength` = 8 pseudo-games. The Vegas adjustment moves yards from passing to
rushing as a team is favoured (`lines_mode="transfer"`; "free" / "relative" are alternatives). Everything uses only earlier games.

    from nflfp import yards as Y
    out = Y.project_week(games, fixtures, 2026, 9)      # games: Y.team_game_yards(player_weeks), fixtures: Y.fixtures_from_schedule(schedule)

`uv run python -m analysis.yards_backtest` runs the walk-forward backtest (2025-2026), `analysis.yards_sensitivity` the parameter grids.
The older modules (`model.py`, `pipeline.py`, ...) are untouched until later modules replace them.

### Module 1b - the same eight numbers from plays (`nflfp/plays.py`)

Splits a team's yards into plays x yards per play, opponent-adjusts each part with the same ridge engine, and puts the Vegas
adjustment on play volume. `data.load_pbp(season)` supplies the play-by-play. `uv run python -m analysis.plays_backtest` compares it
with module 1 on the same actual yards. Result (2025-26, weeks 4+): the two tie on accuracy (rush R2 0.046 vs 0.042, pass 0.094 vs 0.094);
the play-level version additionally exposes expected plays and yards per play (`rush_att_ridge`, `pass_ypa_ridge`, ...) for later modules.

### One idea at a time (`analysis/ablation.py`)

`uv run python -m analysis.ablation` starts from the simplest model (a team's own average yards to date) and adds exactly one idea at a
time - ridge, opponent adjustment, Vegas lines, play-by-play, wind - reporting the change in R2 with a confidence interval.
`--stacked` does the same with the ridge model as the baseline (the other ideas, with their own averages ridged, added one at a time).

`uv run python -m analysis.stack` is round 2: rush = ridge + Vegas (free slopes), pass = ridge + iterative opponent adjustment as the base,
then each remaining idea added alone on top (sequential corrections, each slope fitted on the previous stage's residuals).
`uv run python -m analysis.stack --round3` (passing only): wind joins the pass base (ridge + opponent + wind), then Vegas and play-by-play are added alone.
`uv run python -m analysis.rush_ridge` grids the rush chain (ridge -> Vegas) over ridge strength K and the last-season weight: the surface is flat
(R2 0.043-0.049 for K 8-24, weight 0-0.3). Pass chain is final: ridge -> opponent adjustment -> wind -> Vegas (R2 0.117 weeks 4+).

## Module 1 (final) - `nflfp/team_yards.py`

Expected team rushing and passing yards for a week, the clean version of everything the exploration above settled on:

    rush = ridge average (K=12, 15% last season / 85% league average)  ->  Vegas (yards per point favoured)
    pass = ridge opponent-adjusted fit (K=8, 30% last season)          ->  wind  ->  Vegas

    from nflfp import team_yards as TY, yards as Y
    games = Y.team_game_yards(player_weeks)                  # actual team yards, all seasons needed
    env = TY.environment(schedule, weather.game_weather(schedule))     # per season: fav (spread) and wind5
    TY.project_week(games, env, 2026, 9)                     # rush_exp / pass_exp for every game of the week, from earlier games only

All settings live in `TeamYardsConfig`. `uv run python -m analysis.team_yards_backtest` reproduces the backtest (2025-26, weeks 4+):
rush R2 0.048 (MAE 38.7), pass R2 0.117 (MAE 53.9), against -0.006 / -0.002 for the league average and -0.041 / 0.014 for a team's own average.
`nflfp/yards.py` is the engine for the opponent-adjusted pass fit; `nflfp/plays.py` (plays x yards per play) is kept for later modules.

## Module 2 - player attribution (`nflfp/attribution.py`)

Module 1's team rush and pass yards are split among players by Sleeper's *projected volume* (public feed
`api.sleeper.app/projections/nfl/{season}/{week}`, only `rush_att`, `rec_tgt` and `pass_att` are used - none of Sleeper's yards or points):
a player gets his share of the team's projected carries of the team's rush yards, his share of projected targets of its pass yards
(receiving), and his share of projected attempts of its pass yards (passing - in practice the starting QB). Touchdowns are left out (too noisy);
receptions follow the yards at this season's league rate (receptions per pass yard). Out / IR players are
removed first, so their volume goes to teammates in proportion to the teammates' own volume. This replaces the old heuristic shares in
`model.usage_shares`. Every player of a team gets the same yards per carry / per target (player efficiency showed no predictive value).

`uv run python -m analysis.week_players 2026 9` prints the week's top players. Caveat for backtests: Sleeper's feed for finished weeks is
refreshed after the games, so it knows who actually played; only the current week's volume is a clean pre-game forecast.

## Module 3 - expected team touchdowns (`nflfp/team_tds.py`)

Rush TDs and pass TDs per team-game, each a Poisson regression on the Vegas implied team total (`total_line / 2 + spread / 2`) and module 1's expected
yards for that side, fitted on last season plus this season's earlier weeks and refitted every week (only earlier games). Falls back to the league average
with fewer than 150 usable games. Backtest 2025-26 weeks 4+ (`analysis/team_tds_backtest.py`, 596 team-games): squared error 4.7% (rush) and 10.1% (pass)
below the league average to date; total TDs correlate 0.39 with the actual. `analysis/td_team.py` tried ~25 alternatives (a team's own TD history, opponent
adjustment, TD per yard, distance-aware expected TDs, blends): none beat the Vegas total by a meaningful margin, and team history alone is about half as good.

## Module 4 - touchdowns to players (`nflfp/td_attribution.py`)

Module 3's team TDs are split by shares from the team's own play-by-play history (its games since last season, weighted 0.85 ** games-ago):
rushing TDs by the player's share of the team's rushing *expected TDs* (each carry weighted by the league scoring rate at its yard line, last season's curve),
receiving TDs by target share, passing TDs to the QB by module 2's Sleeper attempts share. Out / IR players are removed first. Backtest
(`analysis/td_player.py`, ~45 variants): the rushing xTD share removes 13% of the squared error against the average candidate's rate (more for the top backs),
receiving target share about 5%; a player's own TD history, red-zone share and fitted combinations added nothing. Per-game player TDs stay mostly noise
(correlation 0.37 rushing, 0.23 receiving), but season totals track well (0.92 / 0.82). Players with no games for their current team (rookies, trades) get no TDs.

`uv run python -m analysis.week_players 2026 9 25 <sleeper user>` runs modules 1-4 and prints expected points (yards + receptions + TDs) next to Sleeper's.

## Black-box player model (`analysis/ml_*.py`)

A second, independent approach: instead of hand-built steps, learn next-game fantasy points per player directly from raw inputs. It is research code under
`analysis/`; the CLI and the league tool still run the older pipeline.

**How it works**
- One row per player-game (QB/RB/WR/TE who played), `analysis/ml_data.py`. Every input comes from games before that one, from the current season and the one
  before it. Players with no NFL game in that window (mostly rookies before their second game) get no estimate.
- Inputs (154 in the working set): the player's recent points, carries, targets, yards, TDs, shares of team volume and snap share over his last 1 / 3 / 8 games,
  recency-weighted, this season and last season; expected TDs and goal-line / red-zone chances from play-by-play; Vegas implied total, spread and game total; the
  team's recent volume; what the opponent allows to the position; availability known at kickoff (teammates' vacated carries and targets, QB change, games he
  missed); the quarterback's recent form and the player's history with him; calendar time (days since his last game, day-weighted averages).
- Models (`analysis/ml_model.py`): ridge regression per position, gradient-boosted trees (squared and Poisson loss), a small neural net, and their average
  ("ensemble"). Trained on 2021-25, tested on 2026 weeks 1-9, a season the models never saw.
- The benchmark is Sleeper's pre-game projection (its feed for past weeks is the pre-game number: week-9 values fetched before the games match the feed afterwards).
- The working projection is a 50-50 blend of the ensemble and Sleeper.

**Results** (2026 weeks 1-9, 1,673 player-games Sleeper projected at 4+ points; "pairs" = pairs of players at a position in a week put in the right order)

| | MAE | corr | pairs |
|---|---|---|---|
| Sleeper | 5.03 | 0.603 | 67.4% |
| Ensemble, 154 inputs | 4.98 | 0.598 | 66.5% |
| 50-50 blend | 4.96 | 0.611 | 67.7% |

The blend is 0.08 MAE better than Sleeper (95% interval -0.12 to -0.03). The ensemble alone is within noise of Sleeper on MAE and slightly worse at ordering
players, mostly running backs. The hand-built modules 1-4 are in the same range (weeks 4-8: MAE 4.97 vs Sleeper 4.99), using Sleeper's volume as an input.

**What moved accuracy, and what did not** (each tested on held-out games; scripts in brackets)
- Helped: training on five seasons instead of one (-0.05 MAE, `ml_multiseason`); blending with Sleeper; quarterback inputs, for QBs only (`ml_qb_ranking`).
- No measurable gain: team run/pass trends, EPA, opposing defense, game environment; situational detail (field position, target depth, down and distance, run
  direction - slightly worse); offensive-line stability (`ml_oline`); adjusted line yards (`ml_addgroup`); calendar time (a gain that disappeared once two data
  bugs were fixed); one model per position (`ml_split`); how a team scores near the goal line and the player's share of it (`ml_scoring_context`); a player's
  history of missing projections; a week-based switch from public to our projections (a flat 50-50 did better).
- Touchdown chances (`ml_td_opps`, `ml_vulture`, `ml_td_adjust`): a back's goal-line chances barely carry forward (correlation 0.16-0.19 with the next game), and
  lead backs who lose goal-line work are not over-projected on average. A TD adjustment (projected TDs vs chances-based TDs) improves Sleeper's RB projections by
  0.04 MAE but adds nothing inside the blend.
- The models lean almost entirely on a player's recent non-TD points, snap share and the Vegas implied total.

**Other tools**: `ml_ranking` (ordering metrics), `ml_diagnose` (why a player was projected where he was), `winprob` (lineup choice by simulated win probability),
`backtest_players` / `compare_roster*` (roster-level comparisons with Sleeper).

    uv run python -m analysis.ml_data         # build the feature table (.cache/ml_features.pkl)
    uv run python -m analysis.ml_addgroup avail+qb avail+qb+cal     # train on 2021-25, score 2026 against Sleeper

**Caveats**: one test season (nine weeks), and about a dozen variants were tried against it, so the best numbers are slightly flattered; availability and
quarterback inputs use who actually played (known at kickoff, not midweek); points ignore interceptions and fumbles.
