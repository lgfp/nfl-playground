# nflfp - opponent-adjusted expected fantasy points

```
uv run nflfp                      # next unplayed games of the 2026 season
uv run nflfp --week 9 --window 4  # a specific week, usage shares from last 4 games
uv run nflfp --csv out.csv        # full player table
uv run pytest
uv run python -m nflfp.backtest           # backtest team rush vs receiving predictions (train on all other weeks; --forward = earlier weeks only)
```

Data comes from [nflverse](https://github.com/nflverse/nflverse-data) (weekly player stats + schedule), cached in `.cache/`.

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
