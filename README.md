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

Default scoring: 0.1/yd rushing/receiving, 0.04/yd passing, 6 per rushing/receiving TD, 4 per passing TD, 0.5 PPR (all overridable, see `--help`).

Limitations: no injuries/game script/home field, usage shares are volume-only (not efficiency), no kickers/DST.
