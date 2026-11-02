# nflfp - opponent-adjusted expected fantasy points

```
uv run nflfp                      # next unplayed games of the 2026 season
uv run nflfp --week 9 --window 4  # a specific week, usage shares from last 4 games
uv run nflfp --csv out.csv        # full player table
uv run pytest
```

Data comes from [nflverse](https://github.com/nflverse/nflverse-data) (weekly player stats + schedule), cached in `.cache/`.

## Method

1. **Raw**: per team-game, sum rushing yds/TDs and receiving yds/TDs/receptions (passing yds/TDs = receiving).
2. **Opponent adjustment**: for each component, `output = league_avg x OFF[team] x DEF[opp]`. OFF and DEF are
   solved by alternating updates until converged, so a defense that faced strong offenses is credited.
   Small shrinkage toward 1.0 (stronger for TDs) tames small samples.
   `league_avg x DEF[team]` is what that defense would give up to an average offense.
3. **Matchup**: expected output = `league_avg x DEF[opp] x OFF[team]` (the opposing offense's deviation from average;
   `--no-offense-adj` to skip).
4. **Player split**: rushing points by share of team carries, receiving points by share of targets,
   QB passing points by share of pass attempts (shares use the player's current team; `--window N` for recent usage).

Default scoring: 0.1/yd, 6 per rushing/receiving TD, 4 per passing TD, 0.5 PPR (all overridable, see `--help`).

Limitations: no injuries/game script/home field, usage shares are volume-only (not efficiency), no kickers/DST.
