import pandas as pd

from analysis.ml_week import fallback_qbs

GAMES = pd.DataFrame({"team": ["SEA", "SF"], "opp": ["SF", "SEA"], "gameday": ["2026-11-29", "2026-11-29"]})
HIST = pd.DataFrame({"player_id": ["darnold", "lock", "lock", "purdy"], "player_display_name": ["Sam Darnold", "Drew Lock", "Drew Lock", "Brock Purdy"],
                     "season": [2026] * 4, "week": [8, 9, 10, 10], "team": ["SEA", "SEA", "SEA", "SF"], "pass_att": [34.0, 28.0, 17.0, 30.0]})
DUMP = {"4943": {"position": "QB", "first_name": "Sam", "last_name": "Darnold", "team": "SEA", "injury_status": "Out"},
        "5854": {"position": "QB", "first_name": "Drew", "last_name": "Lock", "team": "SEA", "injury_status": None},
        "1": {"position": "QB", "first_name": "Brock", "last_name": "Purdy", "team": "SF", "injury_status": None}}
VOL = pd.DataFrame({"sleeper_id": ["4943", "1"], "name": ["Sam Darnold", "Brock Purdy"], "position": ["QB", "QB"], "team": ["SEA", "SF"],
                    "injury_status": ["Out", None], "attempts": [31.9, 33.0]})


def test_ruled_out_starter_is_replaced_by_the_last_available_starter():
    available = pd.DataFrame({"position": ["QB", "WR"], "team": ["SF", "SEA"], "attempts": [33.0, 0.0]})    # SEA's only projected QB was dropped (Out)
    fb = fallback_qbs(GAMES, available, VOL, HIST, DUMP)
    assert fb.team.tolist() == ["SEA"] and fb.player_id.tolist() == ["lock"] and fb.sleeper_id.tolist() == ["5854"]
    assert fb.attempts.iloc[0] == 31.9                                       # the ruled-out QB's projected attempts
    assert fb.opp.iloc[0] == "SF"


def test_no_fallback_when_every_team_has_a_quarterback():
    available = pd.DataFrame({"position": ["QB", "QB"], "team": ["SF", "SEA"], "attempts": [33.0, 31.9]})
    assert fallback_qbs(GAMES, available, VOL, HIST, DUMP).empty


def test_skips_recent_starters_who_are_ruled_out():
    dump = {**DUMP, "5854": {**DUMP["5854"], "injury_status": "IR"}}
    available = pd.DataFrame({"position": ["QB"], "team": ["SF"], "attempts": [33.0]})
    fb = fallback_qbs(GAMES, available, VOL, HIST, dump)
    assert fb.empty                                                          # Lock (weeks 9-10) on IR, Darnold Out: nobody to use


def test_debuting_rookie_counts_as_the_quarterback():
    vol = pd.concat([VOL, pd.DataFrame({"sleeper_id": ["99"], "name": ["Rookie Qb"], "position": ["QB"], "team": ["SEA"], "injury_status": [None],
                                        "attempts": [30.0]})], ignore_index=True)
    available = pd.DataFrame({"position": ["QB"], "team": ["SF"], "attempts": [33.0]})   # the rookie has no nflverse history, so no mapped row
    assert fallback_qbs(GAMES, available, vol, HIST, DUMP).empty
