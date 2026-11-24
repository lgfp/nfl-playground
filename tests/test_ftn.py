import pandas as pd
import pytest

from analysis import ftn


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setattr(ftn, "_choice", {"dir": None, "off": False})
    monkeypatch.delenv("FTN_DIR", raising=False)
    monkeypatch.delenv("FTN", raising=False)
    monkeypatch.setattr(ftn, "DIR", tmp_path / "nothing_here")


ROWS = pd.DataFrame({"player_id": ["a", "b"], "season": [2026, 2026], "week": [12, 12], "team": ["SEA", "LA"], "opp": ["LA", "SEA"]})


def test_from_argv_strips_and_remembers():
    assert ftn.from_argv(["prog", "user", "--ftn-dir", "/x/y", "12"]) == ["prog", "user", "12"]
    assert ftn.location().as_posix() == "/x/y"
    assert ftn.from_argv(["prog", "--ftn-dir=/z", "--no-ftn"]) == ["prog"]
    assert ftn.location().as_posix() == "/z" and not ftn.enabled()


def test_missing_exports_leave_everything_unchanged():
    feats, (out,), note = ftn.add(["x"], ROWS)
    assert feats == ["x"] and out is ROWS and "not used" in note and ftn.tag() == ""


def test_switched_off_by_environment(monkeypatch, tmp_path):
    (tmp_path / "Premium-All-Years-DVOA-v1.xlsx").write_text("")
    monkeypatch.setenv("FTN_DIR", str(tmp_path))
    monkeypatch.setenv("FTN", "off")
    feats, _, note = ftn.add(["x"], ROWS)
    assert feats == ["x"] and "switched off" in note


def test_unreadable_export_is_ignored(monkeypatch, tmp_path):
    (tmp_path / "Premium-All-Years-DVOA-v1.xlsx").write_text("not a spreadsheet")
    ftn.from_argv(["--ftn-dir", str(tmp_path)])
    feats, (out,), note = ftn.add(["x"], ROWS)
    assert feats == ["x"] and out is ROWS and "could not read" in note and ftn.tag() == ""


def test_found_exports_add_last_seasons_numbers(monkeypatch, tmp_path):
    (tmp_path / "Premium-All-Years-DVOA-v1.xlsx").write_text("")          # only the team file: player inputs are skipped
    ftn.from_argv(["--ftn-dir", str(tmp_path)])
    teams = pd.DataFrame({"season": [2025, 2025, 2026], "team": ["SEA", "LA", "SEA"], **{c: [0.1, -0.2, 9.9] for c in ftn.TEAM_COLS}})
    monkeypatch.setattr(ftn, "team_seasons", lambda first=2019: teams)
    feats, (out,), note = ftn.add(["x"], ROWS)
    assert "team DVOA" in note and "player" not in note and ftn.tag() == "_ftn"
    assert set(feats) == {"x", *[f"ftn_t_{c}" for c in ftn.TEAM_COLS], *[f"ftn_o_{c}" for c in ftn.TEAM_COLS]}
    assert out.ftn_t_off.tolist() == [0.1, -0.2]                            # 2025 numbers for 2026 games, never 2026's own
    assert out.ftn_o_def.tolist() == [-0.2, 0.1]                            # the opponent's
    assert len(out) == len(ROWS) and (out.index == ROWS.index).all()
