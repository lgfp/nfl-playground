from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nflfp import data
from nflfp import ranking_graphic as R

FIXTURE = Path(__file__).parent / "data" / "teams_colors_logos.csv"   # nflverse teams table, trimmed, legacy rows included


@pytest.fixture
def teams(monkeypatch):
    monkeypatch.setattr(data, "_fetch", lambda *a, **k: FIXTURE)
    return data.load_teams()


def solid_logos(entries, size=(300, 200)):
    return {e.team: Image.new("RGBA", size, (0, 0, 255, 255)) for e in entries}


def count(img, rgb, box):
    """Pixels of exactly `rgb` inside box=(x0, y0, x1, y1)."""
    px = np.asarray(img.crop(box).convert("RGB"))
    return int((px == rgb).all(axis=2).sum())


def test_load_teams_drops_legacy_codes(teams):
    assert len(teams) == 32
    assert not {"LA", "OAK", "SD", "STL"} & set(teams.index)
    assert teams.loc["LAR", "team_nick"] == "Rams"


def test_ties_skip_ranks_like_the_example(teams):
    order = ["DEN", "TEN", "TB", ["CIN", "DET"], "NE", "WAS", "KC", "BUF", ["DAL", "NYJ", "PHI"], "MIA", "NYG"]
    got = [(e.rank, e.team, e.tie) for e in R.rank_entries(order, teams)]
    assert got == [(1, "DEN", False), (2, "TEN", False), (3, "TB", False), (4, "CIN", True), (4, "DET", True),
                   (6, "NE", False), (7, "WAS", False), (8, "KC", False), (9, "BUF", False),
                   (10, "DAL", True), (10, "NYJ", True), (10, "PHI", True), (13, "MIA", False), (14, "NYG", False)]


def test_teams_by_abbreviation_nickname_or_full_name(teams):
    entries = R.rank_entries(["den", "Titans", "Tampa Bay Buccaneers", "wsh", "la", "49ers"], teams)
    assert [e.team for e in entries] == ["DEN", "TEN", "TB", "WAS", "LAR", "SF"]


def test_one_team_group_is_not_a_tie(teams):
    assert not R.rank_entries([["DEN"], "TEN"], teams)[0].tie


@pytest.mark.parametrize("order, message", [
    (["DEN", "Denver Broncos"], "twice"),
    (["DEN", "XYZ"], "unknown team 'XYZ'"),
    ([[]], "empty tie group"),
])
def test_bad_orders_are_rejected(teams, order, message):
    with pytest.raises(ValueError, match=message):
        R.rank_entries(order, teams)


def test_parse_order_commas_newlines_comments_and_ties():
    text = "DEN, TEN  # best two\nCIN = DET\n\nDallas Cowboys=NYJ=PHI, MIA"
    assert R.parse_order(text) == ["DEN", "TEN", ["CIN", "DET"], ["Dallas Cowboys", "NYJ", "PHI"], "MIA"]


def test_layout_fills_columns_top_to_bottom_and_uses_style_colors(teams):
    entries = R.rank_entries([t for t in teams.index], teams)          # 32 teams -> 4 columns of 8
    style = R.Style(background="#102030", number_color="#ff0000", width=2000, height=1900)
    img = R.draw_ranking(entries, solid_logos(entries), style)
    assert img.size == (2000, 1900)
    assert img.getpixel((5, 5))[:3] == (0x10, 0x20, 0x30)
    cw, ch = (2000 - 120) / 4, (1900 - 114) / 8
    for col in range(4):                                               # every column has number pixels at its left, logo pixels right
        x0 = 60 + col * cw
        assert count(img, (255, 0, 0), (int(x0), 57, int(x0 + 0.3 * cw), int(57 + ch))) > 200
        assert count(img, (0, 0, 255), (int(x0 + 0.3 * cw), 57, int(x0 + cw), int(57 + ch))) > 1000
    assert count(img, (0, 0, 255), (0, 0, 60, 1900)) == 0              # nothing in the left padding


def test_tie_label_uses_its_own_color_and_only_appears_for_ties(teams):
    entries = R.rank_entries([["DEN", "TEN"], "TB"], teams)
    style = R.Style(number_color="#ff0000", tie_color="#00ff00")
    img = R.draw_ranking(entries, solid_logos(entries), style)
    cw, ch = (style.width - 2 * 0.03 * style.width) / 1, (style.height - 2 * 0.03 * style.height) / style.rows
    top, x1 = 0.03 * style.height, int(0.03 * style.width + 0.3 * cw)
    row = lambda r: (0, int(top + r * ch), x1, int(top + (r + 1) * ch))
    assert count(img, (0, 255, 0), row(0)) > 100 and count(img, (0, 255, 0), row(1)) > 100   # the two tied teams
    assert count(img, (0, 255, 0), row(2)) == 0                                              # TB is alone: no TIE
    assert count(img, (255, 0, 0), row(2)) > 200


def test_tie_text_is_customizable(teams):
    entries = R.rank_entries([["DEN", "TEN"]], teams)
    a = R.draw_ranking(entries, solid_logos(entries), R.Style(tie_text="TIE"))
    b = R.draw_ranking(entries, solid_logos(entries), R.Style(tie_text="TIED"))
    assert a.tobytes() != b.tobytes()


def test_transparent_background(teams):
    entries = R.rank_entries(["DEN"], teams)
    img = R.draw_ranking(entries, solid_logos(entries), R.Style(background="transparent"))
    assert img.getpixel((5, 5))[3] == 0


def test_background_image_fills_the_canvas(teams, tmp_path):
    Image.new("RGB", (50, 20), (1, 2, 3)).save(tmp_path / "bg.png")
    entries = R.rank_entries(["DEN"], teams)
    img = R.draw_ranking(entries, solid_logos(entries), R.Style(background_image=str(tmp_path / "bg.png")))
    assert img.getpixel((1990, 1890))[:3] == (1, 2, 3)


def test_fewer_rows_means_more_columns(teams):
    entries = R.rank_entries(list(teams.index)[:8], teams)
    img = R.draw_ranking(entries, solid_logos(entries), R.Style(rows=4, number_color="#ff0000"))
    right_half = (1000, 0, 2000, 1900)
    assert count(img, (255, 0, 0), right_half) > 200                   # a second column exists


def test_font_from_path_and_unknown_font():
    default = R.load_font(None, 40)
    if not getattr(default, "path", None):
        pytest.skip("no TrueType font installed; only Pillow's built-in font is available")
    assert R.load_font(default.path, 40).path == default.path          # a file path works
    with pytest.raises(ValueError, match="not found"):
        R.load_font("No Such Font Family 123", 40)


def test_bad_color_is_a_clear_error(teams):
    entries = R.rank_entries(["DEN"], teams)
    with pytest.raises(ValueError, match="unknown color 'nope'"):
        R.draw_ranking(entries, solid_logos(entries), R.Style(number_color="nope"))


def test_logo_dir_and_overrides_win_over_downloads(teams, tmp_path):
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(tmp_path / "den.png")
    Image.new("RGBA", (10, 10), (0, 255, 0, 255)).save(tmp_path / "custom.png")
    style = R.Style(logo_dir=str(tmp_path), logo_overrides={"TEN": str(tmp_path / "custom.png")})
    assert R.load_logo("DEN", teams, style).getpixel((0, 0)) == (255, 0, 0, 255)      # file name case does not matter
    assert R.load_logo("TEN", teams, style).getpixel((0, 0)) == (0, 255, 0, 255)


def test_unreachable_logo_host_says_how_to_fix_it(teams, monkeypatch):
    def fail(*a, **k):
        raise OSError("blocked")
    monkeypatch.setattr(data, "_fetch", fail)
    with pytest.raises(RuntimeError, match="logo_dir"):
        R.load_logo("DEN", teams, R.Style())


def test_jpeg_output_is_flattened(teams, tmp_path):
    entries = R.rank_entries(["DEN"], teams)
    img = R.draw_ranking(entries, solid_logos(entries), R.Style(background="transparent"))
    R.save(img, tmp_path / "out.jpg")
    assert Image.open(tmp_path / "out.jpg").mode == "RGB"
