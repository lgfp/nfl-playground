import pandas as pd

from nflfp import league_card as lc


def test_pct_formats():
    assert [lc.pct(p) for p in (0, 0.0004, 0.001, 0.3868, 0.982, 0.99999, 1)] == ["0%", "<0.1%", "0.1%", "38.7%", "98.2%", ">99.9%", "100%"]


def test_card_html_has_league_name_columns_and_row_order():
    t = pd.DataFrame({"team": ["Rio & Co", "B"], "record": ["6-4", "8-2"], "exp_wins": [8.71, 10.06], "p_playoffs": [0.954, 0.982], "p_champion": [0.387, 0.276]})
    h = lc.card_html("Chesterton's Bane", t, "After week 10", ["line one", "line two"])
    assert "Chesterton&#x27;s Bane" in h and "Rio &amp; Co" in h
    assert h.index("Rio &amp; Co") < h.index(">B<")                         # row order is the order given
    assert all(x in h for x in ("Proj. wins", "Playoff odds", "Championship odds", "8.7", "95.4%", "27.6%", "line one<br>line two"))
    assert "track" not in h and "fill" not in h                           # a plain table, no bars


def test_render_writes_html_without_a_browser(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "find_browser", lambda: None)
    t = pd.DataFrame({"team": ["A"], "record": ["1-0"], "exp_wins": [1.0], "p_playoffs": [1.0], "p_champion": [0.5]})
    out = lc.render("L", t, "s", "f", tmp_path / "x.png")
    assert out.suffix == ".html" and "100%" in out.read_text()
