"""A shareable image of the league odds: league name on top, then one ranked row per team with record, projected wins, playoff odds and championship odds.

`card_html` builds a self-contained page (no scripts, no network); `render` screenshots it with headless Chromium / Chrome if one is installed
(set CHROME_BIN to point at it) and otherwise writes the .html next to the requested path, which opens in any browser.
"""
from __future__ import annotations

import html
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

WIDTH, PAD, ROW, HEAD = 1000, 48, 58, 40
TITLE_BLOCK, FOOT_BLOCK = 118, 84
CSS = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; }
* { box-sizing:border-box; margin:0; padding:0; }
body { width:%(w)dpx; background:var(--surface); color:var(--ink); padding:%(pad)dpx; font-family:Inter, system-ui, -apple-system, "Segoe UI", sans-serif; -webkit-font-smoothing:antialiased; }
h1 { font-size:46px; line-height:1.1; font-weight:700; letter-spacing:-.02em; }
.kind { font-size:21px; font-weight:600; color:var(--ink2); margin-top:10px; }
.sub { font-size:15px; color:var(--muted); margin-top:6px; }
table { width:100%%; border-collapse:collapse; margin-top:26px; table-layout:fixed; }
th { height:%(head)dpx; text-align:right; font-size:12px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--grid); vertical-align:middle; white-space:nowrap; }
th.l { text-align:left; }
td { height:%(row)dpx; border-bottom:1px solid var(--grid); vertical-align:middle; text-align:right; font-size:19px; font-variant-numeric:tabular-nums; color:var(--ink2); }
td.rank { text-align:left; color:var(--muted); font-size:15px; }
td.team { text-align:left; font-size:20px; font-weight:600; letter-spacing:-.01em; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding-right:12px; }
td.strong { color:var(--ink); font-weight:600; }
td.champ { color:var(--ink); font-weight:700; }
.foot { margin-top:22px; font-size:13px; line-height:1.5; color:var(--muted); }
"""


def pct(p: float) -> str:
    """Probability as text: whole percent at the extremes, one decimal between, '<0.1%' rather than a misleading 0."""
    v = 100 * p
    if p <= 0:
        return "0%"
    if p >= 1:
        return "100%"
    if v < 0.05:
        return "<0.1%"
    if v > 99.95:
        return ">99.9%"
    return f"{v:.1f}%"


def card_html(league: str, table: pd.DataFrame, subtitle: str, footer: str | list[str]) -> str:
    """`table` needs team, record, exp_wins, p_playoffs, p_champion; rows are printed in the order given."""
    rows = "".join(f'<tr><td class="rank">{i}</td><td class="team">{html.escape(str(r.team))}</td><td>{html.escape(str(r.record))}</td>'
                   f'<td class="strong">{r.exp_wins:.1f}</td><td class="strong">{pct(r.p_playoffs)}</td><td class="champ">{pct(r.p_champion)}</td></tr>'
                   for i, r in enumerate(table.itertuples(index=False), 1))
    head = ('<tr><th class="l" style="width:36px"></th><th class="l">Team</th><th style="width:96px">Record</th><th style="width:130px">Proj. wins</th>'
            '<th style="width:140px">Playoff odds</th><th style="width:190px">Championship odds</th></tr>')
    foot = "<br>".join(html.escape(x) for x in ([footer] if isinstance(footer, str) else footer))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(league)} - odds</title>'
            f'<style>{CSS % dict(w=WIDTH, pad=PAD, row=ROW, head=HEAD)}</style></head><body>'
            f'<h1>{html.escape(league)}</h1><div class="kind">Playoff &amp; championship odds</div><div class="sub">{html.escape(subtitle)}</div>'
            f'<table><thead>{head}</thead><tbody>{rows}</tbody></table><div class="foot">{foot}</div></body></html>')


def height(n_rows: int) -> int:
    return 2 * PAD + TITLE_BLOCK + HEAD + n_rows * ROW + FOOT_BLOCK


def find_browser() -> str | None:
    for c in (os.environ.get("CHROME_BIN"), "/opt/pw-browsers/chromium", shutil.which("chromium"), shutil.which("chromium-browser"),
              shutil.which("google-chrome"), shutil.which("google-chrome-stable"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if c and Path(c).exists():
            return c
    return None


def render(league: str, table: pd.DataFrame, subtitle: str, footer: str | list[str], out: str | Path, scale: int = 2) -> Path:
    """Writes the PNG (or, with no browser available, the .html) and returns the path written."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = card_html(league, table, subtitle, footer)
    browser = find_browser()
    if browser is None or out.suffix.lower() == ".html":
        out = out.with_suffix(".html")
        out.write_text(page)
        return out
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "card.html"
        f.write_text(page)
        r = subprocess.run([browser, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", f"--force-device-scale-factor={scale}",
                            f"--window-size={WIDTH},{height(len(table))}", f"--screenshot={out.resolve()}", f.as_uri()], capture_output=True, text=True, timeout=120)
    if not out.exists():
        raise RuntimeError("screenshot failed: " + r.stderr[-300:])
    return out
